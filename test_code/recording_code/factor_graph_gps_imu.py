"""Batch factor-graph GPS/IMU fusion.

This script is intentionally separate from EKF_only.py. It reads the same
recorder-format CSVs:

  imu_raw.csv, imu_gravity.csv, imu_quat.csv, gps_raw.csv

and builds a lightweight factor graph with:

  - keyframe state: [east, north, up, ve, vn, vu]
  - IMU preintegration factors between adjacent GPS position keyframes
  - GPS position factors from GGA
  - optional GPS velocity factors from RMC/VTG

It is not a full GTSAM-style inertial graph with optimized attitude and IMU
bias states. It is a practical first batch optimizer for validating time
alignment and GPS/IMU consistency using your current CSV format.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Spyder-friendly configuration
# ---------------------------------------------------------------------------

# Use converted PPC input if it exists; otherwise point at the PPC run folder
# and auto-convert it first.
DEFAULT_PPC_RUN_DIR = SCRIPT_DIR / "tokyo" / "run1"
DEFAULT_DATA_DIR = (
    DEFAULT_PPC_RUN_DIR / "ekf_input_full"
    if (DEFAULT_PPC_RUN_DIR / "ekf_input_full" / "imu_raw.csv").exists()
    else DEFAULT_PPC_RUN_DIR
)
DEFAULT_OUTPUT_DIR = DEFAULT_PPC_RUN_DIR / "factor_graph_output"

AUTO_CONVERT_PPC = True
FACTOR_GRAPH_DURATION_S = 300    # use None for full run
KEYFRAME_STRIDE = 1                # 1 = every GPS position, 5 = every 5th
INTERP_DT_S = 0.01                 # points between GPS keyframes for plots/CSV

GRAVITY_MPS2 = 9.80665
GPS_TIME_OFFSET_S = 0.0
MIN_GPS_FIX_QUALITY = 1

# Measurement and factor weights.
GPS_POS_SIGMA_RTK_FIX = 0.10
GPS_POS_SIGMA_RTK_FLOAT = 0.30
GPS_POS_SIGMA_DGPS = 0.70
GPS_POS_SIGMA_STANDALONE = 3.0
MIN_GPS_POS_SIGMA = 0.02
GPS_ALT_SCALE = 2.0
GPS_VEL_SIGMA = 0.20

# IMU factor sigmas are deliberately conservative because this graph does not
# optimize attitude or IMU biases.
IMU_POS_SIGMA = 0.35
IMU_VEL_SIGMA = 0.35
PRIOR_POS_SIGMA = 0.05
PRIOR_VEL_SIGMA = 0.50

MAX_IMU_DT = 0.05
MAX_OPT_NFEV = 800
IMU_TIMESTAMP_WRAP_US = 2 ** 32
MAX_IMU_TIMESTAMP_JUMP_S = 10.0


M_PER_DEG_LAT = 111_132.92
M_PER_DEG_LON_EQ = 111_319.49


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factor-graph GPS/IMU fusion.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--duration-s", type=float, default=FACTOR_GRAPH_DURATION_S)
    parser.add_argument("--keyframe-stride", type=int, default=KEYFRAME_STRIDE)
    parser.add_argument("--no-show", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring non-factor-graph args: {unknown}")
    if args.data_dir is None:
        args.data_dir = DEFAULT_DATA_DIR
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR
    return args


def maybe_convert_ppc(data_dir: Path, duration_s: Optional[float]) -> Path:
    if (data_dir / "imu_raw.csv").exists() and (data_dir / "gps_raw.csv").exists():
        return data_dir

    if not ((data_dir / "imu.csv").exists() and (data_dir / "reference.csv").exists()):
        return data_dir

    if not AUTO_CONVERT_PPC:
        return data_dir

    from convert_ppc_to_ekf_input import (
        f as ppc_float,
        read_rows,
        write_imu_raw,
        write_reference_derived_files,
    )

    label = "full" if duration_s is None else f"{duration_s:g}s"
    converted_dir = data_dir / f"ekf_input_{label}"
    converted_dir.mkdir(parents=True, exist_ok=True)

    imu_rows = read_rows(data_dir / "imu.csv")
    reference_rows = read_rows(data_dir / "reference.csv")
    start_tow = ppc_float(imu_rows[0], "GPS TOW (s)")
    end_tow = None if duration_s is None else start_tow + duration_s

    print("[INFO] PPC folder detected; converting to EKF CSV format...")
    write_imu_raw(imu_rows, converted_dir, start_tow, end_tow)
    write_reference_derived_files(
        reference_rows,
        converted_dir,
        start_tow,
        end_tow,
        GPS_POS_SIGMA_RTK_FIX,
        0.2,
    )
    return converted_dir


def m_per_deg_lon(lat_deg: float) -> float:
    return M_PER_DEG_LON_EQ * math.cos(math.radians(lat_deg))


def gps_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    return np.array(
        [
            (lon - ref_lon) * m_per_deg_lon(ref_lat),
            (lat - ref_lat) * M_PER_DEG_LAT,
            alt - ref_alt,
        ],
        dtype=float,
    )


def enu_to_gps(enu, ref_lat, ref_lon, ref_alt):
    lat = ref_lat + enu[1] / M_PER_DEG_LAT
    lon = ref_lon + enu[0] / m_per_deg_lon(ref_lat)
    alt = ref_alt + enu[2]
    return lat, lon, alt


def qnorm(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_rotate(q_xyzw, v):
    q = qnorm(np.asarray(q_xyzw, dtype=float))
    qv = q[:3]
    w = q[3]
    v = np.asarray(v, dtype=float)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def imu_quat_to_xyzw(q1, q2, q3, q4):
    return np.array([q2, q3, q4, q1], dtype=float)


def load_imu_raw(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "imu_raw.csv"
    df = pd.read_csv(path)
    for col in [
        "wall_time",
        "imu_timestamp_us",
        "gyro_x_rad_s",
        "gyro_y_rad_s",
        "gyro_z_rad_s",
        "accel_x_g",
        "accel_y_g",
        "accel_z_g",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["wall_time", "imu_timestamp_us", "accel_x_g"])
    return df.sort_values("wall_time").reset_index(drop=True)


def load_optional_csv(data_dir: Path, name: str) -> Optional[pd.DataFrame]:
    path = data_dir / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in df.columns:
        if col != "msg_type" and col != "rmc_status" and col != "raw_nmea":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("wall_time").reset_index(drop=True)


def compute_imu_dt(imu_raw: pd.DataFrame) -> pd.DataFrame:
    ts = imu_raw["imu_timestamp_us"].values.astype(np.int64)
    wall = imu_raw["wall_time"].values.astype(float)
    dt = np.full(len(ts), np.nan, dtype=float)
    max_jump_us = int(MAX_IMU_TIMESTAMP_JUMP_S * 1e6)

    for i in range(1, len(ts)):
        raw_diff = int(ts[i] - ts[i - 1])
        wall_diff = wall[i] - wall[i - 1]
        diff = raw_diff

        if raw_diff < 0:
            wrapped = raw_diff + IMU_TIMESTAMP_WRAP_US
            wall_ok = np.isfinite(wall_diff) and 0.0 < wall_diff <= MAX_IMU_TIMESTAMP_JUMP_S
            wrap_matches_wall = wall_ok and abs(wrapped / 1e6 - wall_diff) < 1.0
            if 0 < wrapped <= max_jump_us and (not wall_ok or wrap_matches_wall):
                diff = wrapped
            else:
                diff = 0

        if 0 < diff <= max_jump_us:
            dt[i] = diff / 1e6
        elif np.isfinite(wall_diff) and 0.0 < wall_diff <= MAX_IMU_TIMESTAMP_JUMP_S:
            dt[i] = wall_diff

    if len(dt) > 1:
        valid = dt[1:][np.isfinite(dt[1:]) & (dt[1:] > 0)]
        fill_dt = np.median(valid[: min(len(valid), 200)]) if len(valid) else 0.01
        dt[0] = fill_dt
        dt[~np.isfinite(dt)] = fill_dt
    elif len(dt) == 1:
        dt[0] = 0.01
    imu_raw = imu_raw.copy()
    imu_raw["dt"] = dt
    return imu_raw


def merge_gravity_and_quat(imu_raw, imu_gravity, imu_quat):
    imu = imu_raw.copy()
    raw_t = imu["wall_time"].values

    if imu_gravity is not None and len(imu_gravity) > 0:
        gt = imu_gravity["wall_time"].values
        idx = np.searchsorted(gt, raw_t, side="right") - 1
        idx = np.clip(idx, 0, len(gt) - 1)
        for col in ["gravity_x_g", "gravity_y_g", "gravity_z_g"]:
            imu[col] = imu_gravity[col].values[idx]
    else:
        imu["gravity_x_g"] = 0.0
        imu["gravity_y_g"] = 0.0
        imu["gravity_z_g"] = -1.0
        print("[WARNING] No imu_gravity.csv; assuming fixed gravity.")

    if imu_quat is not None and len(imu_quat) > 0:
        qt = imu_quat["wall_time"].values
        idx = np.searchsorted(qt, raw_t, side="right") - 1
        idx = np.clip(idx, 0, len(qt) - 1)
        imu["ahrs_q1"] = imu_quat["q1"].values[idx]
        imu["ahrs_q2"] = imu_quat["q2"].values[idx]
        imu["ahrs_q3"] = imu_quat["q3"].values[idx]
        imu["ahrs_q4"] = imu_quat["q4"].values[idx]
    else:
        imu["ahrs_q1"] = np.nan
        imu["ahrs_q2"] = np.nan
        imu["ahrs_q3"] = np.nan
        imu["ahrs_q4"] = np.nan
        print("[WARNING] No imu_quat.csv; IMU acceleration cannot be rotated.")

    return imu


def add_enu_acceleration(imu: pd.DataFrame) -> pd.DataFrame:
    acc_body = imu[["accel_x_g", "accel_y_g", "accel_z_g"]].values.astype(float)
    grav_body = imu[["gravity_x_g", "gravity_y_g", "gravity_z_g"]].values.astype(float)
    lin_body = (acc_body - grav_body) * GRAVITY_MPS2
    acc_enu = lin_body.copy()

    quat_cols = ["ahrs_q1", "ahrs_q2", "ahrs_q3", "ahrs_q4"]
    if all(col in imu.columns for col in quat_cols):
        q_wxyz = imu[quat_cols].values.astype(float)
        valid = np.isfinite(q_wxyz).all(axis=1)
        q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
        q_norm = np.linalg.norm(q_xyzw, axis=1)
        valid &= q_norm > 1e-12
        if np.any(valid):
            q_xyzw_valid = q_xyzw[valid] / q_norm[valid, None]
            acc_enu[valid] = Rot.from_quat(q_xyzw_valid).apply(lin_body[valid])

    imu = imu.copy()
    imu["acc_e"] = acc_enu[:, 0]
    imu["acc_n"] = acc_enu[:, 1]
    imu["acc_u"] = acc_enu[:, 2]
    return imu


def load_gps_measurements(data_dir: Path):
    gps_raw = pd.read_csv(data_dir / "gps_raw.csv")
    if "msg_type" not in gps_raw.columns:
        raise RuntimeError("gps_raw.csv must contain a msg_type column.")

    for col in [
        "wall_time",
        "lat",
        "lon",
        "alt",
        "fix_quality",
        "lat_err_m",
        "lon_err_m",
        "alt_err_m",
        "speed_knots",
        "speed_kmh",
        "course_true_deg",
    ]:
        if col in gps_raw.columns:
            gps_raw[col] = pd.to_numeric(gps_raw[col], errors="coerce")

    msg_type = gps_raw["msg_type"].astype(str).str.upper()

    gga = gps_raw[msg_type == "GGA"].dropna(subset=["wall_time", "lat", "lon"])
    if "fix_quality" in gga.columns:
        gga = gga[gga["fix_quality"].fillna(0).astype(int) >= MIN_GPS_FIX_QUALITY]
    else:
        gga = gga.iloc[0:0]

    if len(gga):
        fq = gga["fix_quality"].astype(int).values
        sigma = np.select(
            [fq == 4, fq == 5, fq == 2],
            [
                GPS_POS_SIGMA_RTK_FIX,
                GPS_POS_SIGMA_RTK_FLOAT,
                GPS_POS_SIGMA_DGPS,
            ],
            default=GPS_POS_SIGMA_STANDALONE,
        ).astype(float)
        if "lat_err_m" in gga.columns and "lon_err_m" in gga.columns:
            lat_err = gga["lat_err_m"].values.astype(float)
            lon_err = gga["lon_err_m"].values.astype(float)
            err_valid = np.isfinite(lat_err) & np.isfinite(lon_err)
            err_valid &= (lat_err > 0.0) & (lon_err > 0.0)
            sigma[err_valid] = np.maximum(
                np.maximum(lat_err[err_valid], lon_err[err_valid]),
                MIN_GPS_POS_SIGMA,
            )

        pos_df = pd.DataFrame({
            "t": gga["wall_time"].values.astype(float) + GPS_TIME_OFFSET_S,
            "lat": gga["lat"].values.astype(float),
            "lon": gga["lon"].values.astype(float),
            "alt": gga["alt"].fillna(0.0).values.astype(float)
            if "alt" in gga.columns else np.zeros(len(gga)),
            "sigma_h": sigma,
            "sigma_v": sigma * GPS_ALT_SCALE,
        })
        pos = pos_df.sort_values("t").to_dict("records")
    else:
        pos = []

    vel_frames = []
    if "speed_knots" in gps_raw.columns and "course_true_deg" in gps_raw.columns:
        rmc = gps_raw[msg_type == "RMC"].dropna(
            subset=["wall_time", "speed_knots", "course_true_deg"]
        )
        if "rmc_status" in rmc.columns:
            rmc = rmc[rmc["rmc_status"].fillna("A").astype(str).str.strip() != "V"]
        if len(rmc):
            speed = rmc["speed_knots"].values.astype(float) * 0.514444
            course = np.deg2rad(rmc["course_true_deg"].values.astype(float))
            keep = np.isfinite(speed) & np.isfinite(course) & (speed >= 0.05)
            vel_frames.append(pd.DataFrame({
                "t": rmc["wall_time"].values.astype(float)[keep] + GPS_TIME_OFFSET_S,
                "ve": speed[keep] * np.sin(course[keep]),
                "vn": speed[keep] * np.cos(course[keep]),
            }))

    if "speed_kmh" in gps_raw.columns and "course_true_deg" in gps_raw.columns:
        vtg = gps_raw[msg_type == "VTG"].dropna(
            subset=["wall_time", "speed_kmh", "course_true_deg"]
        )
        if len(vtg):
            speed = vtg["speed_kmh"].values.astype(float) / 3.6
            course = np.deg2rad(vtg["course_true_deg"].values.astype(float))
            keep = np.isfinite(speed) & np.isfinite(course) & (speed >= 0.05)
            vel_frames.append(pd.DataFrame({
                "t": vtg["wall_time"].values.astype(float)[keep] + GPS_TIME_OFFSET_S,
                "ve": speed[keep] * np.sin(course[keep]),
                "vn": speed[keep] * np.cos(course[keep]),
            }))

    if vel_frames:
        vel = pd.concat(vel_frames, ignore_index=True).sort_values("t")
        vel = vel.to_dict("records")
    else:
        vel = []

    return pos, vel


def nearest_velocity(vel_meas, t, max_dt=0.11):
    if not vel_meas:
        return None
    times = np.array([m["t"] for m in vel_meas])
    idx = int(np.searchsorted(times, t))
    candidates = []
    if idx < len(times):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(times[i] - t))
    if abs(times[best] - t) > max_dt:
        return None
    return vel_meas[best]


def preintegrate_interval(imu_t, imu_acc, ti, tj):
    idx0 = int(np.searchsorted(imu_t, ti, side="left"))
    idx1 = int(np.searchsorted(imu_t, tj, side="left"))
    dp = np.zeros(3)
    dv = np.zeros(3)
    if idx0 >= idx1:
        return dp, dv, tj - ti

    for k in range(idx0, idx1):
        t0 = max(imu_t[k], ti)
        if k + 1 < len(imu_t):
            t1 = min(imu_t[k + 1], tj)
        else:
            t1 = tj
        dt = t1 - t0
        if dt <= 0 or dt > MAX_IMU_DT:
            continue
        a = imu_acc[k]
        dp += dv * dt + 0.5 * a * dt * dt
        dv += a * dt
    return dp, dv, tj - ti


def build_keyframes(gps_pos, vel_meas, ref_lat, ref_lon, ref_alt, duration_s, stride):
    if duration_s is not None:
        t0 = gps_pos[0]["t"]
        gps_pos = [m for m in gps_pos if m["t"] <= t0 + duration_s]

    gps_pos = gps_pos[:: max(stride, 1)]
    if len(gps_pos) < 2:
        raise RuntimeError("Need at least two GPS position keyframes.")

    keyframes = []
    for i, m in enumerate(gps_pos):
        p = gps_to_enu(m["lat"], m["lon"], m["alt"], ref_lat, ref_lon, ref_alt)
        v_meas = nearest_velocity(vel_meas, m["t"], max_dt=0.25 * max(stride, 1))
        if v_meas is not None:
            v = np.array([v_meas["ve"], v_meas["vn"], 0.0])
        elif i > 0:
            prev = keyframes[-1]
            dt = m["t"] - prev["t"]
            v = (p - prev["p_gps"]) / dt if dt > 0 else np.zeros(3)
        else:
            v = np.zeros(3)
        keyframes.append(
            {
                "t": m["t"],
                "p_gps": p,
                "v_gps": v,
                "has_vel": v_meas is not None,
                "sigma_h": m["sigma_h"],
                "sigma_v": m["sigma_v"],
                "lat": m["lat"],
                "lon": m["lon"],
                "alt": m["alt"],
            }
        )
    return keyframes


def initial_state(keyframes):
    x = np.zeros(6 * len(keyframes))
    for i, kf in enumerate(keyframes):
        x[6 * i: 6 * i + 3] = kf["p_gps"]
        x[6 * i + 3: 6 * i + 6] = kf["v_gps"]
    return x


def build_imu_factors(keyframes, imu):
    imu_t = imu["wall_time"].values.astype(float)
    imu_acc = imu[["acc_e", "acc_n", "acc_u"]].values.astype(float)
    factors = []
    for i in range(len(keyframes) - 1):
        ti = keyframes[i]["t"]
        tj = keyframes[i + 1]["t"]
        dp, dv, dt = preintegrate_interval(imu_t, imu_acc, ti, tj)
        factors.append({"i": i, "j": i + 1, "dt": dt, "dp": dp, "dv": dv})
    return factors


def make_residual_function(keyframes, imu_factors):
    def residual(x):
        res = []

        p0 = x[0:3]
        v0 = x[3:6]
        res.extend((p0 - keyframes[0]["p_gps"]) / PRIOR_POS_SIGMA)
        res.extend((v0 - keyframes[0]["v_gps"]) / PRIOR_VEL_SIGMA)

        for fac in imu_factors:
            i = fac["i"]
            j = fac["j"]
            xi = x[6 * i: 6 * i + 6]
            xj = x[6 * j: 6 * j + 6]
            pi = xi[0:3]
            vi = xi[3:6]
            pj = xj[0:3]
            vj = xj[3:6]
            pred_pj = pi + vi * fac["dt"] + fac["dp"]
            pred_vj = vi + fac["dv"]
            res.extend((pj - pred_pj) / IMU_POS_SIGMA)
            res.extend((vj - pred_vj) / IMU_VEL_SIGMA)

        for i, kf in enumerate(keyframes):
            xk = x[6 * i: 6 * i + 6]
            res.extend((xk[0:2] - kf["p_gps"][0:2]) / kf["sigma_h"])
            res.append((xk[2] - kf["p_gps"][2]) / kf["sigma_v"])
            if kf["has_vel"]:
                res.extend((xk[3:5] - kf["v_gps"][0:2]) / GPS_VEL_SIGMA)

        return np.asarray(res, dtype=float)

    return residual


def make_jac_sparsity(keyframes, imu_factors, expected_n_res=None):
    n_var = 6 * len(keyframes)
    n_res = 6 + 6 * len(imu_factors) + 3 * len(keyframes)
    n_res += 2 * sum(1 for kf in keyframes if kf["has_vel"])
    sp = lil_matrix((n_res, n_var), dtype=int)

    row = 0
    sp[row: row + 6, 0:6] = 1
    row += 6

    for fac in imu_factors:
        i0 = 6 * fac["i"]
        j0 = 6 * fac["j"]
        sp[row: row + 6, i0: i0 + 6] = 1
        sp[row: row + 6, j0: j0 + 6] = 1
        row += 6

    for i, kf in enumerate(keyframes):
        i0 = 6 * i
        sp[row: row + 3, i0: i0 + 3] = 1
        row += 3
        if kf["has_vel"]:
            sp[row: row + 2, i0 + 3: i0 + 5] = 1
            row += 2

    if row != n_res:
        raise RuntimeError(
            f"Jacobian sparsity row count mismatch: filled {row}, expected {n_res}"
        )
    if expected_n_res is not None and n_res != expected_n_res:
        raise RuntimeError(
            "Jacobian sparsity/residual mismatch: "
            f"sparsity has {n_res} rows, residual has {expected_n_res}"
        )

    return sp.tocsr()


def save_results(x, keyframes, output_dir, ref_lat, ref_lon, ref_alt):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "position_factor_graph.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wall_time", "lat", "lon", "alt", "ve", "vn", "vu"])
        for i, kf in enumerate(keyframes):
            state = x[6 * i: 6 * i + 6]
            lat, lon, alt = enu_to_gps(state[0:3], ref_lat, ref_lon, ref_alt)
            writer.writerow(
                [
                    f"{kf['t']:.6f}",
                    f"{lat:.10f}",
                    f"{lon:.10f}",
                    f"{alt:.4f}",
                    f"{state[3]:.6f}",
                    f"{state[4]:.6f}",
                    f"{state[5]:.6f}",
                ]
            )
    print(f"Saved factor graph trajectory: {path}")


def hermite_position(pi, vi, pj, vj, dt, tau):
    """Cubic Hermite interpolation from optimized endpoint p/v states."""
    h00 = 2.0 * tau ** 3 - 3.0 * tau ** 2 + 1.0
    h10 = tau ** 3 - 2.0 * tau ** 2 + tau
    h01 = -2.0 * tau ** 3 + 3.0 * tau ** 2
    h11 = tau ** 3 - tau ** 2
    return h00 * pi + h10 * dt * vi + h01 * pj + h11 * dt * vj


def hermite_velocity(pi, vi, pj, vj, dt, tau):
    """Velocity is the time derivative of hermite_position."""
    if dt <= 0:
        return vi.copy()
    dh00 = 6.0 * tau ** 2 - 6.0 * tau
    dh10 = 3.0 * tau ** 2 - 4.0 * tau + 1.0
    dh01 = -6.0 * tau ** 2 + 6.0 * tau
    dh11 = 3.0 * tau ** 2 - 2.0 * tau
    return (dh00 * pi + dh01 * pj) / dt + dh10 * vi + dh11 * vj


def build_interpolated_results(x, keyframes, ref_lat, ref_lon, ref_alt):
    """
    Create multiple factor-graph trajectory points between GPS keyframes.

    The optimizer only estimates states at GPS keyframes. For visualization,
    this uses cubic Hermite interpolation with optimized endpoint position and
    velocity, so each GPS interval contains many factor-graph points.
    """
    rows = []
    if len(keyframes) == 0:
        return rows

    for i in range(len(keyframes) - 1):
        xi = x[6 * i: 6 * i + 6]
        xj = x[6 * (i + 1): 6 * (i + 1) + 6]
        ti = keyframes[i]["t"]
        tj = keyframes[i + 1]["t"]
        dt = tj - ti
        if dt <= 0:
            continue

        n_steps = max(2, int(math.ceil(dt / INTERP_DT_S)))
        for s in range(n_steps):
            tau = s / n_steps
            t = ti + tau * dt
            p = hermite_position(xi[0:3], xi[3:6], xj[0:3], xj[3:6], dt, tau)
            v = hermite_velocity(xi[0:3], xi[3:6], xj[0:3], xj[3:6], dt, tau)
            lat, lon, alt = enu_to_gps(p, ref_lat, ref_lon, ref_alt)
            rows.append({
                "wall_time": t,
                "source": "interpolated",
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "ve": v[0],
                "vn": v[1],
                "vu": v[2],
            })

    last = x[6 * (len(keyframes) - 1): 6 * len(keyframes)]
    lat, lon, alt = enu_to_gps(last[0:3], ref_lat, ref_lon, ref_alt)
    rows.append({
        "wall_time": keyframes[-1]["t"],
        "source": "interpolated",
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "ve": last[3],
        "vn": last[4],
        "vu": last[5],
    })
    return rows


def save_interpolated_results(interp_rows, output_dir):
    path = output_dir / "position_factor_graph_interpolated.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["wall_time", "source", "lat", "lon", "alt", "ve", "vn", "vu"],
        )
        writer.writeheader()
        for row in interp_rows:
            writer.writerow(row)
    print(f"Saved interpolated factor graph trajectory: {path}")


def heading_from_quat_row(row):
    q = imu_quat_to_xyzw(row["q1"], row["q2"], row["q3"], row["q4"])
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        return np.nan
    R = Rot.from_quat(q / q_norm).as_matrix()
    forward = R @ np.array([1.0, 0.0, 0.0])
    return math.degrees(math.atan2(forward[0], forward[1])) % 360.0


def course_from_velocity(ve, vn):
    return math.degrees(math.atan2(ve, vn)) % 360.0


def unwrap_heading_deg(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    out = np.full(len(arr), np.nan, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out

    valid_idx = np.flatnonzero(valid)
    valid_values = arr[valid]
    diff = np.diff(valid_values)
    diff[diff > 180.0] -= 360.0
    diff[diff < -180.0] += 360.0
    out[valid_idx] = np.concatenate(
        [[valid_values[0]], valid_values[0] + np.cumsum(diff)]
    )
    return out


def load_heading_series_for_plot(data_dir, keyframes):
    """Load GPS/reference heading and IMU quaternion heading for comparison."""
    gps_headings = []
    imu_headings = []

    gps_path = Path(data_dir) / "gps_raw.csv"
    if gps_path.exists():
        gps_raw = pd.read_csv(gps_path)
        for col in ["wall_time", "heading_deg", "course_true_deg",
                    "speed_knots", "speed_kmh"]:
            if col in gps_raw.columns:
                gps_raw[col] = pd.to_numeric(gps_raw[col], errors="coerce")

        if "heading_deg" in gps_raw.columns:
            hdt = gps_raw[gps_raw["msg_type"] == "HDT"].dropna(
                subset=["heading_deg"])
            for _, row in hdt.iterrows():
                gps_headings.append({
                    "t": float(row["wall_time"]) + GPS_TIME_OFFSET_S,
                    "heading": float(row["heading_deg"]) % 360.0,
                    "source": "HDT/reference",
                })

        if not gps_headings and "course_true_deg" in gps_raw.columns:
            course_rows = gps_raw[
                gps_raw["msg_type"].isin(["RMC", "VTG"])
            ].dropna(subset=["course_true_deg"])
            for _, row in course_rows.iterrows():
                gps_headings.append({
                    "t": float(row["wall_time"]) + GPS_TIME_OFFSET_S,
                    "heading": float(row["course_true_deg"]) % 360.0,
                    "source": "course",
                })

    quat_path = Path(data_dir) / "imu_quat.csv"
    if quat_path.exists():
        imu_quat = pd.read_csv(quat_path)
        for col in ["wall_time", "q1", "q2", "q3", "q4"]:
            imu_quat[col] = pd.to_numeric(imu_quat[col], errors="coerce")
        imu_quat = imu_quat.dropna(subset=["wall_time", "q1", "q2", "q3", "q4"])
        if len(imu_quat) > 0:
            q_times = imu_quat["wall_time"].values
            target_times = [kf["t"] for kf in keyframes]
            for t in target_times:
                idx = np.searchsorted(q_times, t, side="right") - 1
                if idx < 0:
                    continue
                idx = min(idx, len(imu_quat) - 1)
                row = imu_quat.iloc[idx]
                imu_headings.append({
                    "t": t,
                    "heading": heading_from_quat_row(row),
                })

    gps_headings.sort(key=lambda m: m["t"])
    imu_headings.sort(key=lambda m: m["t"])
    return gps_headings, imu_headings


def plot_results(x, keyframes, interp_rows, output_dir, ref_lat, ref_lon,
                 ref_alt, data_dir):
    fg_lats = []
    fg_lons = []
    fg_alts = []
    gps_lats = []
    gps_lons = []
    times = []
    speeds = []

    for i, kf in enumerate(keyframes):
        state = x[6 * i: 6 * i + 6]
        lat, lon, alt = enu_to_gps(state[0:3], ref_lat, ref_lon, ref_alt)
        fg_lats.append(lat)
        fg_lons.append(lon)
        fg_alts.append(alt)
        gps_lats.append(kf["lat"])
        gps_lons.append(kf["lon"])
        times.append(kf["t"] - keyframes[0]["t"])
        speeds.append(math.hypot(state[3], state[4]))

    interp_lats = [row["lat"] for row in interp_rows]
    interp_lons = [row["lon"] for row in interp_rows]
    interp_times = [row["wall_time"] - keyframes[0]["t"] for row in interp_rows]
    interp_speeds = [math.hypot(row["ve"], row["vn"]) for row in interp_rows]
    interp_enu = np.array([
        gps_to_enu(row["lat"], row["lon"], row["alt"], ref_lat, ref_lon, ref_alt)
        for row in interp_rows
    ])
    fg_enu = np.array([
        gps_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
        for lat, lon, alt in zip(fg_lats, fg_lons, fg_alts)
    ])
    gps_enu = np.array([
        gps_to_enu(kf["lat"], kf["lon"], kf["alt"], ref_lat, ref_lon, ref_alt)
        for kf in keyframes
    ])

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.canvas.manager.set_window_title("Factor Graph Trajectory")
    ax.plot(gps_lons, gps_lats, "b.", markersize=4, label="GPS/reference")
    ax.plot(interp_lons, interp_lats, ".", color="seagreen", markersize=1.2,
            alpha=0.6, label="Factor graph interpolated")
    ax.plot(fg_lons, fg_lats, "o", color="darkorange", markersize=3,
            label="Factor graph keyframes")
    ax.scatter(gps_lons[0], gps_lats[0], c="green", s=80, marker="^",
               label="Start", zorder=5)
    ax.scatter(gps_lons[-1], gps_lats[-1], c="red", s=80, marker="s",
               label="End", zorder=5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Factor Graph GPS/IMU Trajectory with Interpolated Points")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.ticklabel_format(useOffset=False, style="plain")
    plt.tight_layout()
    fig.savefig(output_dir / "1_factor_graph_trajectory.png", dpi=150,
                bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.canvas.manager.set_window_title("Factor Graph Speed")
    ax.plot(interp_times, interp_speeds, ".", color="seagreen", markersize=1.2,
            alpha=0.6, label="Interpolated")
    ax.plot(times, speeds, "o", color="darkorange", markersize=3,
            label="Keyframes")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Factor Graph Speed")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "2_factor_graph_speed.png", dpi=150,
                bbox_inches="tight")

    # ENU top-view in meters.
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.canvas.manager.set_window_title("Factor Graph ENU Top View")
    ax.plot(gps_enu[:, 0], gps_enu[:, 1], "b.", markersize=4,
            label="GPS/reference keyframes")
    ax.plot(interp_enu[:, 0], interp_enu[:, 1], ".", color="seagreen",
            markersize=1.2, alpha=0.6, label="Factor graph interpolated")
    ax.plot(fg_enu[:, 0], fg_enu[:, 1], "o", color="darkorange",
            markersize=3, label="Factor graph keyframes")
    ax.scatter(gps_enu[0, 0], gps_enu[0, 1], c="green", s=80,
               marker="^", zorder=5, label="Start")
    ax.scatter(gps_enu[-1, 0], gps_enu[-1, 1], c="red", s=80,
               marker="s", zorder=5, label="End")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Top View in ENU Meters")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "3_top_view_enu_meters.png", dpi=150,
                bbox_inches="tight")

    # 3D point view in meters.
    fig = plt.figure(figsize=(12, 9))
    fig.canvas.manager.set_window_title("Factor Graph 3D ENU")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gps_enu[:, 0], gps_enu[:, 1], gps_enu[:, 2],
               c="blue", s=8, label="GPS/reference keyframes")
    ax.scatter(interp_enu[:, 0], interp_enu[:, 1], interp_enu[:, 2],
               c="seagreen", s=2, alpha=0.55,
               label="Factor graph interpolated")
    ax.scatter(fg_enu[:, 0], fg_enu[:, 1], fg_enu[:, 2],
               c="darkorange", s=10, label="Factor graph keyframes")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title("3D ENU Points")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "4_3d_enu_points.png", dpi=150,
                bbox_inches="tight")

    # IMU heading vs GPS/reference heading.
    gps_hdg, imu_hdg = load_heading_series_for_plot(data_dir, keyframes)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.canvas.manager.set_window_title("IMU Heading vs GPS Heading")
    if imu_hdg:
        imu_t = [m["t"] - keyframes[0]["t"] for m in imu_hdg]
        imu_heading_unwrapped = unwrap_heading_deg(
            [m["heading"] for m in imu_hdg]
        )
        ax.plot(imu_t,
                imu_heading_unwrapped,
                ".", color="purple", markersize=2, label="IMU quaternion heading")
    if gps_hdg:
        gps_t = [m["t"] - keyframes[0]["t"] for m in gps_hdg]
        gps_heading_unwrapped = unwrap_heading_deg(
            [m["heading"] for m in gps_hdg]
        )
        ax.plot(gps_t,
                gps_heading_unwrapped,
                ".", color="steelblue", markersize=3, alpha=0.7,
                label="GPS/reference heading")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unwrapped heading (deg)")
    ax.set_title("IMU Heading vs GPS/Reference Heading (Unwrapped)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "5_heading_imu_vs_gps.png", dpi=150,
                bbox_inches="tight")


def main():
    args = parse_args()
    data_dir = maybe_convert_ppc(Path(args.data_dir), args.duration_s)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Factor Graph GPS/IMU Fusion")
    print("=" * 60)
    print(f"Data dir:      {data_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Duration:      {args.duration_s if args.duration_s else 'full'}")
    print(f"Key stride:    {args.keyframe_stride}")

    imu_raw = load_imu_raw(data_dir)
    imu_raw = compute_imu_dt(imu_raw)
    imu_gravity = load_optional_csv(data_dir, "imu_gravity.csv")
    imu_quat = load_optional_csv(data_dir, "imu_quat.csv")
    imu = merge_gravity_and_quat(imu_raw, imu_gravity, imu_quat)
    imu = add_enu_acceleration(imu)

    gps_pos, gps_vel = load_gps_measurements(data_dir)
    if not gps_pos:
        raise RuntimeError("No valid GPS GGA position measurements found.")

    ref_lat = gps_pos[0]["lat"]
    ref_lon = gps_pos[0]["lon"]
    ref_alt = gps_pos[0]["alt"]

    keyframes = build_keyframes(
        gps_pos,
        gps_vel,
        ref_lat,
        ref_lon,
        ref_alt,
        args.duration_s,
        args.keyframe_stride,
    )
    print(f"Keyframes:     {len(keyframes)}")
    print(f"GPS vel used:  {sum(1 for kf in keyframes if kf['has_vel'])}")

    imu_factors = build_imu_factors(keyframes, imu)
    print(f"IMU factors:   {len(imu_factors)}")

    x0 = initial_state(keyframes)
    fun = make_residual_function(keyframes, imu_factors)
    r0 = fun(x0)
    sparsity = make_jac_sparsity(
        keyframes, imu_factors, expected_n_res=len(r0)
    )
    print(f"Initial cost:  {0.5 * float(r0 @ r0):.3f}")
    result = least_squares(
        fun,
        x0,
        jac_sparsity=sparsity,
        loss="huber",
        f_scale=1.0,
        max_nfev=MAX_OPT_NFEV,
        verbose=2,
        x_scale="jac",
    )

    r1 = fun(result.x)
    print(f"Final cost:    {0.5 * float(r1 @ r1):.3f}")
    print(f"Success:       {result.success}")
    print(f"Message:       {result.message}")

    save_results(result.x, keyframes, output_dir, ref_lat, ref_lon, ref_alt)
    interp_rows = build_interpolated_results(
        result.x, keyframes, ref_lat, ref_lon, ref_alt)
    save_interpolated_results(interp_rows, output_dir)
    plot_results(
        result.x, keyframes, interp_rows, output_dir,
        ref_lat, ref_lon, ref_alt, data_dir)

    print("=" * 60)
    print("DONE")
    print("=" * 60)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
