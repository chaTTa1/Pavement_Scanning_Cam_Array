# -*- coding: utf-8 -*-
"""
GPS/IMU Fusion — pure numpy/scipy implementation (no gtsam dependency)
Drop-in replacement for the GTSAM version.

Created on Wed Apr 29 10:33:25 2026
@author: Desktop
"""

import math
import argparse
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Spyder/F5 defaults for PPC Tokyo dataset.
SPYDER_PPC_RUN_DIR = os.path.join(SCRIPT_DIR, "tokyo", "run1")
SPYDER_OUTPUT_CSV = os.path.join(SPYDER_PPC_RUN_DIR, "factor_graph_output.csv")
SPYDER_DURATION_S = 300.0          # use None for full run
SPYDER_KEYFRAME_STRIDE = 1         # 1 = every reference point, 5 = every 5th


# ──────────────────────────────────────────────────────────────
# Coordinate conversions (identical to original)
# ──────────────────────────────────────────────────────────────

def geodetic_to_ecef(lat_deg, lon_deg, h):
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n = a / math.sqrt(1.0 - e2 * sin_lat ** 2)
    x = (n + h) * cos_lat * cos_lon
    y = (n + h) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + h) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_enu_matrix(lat0_deg, lon0_deg):
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    sin_lat, cos_lat = math.sin(lat0), math.cos(lat0)
    sin_lon, cos_lon = math.sin(lon0), math.cos(lon0)
    return np.array([
        [-sin_lon,            cos_lon,           0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat]
    ], dtype=float)


def gps_dataframe_to_enu(gps_df):
    lat0 = float(gps_df.iloc[0]["lat"])
    lon0 = float(gps_df.iloc[0]["lon"])
    h0   = float(gps_df.iloc[0]["height"])
    ecef0 = geodetic_to_ecef(lat0, lon0, h0)
    r = ecef_to_enu_matrix(lat0, lon0)
    enu_points = []
    for _, row in gps_df.iterrows():
        ecef = geodetic_to_ecef(float(row["lat"]), float(row["lon"]), float(row["height"]))
        enu_points.append(r @ (ecef - ecef0))
    enu_points = np.asarray(enu_points)
    gps_df = gps_df.copy()
    gps_df["e"] = enu_points[:, 0]
    gps_df["n"] = enu_points[:, 1]
    gps_df["u"] = enu_points[:, 2]
    return gps_df


def estimate_initial_yaw_from_gps(gps_df):
    if len(gps_df) < 2:
        return 0.0
    p0 = np.array([float(gps_df.iloc[0]["e"]), float(gps_df.iloc[0]["n"])])
    for i in range(1, len(gps_df)):
        p1 = np.array([float(gps_df.iloc[i]["e"]), float(gps_df.iloc[i]["n"])])
        d = p1 - p0
        if np.linalg.norm(d) > 0.3:
            return math.atan2(d[0], d[1])
    return 0.0


# ──────────────────────────────────────────────────────────────
# Rotation helpers using scipy
# ──────────────────────────────────────────────────────────────

def ypr_to_rotation(yaw, pitch, roll):
    """Return 3x3 rotation matrix from yaw-pitch-roll (ZYX intrinsic)."""
    return Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()


def rotation_to_ypr(R):
    """Return (yaw, pitch, roll) from 3x3 rotation matrix."""
    r = Rotation.from_matrix(R)
    angles = r.as_euler('ZYX')
    return angles[0], angles[1], angles[2]


def rotvec_to_matrix(rv):
    return Rotation.from_rotvec(rv).as_matrix()


def matrix_to_rotvec(R):
    return Rotation.from_matrix(R).as_rotvec()


# ──────────────────────────────────────────────────────────────
# IMU pre-integration (simplified, first-order)
# ──────────────────────────────────────────────────────────────

def preintegrate_imu(imu_rows, accel_bias, gyro_bias, gravity_vec):
    """
    Simple first-order IMU pre-integration between two GPS epochs.
    Returns:
        delta_R : 3x3 rotation change
        delta_v : 3-vector velocity change (in body-start frame)
        delta_p : 3-vector position change (in body-start frame)
        dt_total: total time
    """
    delta_R = np.eye(3)
    delta_v = np.zeros(3)
    delta_p = np.zeros(3)
    dt_total = 0.0

    if len(imu_rows) < 2:
        return delta_R, delta_v, delta_p, dt_total

    imu_rows = imu_rows.sort_values("timestamp").reset_index(drop=True)
    last_t = float(imu_rows.iloc[0]["timestamp"])

    for i in range(1, len(imu_rows)):
        row = imu_rows.iloc[i]
        current_t = float(row["timestamp"])
        dt = current_t - last_t
        if dt <= 0.0:
            last_t = current_t
            continue

        acc = np.array([float(row["ax"]), float(row["ay"]), float(row["az"])]) - accel_bias
        omega = np.array([float(row["gx"]), float(row["gy"]), float(row["gz"])]) - gyro_bias

        # Update position, velocity, rotation (first-order)
        delta_p += delta_v * dt + 0.5 * delta_R @ acc * dt * dt
        delta_v += delta_R @ acc * dt
        delta_R = delta_R @ rotvec_to_matrix(omega * dt)

        dt_total += dt
        last_t = current_t

    return delta_R, delta_v, delta_p, dt_total


# ──────────────────────────────────────────────────────────────
# State packing / unpacking
# ──────────────────────────────────────────────────────────────
# Per-epoch state: [rot_vec(3), pos(3), vel(3)] = 9 floats
# Global bias state: [accel_bias(3), gyro_bias(3)] = 6 floats
# Total: N*9 + 6

STATE_DIM = 9
BIAS_DIM = 6


def pack_state(rotations, positions, velocities, accel_bias, gyro_bias):
    N = len(rotations)
    x = np.zeros(N * STATE_DIM + BIAS_DIM)
    for i in range(N):
        x[i * STATE_DIM: i * STATE_DIM + 3] = matrix_to_rotvec(rotations[i])
        x[i * STATE_DIM + 3: i * STATE_DIM + 6] = positions[i]
        x[i * STATE_DIM + 6: i * STATE_DIM + 9] = velocities[i]
    x[N * STATE_DIM: N * STATE_DIM + 3] = accel_bias
    x[N * STATE_DIM + 3: N * STATE_DIM + 6] = gyro_bias
    return x


def unpack_state(x, N):
    rotations = []
    positions = []
    velocities = []
    for i in range(N):
        rv = x[i * STATE_DIM: i * STATE_DIM + 3]
        rotations.append(rotvec_to_matrix(rv))
        positions.append(x[i * STATE_DIM + 3: i * STATE_DIM + 6].copy())
        velocities.append(x[i * STATE_DIM + 6: i * STATE_DIM + 9].copy())
    accel_bias = x[N * STATE_DIM: N * STATE_DIM + 3].copy()
    gyro_bias = x[N * STATE_DIM + 3: N * STATE_DIM + 6].copy()
    return rotations, positions, velocities, accel_bias, gyro_bias


# ──────────────────────────────────────────────────────────────
# Residual function for least-squares
# ──────────────────────────────────────────────────────────────

def build_residual_function(gps_enu, gps_timestamps, imu_df, config):
    """
    Build a residual vector that encodes:
      1. GPS position factors
      2. IMU pre-integration factors (position, velocity, rotation)
      3. Prior on first pose, velocity, bias
      4. Bias random-walk (small change between epochs)
    """
    N = len(gps_enu)
    gravity_vec = np.array([0.0, 0.0, -config["gravity"]])

    # Pre-compute IMU segments
    imu_segments = []
    for k in range(1, N):
        t0 = gps_timestamps[k - 1]
        t1 = gps_timestamps[k]
        mask = (imu_df["timestamp"] >= t0) & (imu_df["timestamp"] <= t1)
        imu_segments.append(imu_df.loc[mask].copy())

    # Weights (inverse sigma)
    w_gps = 1.0 / np.array([
        config["gps_position_sigma_e"],
        config["gps_position_sigma_n"],
        config["gps_position_sigma_u"]
    ])

    w_prior_rot = 1.0 / config["pose_prior_rotation_sigma"]
    w_prior_pos = 1.0 / config["pose_prior_position_sigma"]
    w_prior_vel = 1.0 / config["velocity_prior_sigma"]
    w_prior_accel_bias = 1.0 / config["bias_prior_accel_sigma"]
    w_prior_gyro_bias = 1.0 / config["bias_prior_gyro_sigma"]

    w_imu_pos = 1.0 / config["accel_noise_sigma"]
    w_imu_vel = 1.0 / config["accel_noise_sigma"]
    w_imu_rot = 1.0 / config["gyro_noise_sigma"]

    # Initial values for prior
    initial_yaw = estimate_initial_yaw_from_gps(
        pd.DataFrame({"e": gps_enu[:, 0], "n": gps_enu[:, 1]})
    )
    prior_pos = gps_enu[0]
    prior_rot = ypr_to_rotation(initial_yaw, 0.0, 0.0)
    prior_rot_rv = matrix_to_rotvec(prior_rot)

    if N >= 2:
        dt01 = gps_timestamps[1] - gps_timestamps[0]
        if dt01 > 0:
            prior_vel = (gps_enu[1] - gps_enu[0]) / dt01
        else:
            prior_vel = np.zeros(3)
    else:
        prior_vel = np.zeros(3)

    def residuals(x):
        rotations, positions, velocities, accel_bias, gyro_bias = unpack_state(x, N)
        res = []

        # ── Prior on first state ──
        rot_err = matrix_to_rotvec(prior_rot.T @ rotations[0])
        res.extend((rot_err * w_prior_rot).tolist())
        res.extend(((positions[0] - prior_pos) * w_prior_pos).tolist())
        res.extend(((velocities[0] - prior_vel) * w_prior_vel).tolist())

        # ── Prior on bias ──
        res.extend((accel_bias * w_prior_accel_bias).tolist())
        res.extend((gyro_bias * w_prior_gyro_bias).tolist())

        # ── GPS factors ──
        for k in range(N):
            pos_err = positions[k] - gps_enu[k]
            res.extend((pos_err * w_gps).tolist())

        # ── IMU factors ──
        for k in range(1, N):
            dt = gps_timestamps[k] - gps_timestamps[k - 1]
            if dt <= 0:
                continue

            delta_R, delta_v, delta_p, dt_total = preintegrate_imu(
                imu_segments[k - 1], accel_bias, gyro_bias, gravity_vec
            )

            if dt_total <= 0:
                continue

            R_prev = rotations[k - 1]
            p_prev = positions[k - 1]
            v_prev = velocities[k - 1]
            R_curr = rotations[k]
            p_curr = positions[k]
            v_curr = velocities[k]

            # Position residual
            predicted_p = p_prev + v_prev * dt_total + 0.5 * gravity_vec * dt_total ** 2 + R_prev @ delta_p
            p_res = (p_curr - predicted_p) * w_imu_pos
            res.extend(p_res.tolist())

            # Velocity residual
            predicted_v = v_prev + gravity_vec * dt_total + R_prev @ delta_v
            v_res = (v_curr - predicted_v) * w_imu_vel
            res.extend(v_res.tolist())

            # Rotation residual
            predicted_R = R_prev @ delta_R
            rot_res = matrix_to_rotvec(predicted_R.T @ R_curr) * w_imu_rot
            res.extend(rot_res.tolist())

        return np.array(res)

    return residuals


# ──────────────────────────────────────────────────────────────
# Main fusion function
# ──────────────────────────────────────────────────────────────

def run_gps_imu_fusion(imu_csv, gps_csv, output_csv):
    config = {
        "gravity": 9.81,
        "gps_position_sigma_e": 0.03,
        "gps_position_sigma_n": 0.03,
        "gps_position_sigma_u": 0.06,
        "pose_prior_position_sigma": 0.05,
        "pose_prior_rotation_sigma": 0.10,
        "velocity_prior_sigma": 0.50,
        "bias_prior_accel_sigma": 0.10,
        "bias_prior_gyro_sigma": 0.01,
        "accel_noise_sigma": 0.10,
        "gyro_noise_sigma": 0.01,
        "integration_sigma": 1e-4,
        "accel_bias_rw_sigma": 0.0001,
        "gyro_bias_rw_sigma": 0.00001,
        "optimizer_max_iterations": 200,
    }

    imu_df = pd.read_csv(imu_csv)
    gps_df = pd.read_csv(gps_csv)

    required_imu_cols = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz"]
    required_gps_cols = ["timestamp", "lat", "lon", "height"]
    for col in required_imu_cols:
        if col not in imu_df.columns:
            raise ValueError(f"imu.csv missing column: {col}")
    for col in required_gps_cols:
        if col not in gps_df.columns:
            raise ValueError(f"gps.csv missing column: {col}")

    imu_df = imu_df.sort_values("timestamp").reset_index(drop=True)
    gps_df = gps_df.sort_values("timestamp").reset_index(drop=True)
    gps_df = gps_dataframe_to_enu(gps_df)

    N = len(gps_df)
    gps_enu = gps_df[["e", "n", "u"]].values.astype(float)
    gps_timestamps = gps_df["timestamp"].values.astype(float)

    # ── Build initial guess ──
    initial_yaw = estimate_initial_yaw_from_gps(gps_df)
    rotations_init = [ypr_to_rotation(initial_yaw, 0.0, 0.0)] * N
    positions_init = [gps_enu[k].copy() for k in range(N)]
    velocities_init = []
    for k in range(N):
        if k == 0 and N >= 2:
            dt = gps_timestamps[1] - gps_timestamps[0]
            v = (gps_enu[1] - gps_enu[0]) / dt if dt > 0 else np.zeros(3)
        elif k > 0:
            dt = gps_timestamps[k] - gps_timestamps[k - 1]
            v = (gps_enu[k] - gps_enu[k - 1]) / dt if dt > 0 else np.zeros(3)
        else:
            v = np.zeros(3)
        velocities_init.append(v)

    accel_bias_init = np.zeros(3)
    gyro_bias_init = np.zeros(3)

    x0 = pack_state(rotations_init, positions_init, velocities_init,
                     accel_bias_init, gyro_bias_init)

    # ── Build and solve ──
    residual_fn = build_residual_function(gps_enu, gps_timestamps, imu_df, config)

    print(f"Optimizing {N} states ({len(x0)} variables)...")
    result = least_squares(
        residual_fn, x0,
        method='trf',
        max_nfev=config["optimizer_max_iterations"] * len(x0),
        verbose=0
    )

    rotations, positions, velocities, accel_bias, gyro_bias = unpack_state(result.x, N)

    # ── Output ──
    output_rows = []
    for k in range(N):
        yaw, pitch, roll = rotation_to_ypr(rotations[k])
        output_rows.append({
            "timestamp": gps_timestamps[k],
            "east_m": positions[k][0],
            "north_m": positions[k][1],
            "up_m": positions[k][2],
            "velocity_e_mps": velocities[k][0],
            "velocity_n_mps": velocities[k][1],
            "velocity_u_mps": velocities[k][2],
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "accel_bias_x": accel_bias[0],
            "accel_bias_y": accel_bias[1],
            "accel_bias_z": accel_bias[2],
            "gyro_bias_x": gyro_bias[0],
            "gyro_bias_y": gyro_bias[1],
            "gyro_bias_z": gyro_bias[2],
            "gps_lat": float(gps_df.iloc[k]["lat"]),
            "gps_lon": float(gps_df.iloc[k]["lon"]),
            "gps_height": float(gps_df.iloc[k]["height"]),
        })

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv, index=False)

    print("Done")
    print(f"Saved result to {output_csv}")
    print(f"Number of optimized states: {len(output_rows)}")
    print(f"Optimizer cost: {result.cost:.6f}")
    print(f"Optimizer status: {result.message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imu_csv")
    parser.add_argument("gps_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    run_gps_imu_fusion(args.imu_csv, args.gps_csv, args.output_csv)


if __name__ == "__main__":
    run_gps_imu_fusion(
        r"D:\Ryan\GitHub\paper\Pavement_Scanning_Cam_Array\Pavement_Scanning_Cam_Array\data\imu.csv",
        r"D:\Ryan\GitHub\paper\Pavement_Scanning_Cam_Array\Pavement_Scanning_Cam_Array\data\gps.csv",
        r"D:\Ryan\GitHub\paper\Pavement_Scanning_Cam_Array\Pavement_Scanning_Cam_Array\data\fused_output.csv"
    )
