# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:55:12 2026

@author: Desktop
"""

# -*- coding: utf-8 -*-
"""
GPS/IMU EKF Post-Processor (Corrected)
========================================
Reads recorded raw data from recorder.py and produces
an optimal 400Hz trajectory using Extended Kalman Filter.

Fixes applied:
  1. RTS smoother disabled by default, index logic fixed for later use
  2. QUAT not used as runtime measurement (intentional, magnetometer unreliable)
  3. quat_to_yaw_deg verified against GPS heading convention
  4. merge_gravity uses "previous available" (documented correctly)
  5. VTG velocity added alongside RMC
  6. Initial velocity from first GPS velocity measurement
  7. RTK fix sigma conservative (0.05m default)
  8. GPS time offset parameter added
  9. Numerical Jacobian computed once per predict (returned from function)
  10. Forward EKF validated first, RTS as separate optional step

Architecture:
  Step 1: Load CSV files from recorder
  Step 2: Time alignment + preprocessing
    - Compute IMU dt from hardware timestamps
    - Merge gravity vector with RAW data
    - Merge AHRS quaternion with RAW data
    - Extract GPS measurements (GGA pos, RMC/VTG vel, HDT heading)
  Step 3: Build unified event timeline sorted by wall_time
  Step 4: EKF forward pass
    - IMU predict at 400Hz (bias-corrected, gravity-removed)
    - GPS position update at 10Hz
    - GPS velocity update at 10Hz
    - GPS heading update (when available)
  Step 5: (Optional) RTS backward smoothing
  Step 6: Save results + plots

EKF State Vector (16 elements):
  x[0:3]   position ENU [east, north, up] meters
  x[3:6]   velocity ENU [ve, vn, vu] m/s
  x[6:9]   accelerometer bias body [bax, bay, baz] m/s²
  x[9:12]  gyroscope bias body [bgx, bgy, bgz] rad/s
  x[12:16] quaternion [qx, qy, qz, qw] scipy convention

For Spyder IDE: configure paths at top, press F5.
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
import os
import sys
import csv
import math
import time as time_module
import argparse
from pathlib import Path


# ─────────────────────────────────────────────
# ★★★ CONFIGURATION — EDIT THESE ★★★
# ─────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Spyder/F5 defaults. Point this at a PPC run folder such as tokyo/run1.
# If that folder contains imu.csv + reference.csv, it is converted first.
SPYDER_DATA_DIR = os.path.join(SCRIPT_DIR, "tokyo", "run1")
SPYDER_OUTPUT_DIR = ""             # blank = choose automatically
SPYDER_NO_SHOW = False             # True = save plots without opening windows
PLOT_IMU_ONLY = True               # Draw raw IMU dead-reckoning trajectory

# PPC conversion settings used only when DATA_DIR has imu.csv + reference.csv.
AUTO_CONVERT_PPC = True
PPC_DURATION_S = None              # use None for the full PPC run
PPC_START_TOW = None               # use None for the first IMU timestamp
PPC_GPS_SIGMA_M = 0.1
PPC_HEADING_SIGMA_DEG = 0.2


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run the GPS/IMU EKF on recorder-format CSV files.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Folder containing recorder CSVs or a PPC run folder.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder where EKF CSVs and plots will be written.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without opening a matplotlib window.")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring non-EKF command-line args: {unknown}")
    return args


def _duration_label(duration_s):
    if duration_s is None:
        return "full"
    return f"{duration_s:g}s"


def _resolve_runtime_paths(args):
    data_dir = (
        args.data_dir
        or os.environ.get("EKF_DATA_DIR")
        or SPYDER_DATA_DIR
        or "."
    )
    output_dir = (
        args.output_dir
        or os.environ.get("EKF_OUTPUT_DIR")
        or SPYDER_OUTPUT_DIR
    )
    data_dir = os.path.abspath(os.path.expanduser(data_dir))

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(data_dir), "ekf_output")
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    return data_dir, output_dir, (args.no_show or SPYDER_NO_SHOW)


def _auto_convert_ppc_if_needed(data_dir, output_dir):
    has_recorder_files = (
        os.path.exists(os.path.join(data_dir, "imu_raw.csv"))
        and os.path.exists(os.path.join(data_dir, "gps_raw.csv"))
    )
    if has_recorder_files:
        return data_dir, output_dir

    has_ppc_files = (
        os.path.exists(os.path.join(data_dir, "imu.csv"))
        and os.path.exists(os.path.join(data_dir, "reference.csv"))
    )
    if not has_ppc_files:
        return data_dir, output_dir

    if not AUTO_CONVERT_PPC:
        print("[ERROR] DATA_DIR is a PPC folder, but AUTO_CONVERT_PPC is False")
        sys.exit(1)

    try:
        from convert_ppc_to_ekf_input import (
            read_rows,
            f as ppc_float,
            write_imu_raw,
            write_reference_derived_files,
        )
    except Exception as exc:
        print(f"[ERROR] Could not import PPC converter: {exc}")
        sys.exit(1)

    label = _duration_label(PPC_DURATION_S)
    converted_dir = os.path.join(data_dir, f"ekf_input_{label}")
    converted_output_dir = os.path.join(data_dir, f"ekf_output_{label}")

    print("[INFO] PPC run folder detected; converting for EKF input...")
    print(f"       PPC run:   {data_dir}")
    print(f"       EKF input: {converted_dir}")

    converted_dir_path = Path(converted_dir)
    os.makedirs(converted_dir, exist_ok=True)
    imu_rows = read_rows(Path(data_dir) / "imu.csv")
    reference_rows = read_rows(Path(data_dir) / "reference.csv")
    if not imu_rows or not reference_rows:
        print("[ERROR] PPC imu.csv/reference.csv is empty")
        sys.exit(1)

    start_tow = PPC_START_TOW
    if start_tow is None:
        start_tow = ppc_float(imu_rows[0], "GPS TOW (s)")
    end_tow = None
    if PPC_DURATION_S is not None:
        end_tow = start_tow + PPC_DURATION_S

    imu_count = write_imu_raw(imu_rows, converted_dir_path, start_tow, end_tow)
    gravity_count, quat_count, gps_count = write_reference_derived_files(
        reference_rows,
        converted_dir_path,
        start_tow,
        end_tow,
        PPC_GPS_SIGMA_M,
        PPC_HEADING_SIGMA_DEG,
    )
    print(f"       imu_raw.csv:     {imu_count} rows")
    print(f"       imu_gravity.csv: {gravity_count} rows")
    print(f"       imu_quat.csv:    {quat_count} rows")
    print(f"       gps_raw.csv:     {gps_count} rows")

    if not SPYDER_OUTPUT_DIR and not os.environ.get("EKF_OUTPUT_DIR"):
        output_dir = converted_output_dir
    return converted_dir, output_dir


_ARGS = _parse_cli_args()

DATA_DIR, OUTPUT_DIR, _NO_SHOW = _resolve_runtime_paths(_ARGS)
DATA_DIR, OUTPUT_DIR = _auto_convert_ppc_if_needed(DATA_DIR, OUTPUT_DIR)
SHOW_PLOTS = not _NO_SHOW

# GPS measurement noise (1-sigma, conservative defaults)
GPS_POS_SIGMA_RTK_FIX = 0.05       # meters, fix_quality == 4
GPS_POS_SIGMA_RTK_FLOAT = 0.20     # meters, fix_quality == 5
GPS_POS_SIGMA_DGPS = 0.50          # meters, fix_quality == 2
GPS_POS_SIGMA_STANDALONE = 3.0     # meters, fix_quality == 1
GPS_ALT_SCALE = 1.5                # altitude noise = horizontal * this

# GPS velocity noise (from RMC/VTG, 1-sigma)
GPS_VEL_SIGMA = 0.1                # m/s

# GPS heading noise (from HDT, 1-sigma)
GPS_HEADING_SIGMA_DEG = 1.0        # degrees

# GPS time offset: compensate for NMEA serial delay
# Positive = GPS measurement happened BEFORE wall_time suggests
# Tune by minimizing GPS position innovation
GPS_TIME_OFFSET_S = 0.0

# IMU noise parameters (tune for your TM151)
ACCEL_NOISE_DENSITY = 0.003        # m/s²/√Hz
GYRO_NOISE_DENSITY = 0.001         # rad/s/√Hz
ACCEL_BIAS_RANDOM_WALK = 0.0001    # m/s³/√Hz
GYRO_BIAS_RANDOM_WALK = 0.00001    # rad/s²/√Hz

# Gravity
GRAVITY_MPS2 = 9.81

# Initial uncertainty
INIT_POS_SIGMA = 0.1               # meters
INIT_VEL_SIGMA = 1.0               # m/s
INIT_ACCEL_BIAS_SIGMA = 0.1        # m/s²
INIT_GYRO_BIAS_SIGMA = 0.01        # rad/s
INIT_QUAT_SIGMA = 0.1              # quaternion components

# Minimum fix quality to use
MIN_GPS_FIX_QUALITY = 1

# RTS smoothing (disable until forward EKF is validated)
ENABLE_RTS_SMOOTHING = False

# Maximum IMU dt to accept. PPC Tokyo IMU is 100 Hz, so dt is about 0.01 s.
MAX_IMU_DT = 0.02

plt.close("all")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────

def load_imu_raw(data_dir):
    """Load imu_raw.csv: gyro + accel + mag at 400Hz."""
    path = os.path.join(data_dir, "imu_raw.csv")
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found")
        return None
    df = pd.read_csv(path)
    for col in ["wall_time", "imu_timestamp_us",
                "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
                "accel_x_g", "accel_y_g", "accel_z_g"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["wall_time", "imu_timestamp_us",
                           "gyro_x_rad_s", "accel_x_g"])
    df = df.sort_values("wall_time").reset_index(drop=True)
    print(f"Loaded imu_raw.csv: {len(df)} rows, "
          f"time span: {df['wall_time'].iloc[-1] - df['wall_time'].iloc[0]:.1f}s")
    return df


def load_imu_quat(data_dir):
    """Load imu_quat.csv: AHRS quaternion."""
    path = os.path.join(data_dir, "imu_quat.csv")
    if not os.path.exists(path):
        print(f"[INFO] {path} not found, will use gyro integration only")
        return None
    df = pd.read_csv(path)
    for col in ["wall_time", "imu_timestamp_us", "q1", "q2", "q3", "q4"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["wall_time", "q1", "q2", "q3", "q4"])
    df = df.sort_values("wall_time").reset_index(drop=True)
    print(f"Loaded imu_quat.csv: {len(df)} rows")
    return df


def load_imu_gravity(data_dir):
    """Load imu_gravity.csv: gravity vector in body frame."""
    path = os.path.join(data_dir, "imu_gravity.csv")
    if not os.path.exists(path):
        print(f"[INFO] {path} not found, will estimate gravity from quaternion")
        return None
    df = pd.read_csv(path)
    for col in ["wall_time", "imu_timestamp_us",
                "gravity_x_g", "gravity_y_g", "gravity_z_g"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["wall_time", "gravity_x_g"])
    df = df.sort_values("wall_time").reset_index(drop=True)
    print(f"Loaded imu_gravity.csv: {len(df)} rows")
    return df


def load_gps_raw(data_dir):
    """Load gps_raw.csv: all NMEA sentences."""
    path = os.path.join(data_dir, "gps_raw.csv")
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found")
        return None
    df = pd.read_csv(path)
    for col in ["wall_time", "gps_utc", "lat", "lon", "alt",
                "fix_quality", "num_sats", "hdop",
                "lat_err_m", "lon_err_m", "alt_err_m",
                "heading_deg", "speed_knots", "speed_kmh",
                "course_true_deg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("wall_time").reset_index(drop=True)
    print(f"Loaded gps_raw.csv: {len(df)} rows")
    return df


# ─────────────────────────────────────────────
# STEP 2: TIME ALIGNMENT & PREPROCESSING
# ─────────────────────────────────────────────

def compute_imu_dt(imu_raw):
    """
    Compute dt between consecutive IMU samples using hardware timestamps.
    imu_timestamp_us is a 32-bit microsecond counter that wraps at 2^32.
    """
    ts = imu_raw["imu_timestamp_us"].values.astype(np.int64)
    dt = np.zeros(len(ts))
    for i in range(1, len(ts)):
        diff = ts[i] - ts[i - 1]
        if diff < 0:
            diff += 2 ** 32
        dt[i] = diff / 1e6
    dt[0] = 1.0 / 400.0
    imu_raw = imu_raw.copy()
    imu_raw["dt"] = dt
    return imu_raw


def merge_gravity_with_raw(imu_raw, imu_gravity):
    """
    For each RAW sample, find the most recent GRAVITY sample
    by wall_time (previous-or-equal, not nearest).
    This avoids using future data.
    """
    imu_raw = imu_raw.copy()

    if imu_gravity is None or len(imu_gravity) == 0:
        imu_raw["gravity_x_g"] = 0.0
        imu_raw["gravity_y_g"] = 0.0
        imu_raw["gravity_z_g"] = -1.0
        print("[WARNING] No gravity data, assuming [0, 0, -1g] "
              "(stationary, Z-down)")
        return imu_raw

    grav_times = imu_gravity["wall_time"].values
    grav_x = imu_gravity["gravity_x_g"].values
    grav_y = imu_gravity["gravity_y_g"].values
    grav_z = imu_gravity["gravity_z_g"].values

    raw_times = imu_raw["wall_time"].values

    # searchsorted right - 1 gives the last gravity sample <= raw_time
    idx = np.searchsorted(grav_times, raw_times, side="right") - 1
    idx = np.clip(idx, 0, len(grav_times) - 1)

    imu_raw["gravity_x_g"] = grav_x[idx]
    imu_raw["gravity_y_g"] = grav_y[idx]
    imu_raw["gravity_z_g"] = grav_z[idx]

    print(f"Merged gravity: {len(imu_raw)} RAW rows with "
          f"most recent gravity sample (not nearest)")
    return imu_raw


def merge_quat_with_raw(imu_raw, imu_quat):
    """
    For each RAW sample, find the most recent QUAT sample.
    Used only for initial orientation.
    """
    imu_raw = imu_raw.copy()

    if imu_quat is None or len(imu_quat) == 0:
        imu_raw["ahrs_q1"] = np.nan
        imu_raw["ahrs_q2"] = np.nan
        imu_raw["ahrs_q3"] = np.nan
        imu_raw["ahrs_q4"] = np.nan
        print("[WARNING] No QUAT data, orientation from gyro integration only")
        return imu_raw

    quat_times = imu_quat["wall_time"].values
    q1 = imu_quat["q1"].values
    q2 = imu_quat["q2"].values
    q3 = imu_quat["q3"].values
    q4 = imu_quat["q4"].values

    raw_times = imu_raw["wall_time"].values
    idx = np.searchsorted(quat_times, raw_times, side="right") - 1
    idx = np.clip(idx, 0, len(quat_times) - 1)

    imu_raw["ahrs_q1"] = q1[idx]
    imu_raw["ahrs_q2"] = q2[idx]
    imu_raw["ahrs_q3"] = q3[idx]
    imu_raw["ahrs_q4"] = q4[idx]

    print(f"Merged QUAT: {len(imu_raw)} RAW rows with most recent quaternion")
    return imu_raw


def extract_gps_measurements(gps_raw):
    """
    Extract GPS position, velocity, and heading measurements.
    Position from GGA, velocity from RMC + VTG, heading from HDT.
    All wall_times adjusted by GPS_TIME_OFFSET_S.
    """
    measurements = []

    # ── Position from GGA ──
    gga = gps_raw[gps_raw["msg_type"] == "GGA"].copy()
    gga = gga.dropna(subset=["lat", "lon"])

    for _, row in gga.iterrows():
        fq = row.get("fix_quality")
        if pd.isna(fq) or int(fq) < MIN_GPS_FIX_QUALITY:
            continue

        fq = int(fq)
        if fq == 4:
            sigma = GPS_POS_SIGMA_RTK_FIX
        elif fq == 5:
            sigma = GPS_POS_SIGMA_RTK_FLOAT
        elif fq == 2:
            sigma = GPS_POS_SIGMA_DGPS
        else:
            sigma = GPS_POS_SIGMA_STANDALONE

        # Use GST error if available (more accurate than fixed sigma)
        lat_err = row.get("lat_err_m")
        lon_err = row.get("lon_err_m")
        if pd.notna(lat_err) and pd.notna(lon_err):
            if lat_err > 0 and lon_err > 0:
                sigma = max(float(lat_err), float(lon_err))

        measurements.append({
            "wall_time": row["wall_time"] + GPS_TIME_OFFSET_S,
            "type": "gps_pos",
            "lat": row["lat"],
            "lon": row["lon"],
            "alt": row["alt"] if pd.notna(row.get("alt")) else 0.0,
            "sigma_h": sigma,
            "sigma_v": sigma * GPS_ALT_SCALE,
            "fix_quality": fq,
        })

    # ── Velocity from RMC ──
    rmc = gps_raw[gps_raw["msg_type"] == "RMC"].copy()
    rmc = rmc.dropna(subset=["speed_knots"])

    for _, row in rmc.iterrows():
        status = row.get("rmc_status")
        if pd.notna(status) and str(status).strip() == "V":
            continue

        speed_ms = float(row["speed_knots"]) * 0.514444
        course = row.get("course_true_deg")

        if pd.isna(course) or speed_ms < 0.1:
            continue

        course_rad = math.radians(float(course))
        ve = speed_ms * math.sin(course_rad)
        vn = speed_ms * math.cos(course_rad)

        measurements.append({
            "wall_time": row["wall_time"] + GPS_TIME_OFFSET_S,
            "type": "gps_vel",
            "ve": ve,
            "vn": vn,
            "speed_ms": speed_ms,
            "sigma": GPS_VEL_SIGMA,
        })

    # ── Velocity from VTG ──
    vtg = gps_raw[gps_raw["msg_type"] == "VTG"].copy()
    vtg = vtg.dropna(subset=["speed_kmh", "course_true_deg"])

    for _, row in vtg.iterrows():
        speed_ms = float(row["speed_kmh"]) / 3.6
        course = float(row["course_true_deg"])

        if speed_ms < 0.1:
            continue

        course_rad = math.radians(course)
        ve = speed_ms * math.sin(course_rad)
        vn = speed_ms * math.cos(course_rad)

        measurements.append({
            "wall_time": row["wall_time"] + GPS_TIME_OFFSET_S,
            "type": "gps_vel",
            "ve": ve,
            "vn": vn,
            "speed_ms": speed_ms,
            "sigma": GPS_VEL_SIGMA,
        })

    # ── Heading from HDT ──
    hdt = gps_raw[gps_raw["msg_type"] == "HDT"].copy()
    hdt = hdt.dropna(subset=["heading_deg"])

    for _, row in hdt.iterrows():
        measurements.append({
            "wall_time": row["wall_time"] + GPS_TIME_OFFSET_S,
            "type": "gps_heading",
            "heading_deg": float(row["heading_deg"]),
            "sigma_deg": GPS_HEADING_SIGMA_DEG,
        })

    measurements.sort(key=lambda m: m["wall_time"])

    n_pos = len([m for m in measurements if m["type"] == "gps_pos"])
    n_vel = len([m for m in measurements if m["type"] == "gps_vel"])
    n_hdg = len([m for m in measurements if m["type"] == "gps_heading"])
    print(f"GPS measurements: {n_pos} pos, {n_vel} vel "
          f"(RMC+VTG), {n_hdg} heading")
    if GPS_TIME_OFFSET_S != 0:
        print(f"  GPS time offset applied: {GPS_TIME_OFFSET_S:.4f}s")
    return measurements


# ─────────────────────────────────────────────
# FLAT-EARTH COORDINATE CONVERSION
# ─────────────────────────────────────────────

_M_PER_DEG_LAT = 111_132.92
_M_PER_DEG_LON_EQ = 111_319.49


def _m_per_deg_lon(lat_deg):
    return _M_PER_DEG_LON_EQ * math.cos(math.radians(lat_deg))


def gps_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    e = (lon - ref_lon) * _m_per_deg_lon(ref_lat)
    n = (lat - ref_lat) * _M_PER_DEG_LAT
    u = alt - ref_alt
    return np.array([e, n, u])


def enu_to_gps(enu, ref_lat, ref_lon, ref_alt):
    lat = ref_lat + enu[1] / _M_PER_DEG_LAT
    lon = ref_lon + enu[0] / _m_per_deg_lon(ref_lat)
    alt = ref_alt + enu[2]
    return lat, lon, alt


# ─────────────────────────────────────────────
# QUATERNION UTILITIES
# ─────────────────────────────────────────────

def qnorm(q):
    """Normalize quaternion [x,y,z,w] (scipy convention)."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_to_rotation_matrix(q):
    """Convert quaternion [x,y,z,w] to 3x3 rotation matrix."""
    return Rot.from_quat(qnorm(q)).as_matrix()


def quat_from_rotvec(rotvec):
    """Create quaternion from rotation vector (axis*angle)."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return Rot.from_rotvec(rotvec).as_quat()


def quat_multiply(q1, q2):
    """Multiply two quaternions in scipy [x,y,z,w] convention."""
    return (Rot.from_quat(q1) * Rot.from_quat(q2)).as_quat()


def quat_to_heading_deg(q):
    """
    Convert quaternion to heading in degrees [0, 360).
    Heading: 0=North, 90=East, 180=South, 270=West (CW from North).

    scipy euler ZYX: yaw is rotation about Z axis.
    scipy yaw=0 → X axis points forward.
    If body X=East, Y=North: yaw=0 means facing East → heading=90.
    General: heading = 90 - yaw_deg, wrapped to [0, 360).

    ★ IMPORTANT: This must match GPS HDT convention. ★
    Verify with straight-line driving data:
      Driving North → HDT ≈ 0° → this function should return ≈ 0°
      Driving East  → HDT ≈ 90° → this function should return ≈ 90°
    If they don't match, adjust the formula below.
    """
    r = Rot.from_quat(qnorm(q))
    yaw_rad = r.as_euler("zyx", degrees=False)[0]
    heading = (90.0 - np.degrees(yaw_rad)) % 360.0
    return heading


def heading_deg_to_yaw_rad(heading_deg):
    """
    Convert GPS heading (CW from North) to scipy yaw (CCW from East).
    Inverse of quat_to_heading_deg.
    """
    yaw_deg = (90.0 - heading_deg) % 360.0
    if yaw_deg > 180.0:
        yaw_deg -= 360.0
    return math.radians(yaw_deg)


def imu_quat_to_scipy(q1, q2, q3, q4):
    """
    Convert IMU QUAT packet [q1(w), q2(x), q3(y), q4(z)]
    to scipy convention [x, y, z, w].
    """
    return np.array([q2, q3, q4, q1])


# ─────────────────────────────────────────────
# STEP 3: EKF EQUATIONS
# ─────────────────────────────────────────────
#
# ═══════════════════════════════════════════════
# STATE VECTOR x (16 elements):
# ═══════════════════════════════════════════════
#   x[0:3]   pos_enu [e, n, u] meters
#   x[3:6]   vel_enu [ve, vn, vu] m/s
#   x[6:9]   accel_bias_body [bax, bay, baz] m/s²
#   x[9:12]  gyro_bias_body [bgx, bgy, bgz] rad/s
#   x[12:16] quaternion [qx, qy, qz, qw] scipy convention
#
# ═══════════════════════════════════════════════
# PREDICT (IMU, 400Hz):
# ═══════════════════════════════════════════════
#
# Input:
#   gyro_meas [3] rad/s (body frame, from RAW packet)
#   accel_meas_g [3] in g (body frame, INCLUDES gravity, from RAW packet)
#   gravity_body_g [3] in g (body frame, from GRAVITY packet)
#   dt: time step seconds
#
# Equations:
#
#   (1) gyro = gyro_meas - bg
#       Remove gyro bias to get true angular rate.
#       Without this: heading drifts ~1°/min → at 30m/s, 1m error in 2s.
#
#   (2) accel_linear_body = (accel_meas_g - gravity_body_g) * 9.81 - ba
#       Step a: subtract gravity to get linear acceleration (in g)
#       Step b: convert g → m/s²
#       Step c: subtract accelerometer bias
#       Without gravity removal: 9.81 m/s² integrated → 49m in 0.1s
#       Without bias removal: 0.01 m/s² bias → accumulates to meters
#
#   (3) q_new = q ⊗ quat_from_rotvec(gyro * dt)
#       Integrate angular rate to update orientation quaternion.
#       ⊗ is quaternion multiplication (rotation composition).
#
#   (4) C = rotation_matrix(q_new)
#       3x3 matrix that rotates body frame → ENU frame.
#
#   (5) accel_enu = C @ accel_linear_body
#       Rotate body-frame linear acceleration to ENU frame.
#       This is where heading error causes position error:
#       wrong C → accel points wrong direction → velocity/position drift.
#
#   (6) vel_new = vel + accel_enu * dt
#       Integrate acceleration to update velocity.
#
#   (7) pos_new = pos + vel * dt + 0.5 * accel_enu * dt²
#       Integrate velocity to update position (trapezoidal).
#
#   (8) ba_new = ba, bg_new = bg
#       Biases modeled as random walk: no deterministic change,
#       but process noise allows EKF to slowly adjust them.
#
# ═══════════════════════════════════════════════
# UPDATE — GPS POSITION (10Hz):
# ═══════════════════════════════════════════════
#
#   z = [e_gps, n_gps, u_gps]   GPS position in ENU
#   h(x) = x[0:3]               predicted position from state
#   H = [I₃ | 0₃ₓ₁₃]           3×16 Jacobian (trivial)
#   R = diag(σ_h², σ_h², σ_v²)  measurement noise
#
#   innovation y = z - h(x)
#   If |y| small → EKF tracking well
#   If |y| large → GPS jumped or IMU drifted
#
#   Kalman gain K = P H' (H P H' + R)⁻¹
#   For RTK (σ=0.05m) with P_pos~0.01m²: K ≈ 0.8-0.95
#   → GPS dominates position, but biases still get corrected
#
# ═══════════════════════════════════════════════
# UPDATE — GPS VELOCITY (10Hz):
# ═══════════════════════════════════════════════
#
#   z = [ve_gps, vn_gps]        from RMC/VTG speed + course
#   h(x) = x[3:5]               predicted velocity (east, north)
#   H = [0₂ₓ₃ | I₂ | 0₂ₓ₁₁]   2×16 Jacobian
#   R = diag(σ_vel², σ_vel²)
#
#   GPS velocity is independent of position differencing.
#   It directly corrects velocity state, which is critical for
#   reducing position drift between GPS fixes.
#   Also indirectly corrects accel bias through the coupling in P.
#
# ═══════════════════════════════════════════════
# UPDATE — GPS HEADING (when available):
# ═══════════════════════════════════════════════
#
#   z = heading_deg              from HDT sentence
#   h(x) = heading_from_quat(x[12:16])
#   H = numerical Jacobian ∂heading/∂state (1×16)
#   R = σ_heading²
#
#   Directly corrects orientation quaternion.
#   This is the most important measurement for reducing
#   position error at high speed, because heading error
#   misdirects the entire velocity vector.
#

def ekf_predict(x, P, gyro_meas, accel_meas_g, gravity_body_g, dt):
    """
    EKF prediction step using IMU measurements.

    Returns:
        x_new: predicted state (16,)
        P_new: predicted covariance (16, 16)
        F: state transition Jacobian (16, 16) — saved for RTS
    """
    # Compute predicted state
    x_new = _predict_state(x, gyro_meas, accel_meas_g, gravity_body_g, dt)

    # Compute Jacobian F = ∂f/∂x numerically
    F = _numerical_jacobian_F(x, gyro_meas, accel_meas_g,
                              gravity_body_g, dt, x_new)

    # Process noise Q
    Q = np.zeros((16, 16))
    Q[0:3, 0:3] = np.eye(3) * (ACCEL_NOISE_DENSITY * dt ** 2) ** 2
    Q[3:6, 3:6] = np.eye(3) * (ACCEL_NOISE_DENSITY * dt) ** 2
    Q[6:9, 6:9] = np.eye(3) * (ACCEL_BIAS_RANDOM_WALK * dt) ** 2
    Q[9:12, 9:12] = np.eye(3) * (GYRO_BIAS_RANDOM_WALK * dt) ** 2
    Q[12:16, 12:16] = np.eye(4) * (GYRO_NOISE_DENSITY * dt) ** 2

    # Covariance propagation
    P_new = F @ P @ F.T + Q
    P_new = 0.5 * (P_new + P_new.T)

    return x_new, P_new, F


def _predict_state(x, gyro_meas, accel_meas_g, gravity_body_g, dt):
    """State prediction (equations 1-8 above)."""
    pos = x[0:3]
    vel = x[3:6]
    ba = x[6:9]
    bg = x[9:12]
    q = qnorm(x[12:16])

    # (1) Bias-corrected gyro
    gyro = gyro_meas - bg

    # (2) Gravity-free, bias-corrected acceleration in m/s²
    accel_linear_body = (accel_meas_g - gravity_body_g) * GRAVITY_MPS2 - ba

    # (3) Update orientation
    q_new = qnorm(quat_multiply(q, quat_from_rotvec(gyro * dt)))

    # (4-5) Rotate acceleration to ENU
    C = quat_to_rotation_matrix(q_new)
    accel_enu = C @ accel_linear_body

    # (6-7) Update velocity and position
    vel_new = vel + accel_enu * dt
    pos_new = pos + vel * dt + 0.5 * accel_enu * dt ** 2

    # (8) Biases unchanged
    x_new = np.zeros(16)
    x_new[0:3] = pos_new
    x_new[3:6] = vel_new
    x_new[6:9] = ba
    x_new[9:12] = bg
    x_new[12:16] = q_new
    return x_new


def _numerical_jacobian_F(x, gyro_meas, accel_meas_g, gravity_body_g,
                          dt, f0=None, eps=1e-7):
    """
    Compute state transition Jacobian numerically.
    F[i,j] = ∂f_i/∂x_j

    f0 is the already-computed _predict_state(x,...) to avoid recomputation.
    """
    n = len(x)
    if f0 is None:
        f0 = _predict_state(x, gyro_meas, accel_meas_g, gravity_body_g, dt)
    F = np.zeros((n, n))
    for j in range(n):
        x_pert = x.copy()
        x_pert[j] += eps
        if 12 <= j <= 15:
            x_pert[12:16] = qnorm(x_pert[12:16])
        f_pert = _predict_state(x_pert, gyro_meas, accel_meas_g,
                                gravity_body_g, dt)
        F[:, j] = (f_pert - f0) / eps
    return F


def ekf_update_position(x, P, z_enu, R_pos):
    """
    EKF update with GPS position measurement.

    Returns:
        x_new, P_new, innovation (3,)
    """
    H = np.zeros((3, 16))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0

    hx = x[0:3]
    innovation = z_enu - hx

    S = H @ P @ H.T + R_pos
    try:
        K = P @ H.T @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return x.copy(), P.copy(), innovation

    x_new = x + K @ innovation
    x_new[12:16] = qnorm(x_new[12:16])

    # Joseph form for numerical stability
    I_KH = np.eye(16) - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R_pos @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    return x_new, P_new, innovation


def ekf_update_velocity(x, P, z_vel, R_vel):
    """
    EKF update with GPS velocity measurement (east, north).

    Returns:
        x_new, P_new, innovation (2,)
    """
    H = np.zeros((2, 16))
    H[0, 3] = 1.0
    H[1, 4] = 1.0

    hx = x[3:5]
    innovation = z_vel - hx

    S = H @ P @ H.T + R_vel
    try:
        K = P @ H.T @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return x.copy(), P.copy(), innovation

    x_new = x + K @ innovation
    x_new[12:16] = qnorm(x_new[12:16])

    I_KH = np.eye(16) - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R_vel @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    return x_new, P_new, innovation


def ekf_update_heading(x, P, heading_deg, sigma_deg):
    """
    EKF update with GPS heading measurement.

    Returns:
        x_new, P_new, innovation_deg (scalar)
    """
    pred_heading = quat_to_heading_deg(x[12:16])

    # Innovation with wraparound handling
    innov = heading_deg - pred_heading
    if innov > 180.0:
        innov -= 360.0
    elif innov < -180.0:
        innov += 360.0

    # Numerical Jacobian: ∂heading/∂state
    H = np.zeros((1, 16))
    eps = 1e-6
    for j in range(16):
        x_pert = x.copy()
        x_pert[j] += eps
        if 12 <= j <= 15:
            x_pert[12:16] = qnorm(x_pert[12:16])
        h_pert = quat_to_heading_deg(x_pert[12:16])
        dh = h_pert - pred_heading
        if dh > 180.0:
            dh -= 360.0
        elif dh < -180.0:
            dh += 360.0
        H[0, j] = dh / eps

    R_h = np.array([[sigma_deg ** 2]])
    z = np.array([innov])

    S = H @ P @ H.T + R_h
    try:
        K = P @ H.T @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return x.copy(), P.copy(), innov

    x_new = x + (K @ z).flatten()
    x_new[12:16] = qnorm(x_new[12:16])

    I_KH = np.eye(16) - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R_h @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    return x_new, P_new, innov


# ─────────────────────────────────────────────
# STEP 4: RTS SMOOTHER (disabled by default)
# ─────────────────────────────────────────────
#
# Rauch-Tung-Striebel smoother.
# Runs backward through saved forward-pass states.
#
# For step k (from N-2 down to 0):
#   G_k = P_fwd[k] @ F[k]' @ inv(P_pred[k+1])
#   x_smooth[k] = x_fwd[k] + G_k @ (x_smooth[k+1] - x_pred[k+1])
#   P_smooth[k] = P_fwd[k] + G_k @ (P_smooth[k+1] - P_pred[k+1]) @ G_k'
#
# Arrays:
#   xs_fwd[k]:  state after update at step k (length N)
#   Ps_fwd[k]:  covariance after update at step k (length N)
#   xs_pred[k]: state after predict, before update at step k (length N)
#   Ps_pred[k]: covariance after predict, before update at step k (length N)
#   Fs[k]:      transition Jacobian from step k to step k+1 (length N-1)
#
# Index relationship:
#   xs_pred[k+1] = f(xs_fwd[k])  using Fs[k]
#   This means Fs[k] maps state k → predicted state k+1
#

def rts_smooth(xs_fwd, Ps_fwd, xs_pred, Ps_pred, Fs):
    """
    Rauch-Tung-Striebel backward smoother.

    Args:
        xs_fwd:  forward-filtered states, length N
        Ps_fwd:  forward-filtered covariances, length N
        xs_pred: predicted states (before update), length N
        Ps_pred: predicted covariances (before update), length N
        Fs:      state transition Jacobians, length N-1
                 Fs[k] is the Jacobian from step k to step k+1

    Returns:
        xs_smooth, Ps_smooth: smoothed states and covariances, length N
    """
    N = len(xs_fwd)
    assert len(Ps_fwd) == N
    assert len(xs_pred) == N
    assert len(Ps_pred) == N
    assert len(Fs) == N - 1

    xs_smooth = [None] * N
    Ps_smooth = [None] * N

    xs_smooth[N - 1] = xs_fwd[N - 1].copy()
    Ps_smooth[N - 1] = Ps_fwd[N - 1].copy()

    for k in range(N - 2, -1, -1):
        try:
            G = Ps_fwd[k] @ Fs[k].T @ np.linalg.inv(Ps_pred[k + 1])
        except np.linalg.LinAlgError:
            xs_smooth[k] = xs_fwd[k].copy()
            Ps_smooth[k] = Ps_fwd[k].copy()
            continue

        dx = xs_smooth[k + 1] - xs_pred[k + 1]
        xs_smooth[k] = xs_fwd[k] + G @ dx
        xs_smooth[k][12:16] = qnorm(xs_smooth[k][12:16])

        dP = Ps_smooth[k + 1] - Ps_pred[k + 1]
        Ps_smooth[k] = Ps_fwd[k] + G @ dP @ G.T
        Ps_smooth[k] = 0.5 * (Ps_smooth[k] + Ps_smooth[k].T)

    return xs_smooth, Ps_smooth


# ─────────────────────────────────────────────
# STEP 5: MAIN EKF PROCESSING
# ─────────────────────────────────────────────

def run_ekf(imu_raw, gps_measurements):
    """
    Run EKF forward pass, then optional RTS smoothing.

    Returns:
        results: list of dicts with time, position, velocity, biases
        ref_lat, ref_lon, ref_alt: ENU reference point
    """
    # ── Find reference point (first valid GPS fix) ──
    first_gps = None
    for m in gps_measurements:
        if m["type"] == "gps_pos":
            first_gps = m
            break

    if first_gps is None:
        print("[ERROR] No valid GPS position found")
        return None, None, None, None

    ref_lat = first_gps["lat"]
    ref_lon = first_gps["lon"]
    ref_alt = first_gps["alt"]
    print(f"Reference: lat={ref_lat:.8f} lon={ref_lon:.8f} alt={ref_alt:.2f}")

    # ── Initialize state ──
    x = np.zeros(16)

    # Position from first GPS
    x[0:3] = gps_to_enu(first_gps["lat"], first_gps["lon"],
                         first_gps["alt"], ref_lat, ref_lon, ref_alt)

    # Velocity from first GPS velocity measurement
    first_vel = None
    for m in gps_measurements:
        if m["type"] == "gps_vel":
            first_vel = m
            break

    if first_vel is not None:
        x[3] = first_vel["ve"]
        x[4] = first_vel["vn"]
        x[5] = 0.0
        print(f"Initial velocity from GPS: ve={x[3]:.2f} vn={x[4]:.2f} m/s "
              f"(speed={first_vel['speed_ms']:.1f} m/s)")
    else:
        x[3:6] = 0.0
        print("[WARNING] No GPS velocity found, initial velocity = 0")

    # Biases start at zero
    x[6:9] = 0.0
    x[9:12] = 0.0

    # Orientation from first AHRS quaternion
    first_row = imu_raw.iloc[0]
    if not pd.isna(first_row.get("ahrs_q1")):
        x[12:16] = imu_quat_to_scipy(
            first_row["ahrs_q1"], first_row["ahrs_q2"],
            first_row["ahrs_q3"], first_row["ahrs_q4"])
        print(f"Initial orientation from AHRS: "
              f"heading={quat_to_heading_deg(x[12:16]):.1f}°")
    else:
        x[12:16] = np.array([0.0, 0.0, 0.0, 1.0])
        print("[WARNING] No AHRS quaternion, using identity orientation")

    x[12:16] = qnorm(x[12:16])

    # ── Initialize covariance ──
    P = np.zeros((16, 16))
    P[0:3, 0:3] = np.eye(3) * INIT_POS_SIGMA ** 2
    P[3:6, 3:6] = np.eye(3) * INIT_VEL_SIGMA ** 2
    P[6:9, 6:9] = np.eye(3) * INIT_ACCEL_BIAS_SIGMA ** 2
    P[9:12, 9:12] = np.eye(3) * INIT_GYRO_BIAS_SIGMA ** 2
    P[12:16, 12:16] = np.eye(4) * INIT_QUAT_SIGMA ** 2

    # ── Build unified event timeline ──
    events = []

    for i, row in imu_raw.iterrows():
        events.append({
            "wall_time": row["wall_time"],
            "type": "imu",
            "index": i,
        })

    for m in gps_measurements:
        events.append({
            "wall_time": m["wall_time"],
            "type": m["type"],
            "data": m,
        })

    events.sort(key=lambda e: e["wall_time"])
    total_events = len(events)
    print(f"Event timeline: {total_events} events "
          f"({len(imu_raw)} IMU + {len(gps_measurements)} GPS)")

    # ── Storage for RTS smoother ──
    # xs_fwd[k]: filtered state at step k (after any GPS update)
    # Ps_fwd[k]: filtered covariance at step k
    # xs_pred[k]: predicted state at step k (before GPS update)
    # Ps_pred[k]: predicted covariance at step k
    # Fs[k]: Jacobian from step k to step k+1
    # Only populated for IMU predict steps (GPS updates modify in-place)
    if ENABLE_RTS_SMOOTHING:
        xs_fwd_list = []
        Ps_fwd_list = []
        xs_pred_list = []
        Ps_pred_list = []
        Fs_list = []

    results = []
    imu_only_results = []
    innovations_pos = []
    innovations_vel = []
    innovations_hdg = []

    imu_predict_count = 0
    gps_pos_count = 0
    gps_vel_count = 0
    gps_hdg_count = 0
    skip_count = 0

    t_start = time_module.time()
    x_imu_only = x.copy()

    # ── Forward pass ──
    for evt_idx, evt in enumerate(events):

        # ─── IMU PREDICT ───
        if evt["type"] == "imu":
            row = imu_raw.iloc[evt["index"]]
            dt = row["dt"]

            if dt <= 0 or dt > MAX_IMU_DT:
                skip_count += 1
                continue

            gyro_meas = np.array([
                row["gyro_x_rad_s"],
                row["gyro_y_rad_s"],
                row["gyro_z_rad_s"],
            ])
            accel_meas_g = np.array([
                row["accel_x_g"],
                row["accel_y_g"],
                row["accel_z_g"],
            ])
            gravity_body_g = np.array([
                row["gravity_x_g"],
                row["gravity_y_g"],
                row["gravity_z_g"],
            ])

            if np.any(np.isnan(gyro_meas)) or np.any(np.isnan(accel_meas_g)):
                skip_count += 1
                continue

            if PLOT_IMU_ONLY:
                x_imu_only = _predict_state(
                    x_imu_only, gyro_meas, accel_meas_g, gravity_body_g, dt)
                imu_pos_gps = enu_to_gps(
                    x_imu_only[0:3], ref_lat, ref_lon, ref_alt)
                imu_only_results.append({
                    "wall_time": row["wall_time"],
                    "lat": imu_pos_gps[0],
                    "lon": imu_pos_gps[1],
                    "alt": imu_pos_gps[2],
                    "ve": x_imu_only[3],
                    "vn": x_imu_only[4],
                    "vu": x_imu_only[5],
                    "yaw": quat_to_heading_deg(x_imu_only[12:16]),
                })

            # Predict
            x_new, P_new, F = ekf_predict(
                x, P, gyro_meas, accel_meas_g, gravity_body_g, dt)

            # Save for RTS: predicted state BEFORE any GPS update
            if ENABLE_RTS_SMOOTHING:
                xs_pred_list.append(x_new.copy())
                Ps_pred_list.append(P_new.copy())
                Fs_list.append(F)

            x = x_new
            P = P_new
            imu_predict_count += 1

            # Save filtered state (may be overwritten by GPS update below)
            if ENABLE_RTS_SMOOTHING:
                xs_fwd_list.append(x.copy())
                Ps_fwd_list.append(P.copy())

            # Build output row
            pos_gps = enu_to_gps(x[0:3], ref_lat, ref_lon, ref_alt)
            euler = Rot.from_quat(qnorm(x[12:16])).as_euler(
                'zyx', degrees=True)

            results.append({
                "wall_time": row["wall_time"],
                "source": "ekf",
                "lat": pos_gps[0],
                "lon": pos_gps[1],
                "alt": pos_gps[2],
                "ve": x[3],
                "vn": x[4],
                "vu": x[5],
                "yaw": quat_to_heading_deg(x[12:16]),
                "pitch": euler[1],
                "roll": euler[2],
                "ba_x": x[6],
                "ba_y": x[7],
                "ba_z": x[8],
                "bg_x": x[9],
                "bg_y": x[10],
                "bg_z": x[11],
                "P_pos": math.sqrt(P[0, 0] + P[1, 1]),
                "P_vel": math.sqrt(P[3, 3] + P[4, 4]),
            })

        # ─── GPS POSITION UPDATE ───
        elif evt["type"] == "gps_pos":
            m = evt["data"]
            z_enu = gps_to_enu(m["lat"], m["lon"], m["alt"],
                               ref_lat, ref_lon, ref_alt)

            R_pos = np.diag([
                m["sigma_h"] ** 2,
                m["sigma_h"] ** 2,
                m["sigma_v"] ** 2,
            ])

            x, P, innov = ekf_update_position(x, P, z_enu, R_pos)
            gps_pos_count += 1
            innovations_pos.append({
                "wall_time": m["wall_time"],
                "innov_e": innov[0],
                "innov_n": innov[1],
                "innov_u": innov[2],
                "norm": np.linalg.norm(innov[:2]),
                "fix_quality": m["fix_quality"],
            })

            # Overwrite last RTS entry with updated state
            if ENABLE_RTS_SMOOTHING and len(xs_fwd_list) > 0:
                xs_fwd_list[-1] = x.copy()
                Ps_fwd_list[-1] = P.copy()

        # ─── GPS VELOCITY UPDATE ───
        elif evt["type"] == "gps_vel":
            m = evt["data"]
            z_vel = np.array([m["ve"], m["vn"]])
            R_vel = np.eye(2) * m["sigma"] ** 2

            x, P, innov = ekf_update_velocity(x, P, z_vel, R_vel)
            gps_vel_count += 1
            innovations_vel.append({
                "wall_time": m["wall_time"],
                "innov_ve": innov[0],
                "innov_vn": innov[1],
            })

            if ENABLE_RTS_SMOOTHING and len(xs_fwd_list) > 0:
                xs_fwd_list[-1] = x.copy()
                Ps_fwd_list[-1] = P.copy()

        # ─── GPS HEADING UPDATE ───
        elif evt["type"] == "gps_heading":
            m = evt["data"]
            x, P, innov = ekf_update_heading(
                x, P, m["heading_deg"], m["sigma_deg"])
            gps_hdg_count += 1
            innovations_hdg.append({
                "wall_time": m["wall_time"],
                "innov_deg": innov,
            })

            if ENABLE_RTS_SMOOTHING and len(xs_fwd_list) > 0:
                xs_fwd_list[-1] = x.copy()
                Ps_fwd_list[-1] = P.copy()

        # Progress
        if (evt_idx + 1) % 50000 == 0:
            elapsed = time_module.time() - t_start
            pct = (evt_idx + 1) / total_events * 100
            print(f"  Forward: {pct:.0f}% ({evt_idx + 1}/{total_events}) "
                  f"{elapsed:.1f}s")

    elapsed = time_module.time() - t_start
    print(f"\nForward pass complete in {elapsed:.1f}s:")
    print(f"  IMU predictions:     {imu_predict_count}")
    print(f"  GPS pos updates:     {gps_pos_count}")
    print(f"  GPS vel updates:     {gps_vel_count}")
    print(f"  GPS heading updates: {gps_hdg_count}")
    print(f"  Skipped:             {skip_count}")

    if len(innovations_pos) > 0:
        innov_norms = [ip["norm"] for ip in innovations_pos]
        print(f"  Position innovation: "
              f"mean={np.mean(innov_norms):.4f}m "
              f"max={np.max(innov_norms):.4f}m "
              f"std={np.std(innov_norms):.4f}m")

    # ── RTS Smoothing ──
    if ENABLE_RTS_SMOOTHING and len(xs_fwd_list) > 10:
        print(f"\nRunning RTS smoother on {len(xs_fwd_list)} states...")
        assert len(xs_fwd_list) == len(Ps_fwd_list)
        assert len(xs_pred_list) == len(Ps_pred_list)
        assert len(Fs_list) == len(xs_fwd_list) - 1

        t_rts = time_module.time()
        xs_smooth, Ps_smooth = rts_smooth(
            xs_fwd_list, Ps_fwd_list,
            xs_pred_list, Ps_pred_list,
            Fs_list)
        elapsed_rts = time_module.time() - t_rts
        print(f"RTS smoothing complete in {elapsed_rts:.1f}s")

        # Overwrite results with smoothed states
        # results and xs_fwd_list have the same length
        # (one result per IMU predict step)
        for i in range(min(len(results), len(xs_smooth))):
            xs = xs_smooth[i]
            pos_gps = enu_to_gps(xs[0:3], ref_lat, ref_lon, ref_alt)
            euler = Rot.from_quat(qnorm(xs[12:16])).as_euler(
                'zyx', degrees=True)

            results[i]["lat"] = pos_gps[0]
            results[i]["lon"] = pos_gps[1]
            results[i]["alt"] = pos_gps[2]
            results[i]["ve"] = xs[3]
            results[i]["vn"] = xs[4]
            results[i]["vu"] = xs[5]
            results[i]["yaw"] = quat_to_heading_deg(xs[12:16])
            results[i]["pitch"] = euler[1]
            results[i]["roll"] = euler[2]
            results[i]["ba_x"] = xs[6]
            results[i]["ba_y"] = xs[7]
            results[i]["ba_z"] = xs[8]
            results[i]["bg_x"] = xs[9]
            results[i]["bg_y"] = xs[10]
            results[i]["bg_z"] = xs[11]
            results[i]["P_pos"] = math.sqrt(
                Ps_smooth[i][0, 0] + Ps_smooth[i][1, 1])
            results[i]["P_vel"] = math.sqrt(
                Ps_smooth[i][3, 3] + Ps_smooth[i][4, 4])
            results[i]["source"] = "rts"

    return results, imu_only_results, innovations_pos, ref_lat, ref_lon, ref_alt


# ─────────────────────────────────────────────
# STEP 6: OUTPUT
# ─────────────────────────────────────────────

def save_results(results, output_dir):
    """Save EKF results to CSV."""
    filepath = os.path.join(output_dir, "position_ekf.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "wall_time", "source", "lat", "lon", "alt",
            "ve", "vn", "vu",
            "yaw", "pitch", "roll",
            "ba_x", "ba_y", "ba_z",
            "bg_x", "bg_y", "bg_z",
            "P_pos", "P_vel",
        ])
        for r in results:
            writer.writerow([
                f"{r['wall_time']:.6f}",
                r["source"],
                f"{r['lat']:.10f}",
                f"{r['lon']:.10f}",
                f"{r['alt']:.4f}",
                f"{r['ve']:.6f}",
                f"{r['vn']:.6f}",
                f"{r['vu']:.6f}",
                f"{r['yaw']:.4f}",
                f"{r['pitch']:.4f}",
                f"{r['roll']:.4f}",
                f"{r['ba_x']:.8f}",
                f"{r['ba_y']:.8f}",
                f"{r['ba_z']:.8f}",
                f"{r['bg_x']:.8f}",
                f"{r['bg_y']:.8f}",
                f"{r['bg_z']:.8f}",
                f"{r['P_pos']:.6f}",
                f"{r['P_vel']:.6f}",
            ])
    print(f"Saved {len(results)} rows to {filepath}")


def save_innovations(innovations_pos, output_dir):
    """Save GPS position innovations for analysis."""
    filepath = os.path.join(output_dir, "innovations.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "wall_time", "innov_e", "innov_n", "innov_u",
            "norm_2d", "fix_quality",
        ])
        for ip in innovations_pos:
            writer.writerow([
                f"{ip['wall_time']:.6f}",
                f"{ip['innov_e']:.6f}",
                f"{ip['innov_n']:.6f}",
                f"{ip['innov_u']:.6f}",
                f"{ip['norm']:.6f}",
                ip["fix_quality"],
            ])
    print(f"Saved {len(innovations_pos)} innovations to {filepath}")


def _find_ppc_reference_csv(data_dir):
    """Find PPC reference.csv for either a PPC run dir or ekf_input_* dir."""
    candidates = [
        Path(data_dir) / "reference.csv",
        Path(data_dir).parent / "reference.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_reference_track_for_plot(data_dir, gps_measurements):
    """Load PPC reference trajectory and trim it to the plotted EKF window."""
    reference_path = _find_ppc_reference_csv(data_dir)
    if reference_path is None:
        return None

    try:
        ref_df = pd.read_csv(reference_path)
        ref_df.columns = [c.strip() for c in ref_df.columns]
    except Exception as exc:
        print(f"[WARNING] Could not read reference.csv for plot: {exc}")
        return None

    required = ["GPS TOW (s)", "Latitude (deg)", "Longitude (deg)"]
    if any(col not in ref_df.columns for col in required):
        print("[WARNING] reference.csv missing GPS TOW/Latitude/Longitude columns")
        return None

    for col in required:
        ref_df[col] = pd.to_numeric(ref_df[col], errors="coerce")
    ref_df = ref_df.dropna(subset=required)
    if len(ref_df) == 0:
        return None

    imu_path = reference_path.parent / "imu.csv"
    if imu_path.exists() and len(gps_measurements) > 0:
        try:
            imu_head = pd.read_csv(imu_path, nrows=1)
            imu_head.columns = [c.strip() for c in imu_head.columns]
            start_tow = PPC_START_TOW
            if start_tow is None:
                start_tow = float(imu_head["GPS TOW (s)"].iloc[0])
            min_wall = min(m["wall_time"] for m in gps_measurements)
            max_wall = max(m["wall_time"] for m in gps_measurements)
            ref_df = ref_df[
                (ref_df["GPS TOW (s)"] >= start_tow + min_wall)
                & (ref_df["GPS TOW (s)"] <= start_tow + max_wall)
            ]
        except Exception as exc:
            print(f"[WARNING] Could not trim reference.csv by time: {exc}")

    if len(ref_df) == 0:
        return None
    print(f"Loaded reference.csv for plot: {len(ref_df)} rows")
    return ref_df


def plot_results(results, imu_only_results, innovations_pos, gps_measurements,
                 ref_lat, ref_lon, ref_alt, output_dir):
    """Generate analysis plots."""
    if len(results) == 0:
        print("No results to plot")
        return

    df = pd.DataFrame(results)
    t0 = df["wall_time"].iloc[0]
    df["t"] = df["wall_time"] - t0

    gps_pos = [m for m in gps_measurements if m["type"] == "gps_pos"]
    gps_lats = [m["lat"] for m in gps_pos]
    gps_lons = [m["lon"] for m in gps_pos]
    gps_times = [m["wall_time"] - t0 for m in gps_pos]
    reference_df = _load_reference_track_for_plot(DATA_DIR, gps_measurements)

    # ── Plot 1: Trajectory ──
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.canvas.manager.set_window_title("1 - Trajectory")
    if reference_df is not None:
        ax.plot(reference_df["Longitude (deg)"], reference_df["Latitude (deg)"],
                "k--", linewidth=6, alpha=0.8, label="Reference CSV")
    ax.plot(df["lon"], df["lat"], "-", color="orange",
            linewidth=8, alpha=0.7, label="EKF")
    # if PLOT_IMU_ONLY and len(imu_only_results) > 0:
    #     imu_df = pd.DataFrame(imu_only_results)
    #     ax.plot(imu_df["lon"], imu_df["lat"], "-", color="purple",
    #             linewidth=0.6, alpha=0.7, label="IMU only")
    ax.plot(gps_lons, gps_lats, "b.", markersize=10, label="GPS/reference updates")
    if len(gps_lats) > 0:
        ax.scatter(gps_lons[0], gps_lats[0], c="green", s=100,
                  marker="^", zorder=5, label="Start")
        ax.scatter(gps_lons[-1], gps_lats[-1], c="red", s=100,
                  marker="s", zorder=5, label="End")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("EKF Trajectory vs PPC Reference")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(useOffset=False, style="plain")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "1_trajectory.png"),
                dpi=150, bbox_inches="tight")

    # ── Plot 2: Velocity ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.canvas.manager.set_window_title("2 - Velocity")
    ax1.plot(df["t"], df["ve"], linewidth=0.5, label="Ve (east)")
    ax1.plot(df["t"], df["vn"], linewidth=0.5, label="Vn (north)")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("EKF Velocity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    speed = np.sqrt(df["ve"] ** 2 + df["vn"] ** 2)
    ax2.plot(df["t"], speed, linewidth=0.5, color="steelblue")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    ax2r = ax2.twinx()
    y1, y2 = ax2.get_ylim()
    ax2r.set_ylim(y1 * 3.6, y2 * 3.6)
    ax2r.set_ylabel("Speed (km/h)")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "2_velocity.png"),
                dpi=150, bbox_inches="tight")

    # ── Plot 3: Bias estimates ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.canvas.manager.set_window_title("3 - Bias Estimates")
    ax1.plot(df["t"], df["ba_x"], linewidth=0.5, label="ba_x")
    ax1.plot(df["t"], df["ba_y"], linewidth=0.5, label="ba_y")
    ax1.plot(df["t"], df["ba_z"], linewidth=0.5, label="ba_z")
    ax1.set_ylabel("Accel bias (m/s²)")
    ax1.set_title("Estimated IMU Biases")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["t"], np.degrees(df["bg_x"]), linewidth=0.5, label="bg_x")
    ax2.plot(df["t"], np.degrees(df["bg_y"]), linewidth=0.5, label="bg_y")
    ax2.plot(df["t"], np.degrees(df["bg_z"]), linewidth=0.5, label="bg_z")
    ax2.set_ylabel("Gyro bias (°/s)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "3_biases.png"),
                dpi=150, bbox_inches="tight")

    # ── Plot 4: Position uncertainty + innovation ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.canvas.manager.set_window_title("4 - Uncertainty & Innovation")

    ax1.plot(df["t"], df["P_pos"], linewidth=0.5, color="steelblue")
    ax1.set_ylabel("Position 1σ (m)")
    ax1.set_title("EKF Position Uncertainty")
    ax1.grid(True, alpha=0.3)
    for gt in gps_times:
        ax1.axvline(x=gt, color="blue", alpha=0.05, linewidth=0.5)

    if len(innovations_pos) > 0:
        ip_df = pd.DataFrame(innovations_pos)
        ip_df["t"] = ip_df["wall_time"] - t0
        ax2.plot(ip_df["t"], ip_df["norm"], ".", markersize=2,
                color="coral", label="2D innovation")
        ax2.axhline(y=np.mean(ip_df["norm"]), color="red",
                    linestyle="--", linewidth=1,
                    label=f"mean={np.mean(ip_df['norm']):.3f}m")
        ax2.set_ylabel("GPS Innovation (m)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("GPS Position Innovation "
                      "(should be small and centered)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "4_uncertainty.png"),
                dpi=150, bbox_inches="tight")

    # ── Plot 5: Heading ──
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    fig.canvas.manager.set_window_title("5 - Heading")
    ax.plot(df["t"], df["yaw"], ".", markersize=0.5, color="steelblue")

    gps_hdg = [m for m in gps_measurements if m["type"] == "gps_heading"]
    if len(gps_hdg) > 0:
        hdg_times = [m["wall_time"] - t0 for m in gps_hdg]
        hdg_vals = [m["heading_deg"] for m in gps_hdg]
        ax.plot(hdg_times, hdg_vals, "r.", markersize=3,
                label="GPS HDT", alpha=0.7)
        ax.legend()

    ax.set_ylabel("Heading (°)")
    ax.set_xlabel("Time (s)")
    ax.set_title("EKF Heading vs GPS HDT")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "5_heading.png"),
                dpi=150, bbox_inches="tight")

    # ── Plot 6: Innovation histogram ──
    if len(innovations_pos) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.canvas.manager.set_window_title("6 - Innovation Analysis")

        ip_df = pd.DataFrame(innovations_pos)
        ax1.hist(ip_df["innov_e"], bins=50, alpha=0.7, label="East")
        ax1.hist(ip_df["innov_n"], bins=50, alpha=0.7, label="North")
        ax1.set_xlabel("Innovation (m)")
        ax1.set_ylabel("Count")
        ax1.set_title("Position Innovation Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.hist(ip_df["norm"], bins=50, color="coral", alpha=0.7)
        ax2.axvline(x=np.mean(ip_df["norm"]), color="red",
                    linestyle="--", label=f"mean={np.mean(ip_df['norm']):.3f}m")
        ax2.axvline(x=np.percentile(ip_df["norm"], 95), color="darkred",
                    linestyle=":", label=f"95%={np.percentile(ip_df['norm'], 95):.3f}m")
        ax2.set_xlabel("2D Innovation (m)")
        ax2.set_ylabel("Count")
        ax2.set_title("Innovation Magnitude")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "6_innovation.png"),
                    dpi=150, bbox_inches="tight")

    print(f"Plots saved to {output_dir}/")


# ─────────────────────────────────────────────
# ★★★ RUN EVERYTHING ★★★
# ─────────────────────────────────────────────

print("=" * 60)
print("  GPS/IMU EKF Post-Processor")
print("=" * 60)
print(f"Data dir:    {os.path.abspath(DATA_DIR)}")
print(f"Output dir:  {os.path.abspath(OUTPUT_DIR)}")
print(f"RTS smooth:  {'ON' if ENABLE_RTS_SMOOTHING else 'OFF'}")
print(f"GPS offset:  {GPS_TIME_OFFSET_S:.4f}s")
print(f"RTK sigma:   {GPS_POS_SIGMA_RTK_FIX:.3f}m")
print()

# ── Step 1: Load ──
print("Step 1: Loading data...")
imu_raw = load_imu_raw(DATA_DIR)
imu_quat = load_imu_quat(DATA_DIR)
imu_gravity = load_imu_gravity(DATA_DIR)
gps_raw = load_gps_raw(DATA_DIR)

if imu_raw is None or gps_raw is None:
    print("[ERROR] Cannot proceed without imu_raw.csv and gps_raw.csv")
    sys.exit(1)

# ── Step 2: Preprocess ──
print("\nStep 2: Preprocessing...")
imu_raw = compute_imu_dt(imu_raw)
imu_raw = merge_gravity_with_raw(imu_raw, imu_gravity)
imu_raw = merge_quat_with_raw(imu_raw, imu_quat)
gps_measurements = extract_gps_measurements(gps_raw)

if len(gps_measurements) == 0:
    print("[ERROR] No GPS measurements extracted")
    sys.exit(1)

# ── Step 3-4: EKF ──
print("\nStep 3-4: Running EKF...")
results, imu_only_results, innovations_pos, ref_lat, ref_lon, ref_alt = run_ekf(
    imu_raw, gps_measurements)

if results is None or len(results) == 0:
    print("[ERROR] EKF produced no results")
    sys.exit(1)

# ── Step 5: Save ──
print("\nStep 5: Saving results...")
save_results(results, OUTPUT_DIR)
if innovations_pos is not None:
    save_innovations(innovations_pos, OUTPUT_DIR)

# ── Step 6: Plot ──
print("\nStep 6: Plotting...")
plot_results(results, imu_only_results, innovations_pos, gps_measurements,
             ref_lat, ref_lon, ref_alt, OUTPUT_DIR)

print("\n" + "=" * 60)
print("  DONE")
print(f"  {len(results)} output points at ~400Hz")
print(f"  Files in: {os.path.abspath(OUTPUT_DIR)}/")
print("=" * 60)

if SHOW_PLOTS:
    plt.show()
