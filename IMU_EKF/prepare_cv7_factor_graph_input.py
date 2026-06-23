from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot


# Convert CV7_INS_EKF.py CSV logs into the recorder-format CSVs expected by:
# test_code/recording_code/factor_graph_gps_imu_own_data.py
#
# Spyder-friendly: run this file directly.

if "__file__" in globals():
    DATA_DIR = Path(__file__).resolve().parent
else:
    DATA_DIR = Path.cwd() / "IMU_EKF" if (Path.cwd() / "IMU_EKF").exists() else Path.cwd()

AUTO_FIND_LATEST_GROUP = True
STAMP = None  # Example: "20260617_155318". Ignored when AUTO_FIND_LATEST_GROUP is True.

GRAVITY_MPS2 = 9.80665
USE_GPS_HEADING_FOR_IMU_YAW = True
DEFAULT_GRAVITY_BODY_G = np.array([0.0, 0.0, -1.0])


def latest_complete_stamp(data_dir):
    ekf_files = sorted(data_dir.glob("cv7_ekf_fused_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    required = [
        "cv7_imu_sensor_{stamp}.csv",
        "cv7_teensy_gps_aid_{stamp}.csv",
        "cv7_ekf_fused_{stamp}.csv",
    ]
    for ekf_file in ekf_files:
        stamp = ekf_file.stem.replace("cv7_ekf_fused_", "")
        if all((data_dir / name.format(stamp=stamp)).exists() for name in required):
            return stamp
    raise FileNotFoundError("No complete CV7 CSV group found in " + str(data_dir))


def to_num(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def wrap_deg_signed(values):
    return (values + 180.0) % 360.0 - 180.0


def heading_to_body_to_enu_quat_wxyz(heading_deg):
    heading_rad = np.deg2rad(heading_deg)
    s = np.sin(heading_rad)
    c = np.cos(heading_rad)
    # Columns are the ENU vectors of body X-forward, Y-right, Z-up.
    rot_mat = np.array(
        [
            [s, c, 0.0],
            [c, -s, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # The matrix above is improper because body X/Y/Zdown convention differs
    # from ENU Z-up. Use a yaw-only proper rotation for heading diagnostics.
    yaw_enu = np.deg2rad(90.0 - heading_deg)
    q_xyzw = Rot.from_euler("z", yaw_enu).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


if AUTO_FIND_LATEST_GROUP:
    STAMP = latest_complete_stamp(DATA_DIR)
elif STAMP is None:
    raise ValueError("Set STAMP or enable AUTO_FIND_LATEST_GROUP")

imu_path = DATA_DIR / f"cv7_imu_sensor_{STAMP}.csv"
teensy_path = DATA_DIR / f"cv7_teensy_gps_aid_{STAMP}.csv"
ekf_path = DATA_DIR / f"cv7_ekf_fused_{STAMP}.csv"

output_dir = DATA_DIR / f"factor_graph_input_{STAMP}"
output_dir.mkdir(parents=True, exist_ok=True)

imu = pd.read_csv(imu_path, low_memory=False)
gps = pd.read_csv(teensy_path, low_memory=False)
ekf = pd.read_csv(ekf_path, low_memory=False)

to_num(
    imu,
    [
        "host_time_unix_s",
        "accel_x_g",
        "accel_y_g",
        "accel_z_g",
        "gyro_x_radps",
        "gyro_y_radps",
        "gyro_z_radps",
        "q_w",
        "q_x",
        "q_y",
        "q_z",
    ],
)
to_num(
    gps,
    [
        "host_time_unix_s",
        "gps_tow_s",
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "vel_n_mps",
        "vel_e_mps",
        "vel_d_mps",
        "pos_sigma_n_m",
        "pos_sigma_e_m",
        "pos_sigma_u_m",
        "heading_valid",
        "heading_deg",
        "heading_sigma_deg",
        "mode",
    ],
)
to_num(ekf, ["host_time_unix_s", "gps_tow_s", "yaw_deg", "attitude_valid"])

imu = imu.dropna(
    subset=[
        "host_time_unix_s",
        "accel_x_g",
        "accel_y_g",
        "accel_z_g",
        "gyro_x_radps",
        "gyro_y_radps",
        "gyro_z_radps",
    ]
).copy()
gps = gps.dropna(
    subset=[
        "host_time_unix_s",
        "gps_tow_s",
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "vel_n_mps",
        "vel_e_mps",
    ]
).copy()

imu = imu.sort_values("host_time_unix_s").reset_index(drop=True)
gps = gps.sort_values("host_time_unix_s").reset_index(drop=True)

t0 = max(float(imu["host_time_unix_s"].min()), float(gps["host_time_unix_s"].min()))
t1 = min(float(imu["host_time_unix_s"].max()), float(gps["host_time_unix_s"].max()))
imu = imu[(imu["host_time_unix_s"] >= t0) & (imu["host_time_unix_s"] <= t1)].copy()
gps = gps[(gps["host_time_unix_s"] >= t0) & (gps["host_time_unix_s"] <= t1)].copy()

gps_t = gps["host_time_unix_s"].to_numpy(dtype=float)
gps_heading = gps["heading_deg"].where(gps["heading_valid"].fillna(0).astype(int) == 1)
gps_heading = pd.to_numeric(gps_heading, errors="coerce")
if gps_heading.notna().sum() >= 2 and USE_GPS_HEADING_FOR_IMU_YAW:
    heading_unwrapped = np.unwrap(np.deg2rad(gps_heading.interpolate(limit_direction="both").to_numpy(dtype=float)))
    imu_heading_rad = np.interp(imu["host_time_unix_s"].to_numpy(dtype=float), gps_t, heading_unwrapped)
    imu_heading_deg = wrap_deg_signed(np.rad2deg(imu_heading_rad))
else:
    ekf_yaw = ekf.dropna(subset=["host_time_unix_s", "yaw_deg"]).sort_values("host_time_unix_s")
    if len(ekf_yaw) < 2:
        imu_heading_deg = np.zeros(len(imu), dtype=float)
    else:
        yaw_unwrapped = np.unwrap(np.deg2rad(ekf_yaw["yaw_deg"].to_numpy(dtype=float)))
        imu_heading_rad = np.interp(
            imu["host_time_unix_s"].to_numpy(dtype=float),
            ekf_yaw["host_time_unix_s"].to_numpy(dtype=float),
            yaw_unwrapped,
        )
        imu_heading_deg = wrap_deg_signed(np.rad2deg(imu_heading_rad))

heading_rad = np.deg2rad(imu_heading_deg)
acc_forward_g = imu["accel_x_g"].to_numpy(dtype=float) - DEFAULT_GRAVITY_BODY_G[0]
acc_right_g = imu["accel_y_g"].to_numpy(dtype=float) - DEFAULT_GRAVITY_BODY_G[1]
acc_down_g = imu["accel_z_g"].to_numpy(dtype=float) - DEFAULT_GRAVITY_BODY_G[2]

acc_e = (np.sin(heading_rad) * acc_forward_g + np.cos(heading_rad) * acc_right_g) * GRAVITY_MPS2
acc_n = (np.cos(heading_rad) * acc_forward_g - np.sin(heading_rad) * acc_right_g) * GRAVITY_MPS2
acc_u = -acc_down_g * GRAVITY_MPS2

imu_timestamp_us = np.round((imu["host_time_unix_s"] - float(imu["host_time_unix_s"].iloc[0])) * 1e6).astype(np.int64)
imu_raw = pd.DataFrame(
    {
        "wall_time": imu["host_time_unix_s"].to_numpy(dtype=float),
        "imu_timestamp_us": imu_timestamp_us,
        "gyro_x_rad_s": imu["gyro_x_radps"].to_numpy(dtype=float),
        "gyro_y_rad_s": imu["gyro_y_radps"].to_numpy(dtype=float),
        "gyro_z_rad_s": imu["gyro_z_radps"].to_numpy(dtype=float),
        "accel_x_g": imu["accel_x_g"].to_numpy(dtype=float),
        "accel_y_g": imu["accel_y_g"].to_numpy(dtype=float),
        "accel_z_g": imu["accel_z_g"].to_numpy(dtype=float),
        "acc_e": acc_e,
        "acc_n": acc_n,
        "acc_u": acc_u,
    }
)
imu_raw.to_csv(output_dir / "imu_raw.csv", index=False)

imu_gravity = pd.DataFrame(
    {
        "wall_time": imu["host_time_unix_s"].to_numpy(dtype=float),
        "gravity_x_g": DEFAULT_GRAVITY_BODY_G[0],
        "gravity_y_g": DEFAULT_GRAVITY_BODY_G[1],
        "gravity_z_g": DEFAULT_GRAVITY_BODY_G[2],
    }
)
imu_gravity.to_csv(output_dir / "imu_gravity.csv", index=False)

quats = np.vstack([heading_to_body_to_enu_quat_wxyz(h) for h in imu_heading_deg])
imu_quat = pd.DataFrame(
    {
        "wall_time": imu["host_time_unix_s"].to_numpy(dtype=float),
        "q1": quats[:, 0],
        "q2": quats[:, 1],
        "q3": quats[:, 2],
        "q4": quats[:, 3],
    }
)
imu_quat.to_csv(output_dir / "imu_quat.csv", index=False)

imu_rpy = pd.DataFrame(
    {
        "wall_time": imu["host_time_unix_s"].to_numpy(dtype=float),
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": imu_heading_deg,
    }
)
imu_rpy.to_csv(output_dir / "imu_rpy.csv", index=False)

speed = np.sqrt(gps["vel_n_mps"] ** 2 + gps["vel_e_mps"] ** 2)
course_deg = (np.rad2deg(np.arctan2(gps["vel_e_mps"], gps["vel_n_mps"])) + 360.0) % 360.0
fix_quality = np.where(gps["mode"].fillna(1).astype(float) >= 4, 4, 1)
sigma_h = np.maximum(
    np.nan_to_num(
        np.maximum(gps["pos_sigma_n_m"].to_numpy(dtype=float), gps["pos_sigma_e_m"].to_numpy(dtype=float)),
        nan=3.0,
    ),
    0.5,
)
sigma_u = np.maximum(np.nan_to_num(gps["pos_sigma_u_m"].to_numpy(dtype=float), nan=6.0), 1.0)

gga = pd.DataFrame(
    {
        "wall_time": gps["host_time_unix_s"].to_numpy(dtype=float),
        "gps_utc": gps["gps_tow_s"].to_numpy(dtype=float),
        "msg_type": "GGA",
        "lat": gps["latitude_deg"].to_numpy(dtype=float),
        "lon": gps["longitude_deg"].to_numpy(dtype=float),
        "alt": gps["height_m"].to_numpy(dtype=float),
        "fix_quality": fix_quality,
        "lat_err_m": sigma_h,
        "lon_err_m": sigma_h,
        "alt_err_m": sigma_u,
        "speed_kmh": np.nan,
        "speed_knots": np.nan,
        "course_true_deg": np.nan,
        "heading_deg": np.nan,
        "rmc_status": "",
    }
)
vtg = pd.DataFrame(
    {
        "wall_time": gps["host_time_unix_s"].to_numpy(dtype=float),
        "gps_utc": gps["gps_tow_s"].to_numpy(dtype=float),
        "msg_type": "VTG",
        "lat": np.nan,
        "lon": np.nan,
        "alt": np.nan,
        "fix_quality": np.nan,
        "lat_err_m": np.nan,
        "lon_err_m": np.nan,
        "alt_err_m": np.nan,
        "speed_kmh": speed.to_numpy(dtype=float) * 3.6,
        "speed_knots": speed.to_numpy(dtype=float) / 0.514444,
        "course_true_deg": course_deg.to_numpy(dtype=float),
        "heading_deg": np.nan,
        "rmc_status": "",
    }
)
heading_rows = gps[gps["heading_valid"].fillna(0).astype(int) == 1].copy()
hdt = pd.DataFrame(
    {
        "wall_time": heading_rows["host_time_unix_s"].to_numpy(dtype=float),
        "gps_utc": heading_rows["gps_tow_s"].to_numpy(dtype=float),
        "msg_type": "HDT",
        "lat": np.nan,
        "lon": np.nan,
        "alt": np.nan,
        "fix_quality": np.nan,
        "lat_err_m": np.nan,
        "lon_err_m": np.nan,
        "alt_err_m": np.nan,
        "speed_kmh": np.nan,
        "speed_knots": np.nan,
        "course_true_deg": np.nan,
        "heading_deg": heading_rows["heading_deg"].to_numpy(dtype=float),
        "rmc_status": "",
    }
)
gps_raw = pd.concat([gga, vtg, hdt], ignore_index=True).sort_values(["wall_time", "msg_type"])
gps_raw.to_csv(output_dir / "gps_raw.csv", index=False)

print("Prepared CV7 factor graph input")
print("Stamp:", STAMP)
print("Output dir:", output_dir)
print("IMU rows:", len(imu_raw))
print("GPS GGA rows:", len(gga), "VTG rows:", len(vtg), "HDT rows:", len(hdt))
print()
print("Run factor graph with:")
print(
    "python test_code\\recording_code\\factor_graph_gps_imu_own_data.py "
    f"--data-dir {output_dir} "
    f"--output-dir {DATA_DIR / ('factor_graph_output_' + STAMP)} "
    "--duration-s 120"
)
