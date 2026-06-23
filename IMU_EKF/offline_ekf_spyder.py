from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Spyder-friendly offline EKF diagnostic script.
# Run this whole file directly. No main function is used.

if "__file__" in globals():
    DATA_DIR = Path(__file__).resolve().parent
elif (Path.cwd() / "IMU_EKF").exists():
    DATA_DIR = Path.cwd() / "IMU_EKF"
else:
    DATA_DIR = Path.cwd()

AUTO_FIND_LATEST_GROUP = True
STAMP = None  # Example: "20260617_155318". Ignored when AUTO_FIND_LATEST_GROUP is True.

SHOW_FIGURES = True
SAVE_FIGURES = True

# IMU axis assumptions. Change these first if the offline EKF moves the wrong way.
# Vehicle convention used here: X forward, Y right, Z down. Heading is deg clockwise from north.
IMU_FORWARD_ACCEL_COL = "accel_x_g"
IMU_RIGHT_ACCEL_COL = "accel_y_g"
IMU_YAW_RATE_COL = "gyro_z_radps"
ACCEL_FORWARD_SIGN = 1.0
ACCEL_RIGHT_SIGN = 1.0
YAW_RATE_SIGN = 1.0
YAW_OFFSET_DEG = 0.0

# Measurement usage.
USE_GPS_POSITION = True
USE_GPS_VELOCITY = True
USE_GPS_HEADING = True

# Noise tuning. Start conservative; GPS velocity sigma from the receiver can be too optimistic indoors.
MIN_POS_SIGMA_M = 0.50
MIN_VEL_SIGMA_MPS = 0.10
MIN_HEADING_SIGMA_DEG = 3.0
GPS_POS_SIGMA_SCALE = 1.0
GPS_VEL_SIGMA_SCALE = 5.0
GPS_HEADING_SIGMA_SCALE = 2.0

# Process noise. Larger values make the EKF rely more on GPS updates.
ACCEL_PROCESS_NOISE_MPS2 = 1.5
YAW_RATE_PROCESS_NOISE_RADPS = np.deg2rad(5.0)
GYRO_BIAS_RW_RADPS = np.deg2rad(0.05)
ACCEL_BIAS_RW_MPS2 = 0.02

GRAVITY_MPS2 = 9.80665
DOWNSAMPLE_IMU_PLOT = 10


def latest_complete_stamp(data_dir):
    ekf_files = sorted(data_dir.glob("cv7_ekf_fused_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    required_templates = [
        "cv7_ekf_fused_{stamp}.csv",
        "cv7_imu_sensor_{stamp}.csv",
        "cv7_teensy_gps_aid_{stamp}.csv",
    ]
    for ekf_file in ekf_files:
        stamp = ekf_file.stem.replace("cv7_ekf_fused_", "")
        if all((data_dir / t.format(stamp=stamp)).exists() for t in required_templates):
            return stamp
    raise FileNotFoundError("No complete CSV group found in " + str(data_dir))


def read_csv(path):
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {path.name}: {len(df)} rows")
    return df


def to_num(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def wrap_rad(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_deg_signed(angle):
    return (angle + 180.0) % 360.0 - 180.0


def heading_deg_to_rad(heading_deg):
    return np.deg2rad(wrap_deg_signed(heading_deg + YAW_OFFSET_DEG))


def add_local_enu(df, lat0, lon0):
    earth_radius_m = 6378137.0
    lat0_rad = np.deg2rad(lat0)
    df["east_m"] = np.deg2rad(df["longitude_deg"] - lon0) * earth_radius_m * np.cos(lat0_rad)
    df["north_m"] = np.deg2rad(df["latitude_deg"] - lat0) * earth_radius_m
    return df


def finite_rows(df, cols, optional_cols=None):
    optional_cols = optional_cols or []
    needed = ["host_time_unix_s"] + cols
    keep_cols = needed + [col for col in optional_cols if col in df.columns and col not in needed]
    out = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
    return out.sort_values("host_time_unix_s").reset_index(drop=True)


def ekf_update_linear(x, P, z, H, R):
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    I = np.eye(P.shape[0])
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    return x, P, y, S


def ekf_update_yaw(x, P, yaw_meas_rad, sigma_rad):
    H = np.zeros((1, 8))
    H[0, 4] = 1.0
    y = np.array([wrap_rad(yaw_meas_rad - x[4])])
    S = H @ P @ H.T + np.array([[sigma_rad ** 2]])
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    x[4] = wrap_rad(x[4])
    I = np.eye(P.shape[0])
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ np.array([[sigma_rad ** 2]]) @ K.T
    return x, P, y, S


if AUTO_FIND_LATEST_GROUP:
    STAMP = latest_complete_stamp(DATA_DIR)
elif STAMP is None:
    raise ValueError("Set STAMP or enable AUTO_FIND_LATEST_GROUP")

paths = {
    "imu": DATA_DIR / f"cv7_imu_sensor_{STAMP}.csv",
    "teensy": DATA_DIR / f"cv7_teensy_gps_aid_{STAMP}.csv",
    "ekf": DATA_DIR / f"cv7_ekf_fused_{STAMP}.csv",
}
output_dir = DATA_DIR / f"offline_ekf_plots_{STAMP}"

imu = read_csv(paths["imu"])
gps = read_csv(paths["teensy"])
cv7 = read_csv(paths["ekf"])

to_num(
    imu,
    [
        "host_time_unix_s",
        IMU_FORWARD_ACCEL_COL,
        IMU_RIGHT_ACCEL_COL,
        IMU_YAW_RATE_COL,
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
        "vel_sigma_n_mps",
        "vel_sigma_e_mps",
        "vel_sigma_d_mps",
        "heading_valid",
        "heading_deg",
        "heading_sigma_deg",
    ],
)
to_num(
    cv7,
    [
        "host_time_unix_s",
        "gps_tow_s",
        "latitude_deg",
        "longitude_deg",
        "ellipsoid_height_m",
        "vel_n_mps",
        "vel_e_mps",
        "yaw_deg",
        "position_valid",
        "velocity_valid",
        "attitude_valid",
    ],
)

imu = finite_rows(imu, [IMU_FORWARD_ACCEL_COL, IMU_RIGHT_ACCEL_COL, IMU_YAW_RATE_COL])
gps = finite_rows(
    gps,
    [
        "gps_tow_s",
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "vel_n_mps",
        "vel_e_mps",
        "pos_sigma_n_m",
        "pos_sigma_e_m",
        "vel_sigma_n_mps",
        "vel_sigma_e_mps",
    ],
    optional_cols=["heading_valid", "heading_deg", "heading_sigma_deg"],
)

if imu.empty:
    raise ValueError("No usable IMU accel/gyro rows found.")
if gps.empty:
    raise ValueError("No usable Teensy GPS aiding rows found.")

heading_valid = (
    pd.to_numeric(gps.get("heading_valid", np.nan), errors="coerce").fillna(0).to_numpy() == 1
    if "heading_valid" in gps.columns
    else np.zeros(len(gps), dtype=bool)
)
gps["heading_valid_bool"] = heading_valid

lat0 = float(gps["latitude_deg"].median())
lon0 = float(gps["longitude_deg"].median())
gps = add_local_enu(gps, lat0, lon0)

cv7_valid = cv7[
    (cv7.get("position_valid") == 1)
    & (cv7.get("velocity_valid") == 1)
    & (cv7.get("attitude_valid") == 1)
    & cv7["latitude_deg"].notna()
    & cv7["longitude_deg"].notna()
].copy()
if not cv7_valid.empty:
    cv7_valid = add_local_enu(cv7_valid, lat0, lon0)

time_offset_tow_minus_host = float(np.nanmedian(gps["gps_tow_s"] - gps["host_time_unix_s"]))
t0_host = float(max(imu["host_time_unix_s"].min(), gps["host_time_unix_s"].min()))
t1_host = float(min(imu["host_time_unix_s"].max(), gps["host_time_unix_s"].max()))

imu = imu[(imu["host_time_unix_s"] >= t0_host) & (imu["host_time_unix_s"] <= t1_host)].reset_index(drop=True)
gps = gps[(gps["host_time_unix_s"] >= t0_host) & (gps["host_time_unix_s"] <= t1_host)].reset_index(drop=True)

first_gps = gps.iloc[0]
if bool(first_gps.get("heading_valid_bool", False)) and np.isfinite(first_gps.get("heading_deg", np.nan)):
    init_yaw = heading_deg_to_rad(float(first_gps["heading_deg"]))
else:
    init_yaw = np.arctan2(float(first_gps["vel_e_mps"]), float(first_gps["vel_n_mps"]))

# State: east, north, vel_e, vel_n, yaw, gyro_bias_z, accel_bias_forward, accel_bias_right
x = np.array(
    [
        float(first_gps["east_m"]),
        float(first_gps["north_m"]),
        float(first_gps["vel_e_mps"]),
        float(first_gps["vel_n_mps"]),
        float(init_yaw),
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)
P = np.diag(
    [
        5.0 ** 2,
        5.0 ** 2,
        1.0 ** 2,
        1.0 ** 2,
        np.deg2rad(20.0) ** 2,
        np.deg2rad(5.0) ** 2,
        0.5 ** 2,
        0.5 ** 2,
    ]
)

gps_index = 0
states = []
innovations = []
last_t = float(imu.iloc[0]["host_time_unix_s"])

for _, row in imu.iterrows():
    t = float(row["host_time_unix_s"])
    dt = t - last_t
    last_t = t
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.002
    dt = min(dt, 0.05)

    yaw = x[4]
    gyro_z = YAW_RATE_SIGN * float(row[IMU_YAW_RATE_COL])
    accel_forward = ACCEL_FORWARD_SIGN * float(row[IMU_FORWARD_ACCEL_COL]) * GRAVITY_MPS2 - x[6]
    accel_right = ACCEL_RIGHT_SIGN * float(row[IMU_RIGHT_ACCEL_COL]) * GRAVITY_MPS2 - x[7]

    yaw_rate = gyro_z - x[5]
    sin_y = np.sin(yaw)
    cos_y = np.cos(yaw)
    accel_e = sin_y * accel_forward + cos_y * accel_right
    accel_n = cos_y * accel_forward - sin_y * accel_right

    x[0] += x[2] * dt + 0.5 * accel_e * dt * dt
    x[1] += x[3] * dt + 0.5 * accel_n * dt * dt
    x[2] += accel_e * dt
    x[3] += accel_n * dt
    x[4] = wrap_rad(x[4] + yaw_rate * dt)

    F = np.eye(8)
    F[0, 2] = dt
    F[1, 3] = dt
    F[2, 4] = (cos_y * accel_forward - sin_y * accel_right) * dt
    F[3, 4] = (-sin_y * accel_forward - cos_y * accel_right) * dt
    F[2, 6] = -sin_y * dt
    F[2, 7] = -cos_y * dt
    F[3, 6] = -cos_y * dt
    F[3, 7] = sin_y * dt
    F[4, 5] = -dt

    q = np.array(
        [
            0.25 * ACCEL_PROCESS_NOISE_MPS2 ** 2 * dt ** 4,
            0.25 * ACCEL_PROCESS_NOISE_MPS2 ** 2 * dt ** 4,
            ACCEL_PROCESS_NOISE_MPS2 ** 2 * dt ** 2,
            ACCEL_PROCESS_NOISE_MPS2 ** 2 * dt ** 2,
            YAW_RATE_PROCESS_NOISE_RADPS ** 2 * dt ** 2,
            GYRO_BIAS_RW_RADPS ** 2 * dt,
            ACCEL_BIAS_RW_MPS2 ** 2 * dt,
            ACCEL_BIAS_RW_MPS2 ** 2 * dt,
        ]
    )
    P = F @ P @ F.T + np.diag(q)

    while gps_index < len(gps) and float(gps.iloc[gps_index]["host_time_unix_s"]) <= t:
        meas = gps.iloc[gps_index]

        if USE_GPS_POSITION:
            z = np.array([float(meas["east_m"]), float(meas["north_m"])])
            H = np.zeros((2, 8))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            sigma_e = max(MIN_POS_SIGMA_M, GPS_POS_SIGMA_SCALE * float(meas["pos_sigma_e_m"]))
            sigma_n = max(MIN_POS_SIGMA_M, GPS_POS_SIGMA_SCALE * float(meas["pos_sigma_n_m"]))
            R = np.diag([sigma_e ** 2, sigma_n ** 2])
            x, P, y, S = ekf_update_linear(x, P, z, H, R)
            innovations.append({"host_time_unix_s": t, "type": "pos", "e": y[0], "n": y[1]})

        if USE_GPS_VELOCITY:
            z = np.array([float(meas["vel_e_mps"]), float(meas["vel_n_mps"])])
            H = np.zeros((2, 8))
            H[0, 2] = 1.0
            H[1, 3] = 1.0
            sigma_e = max(MIN_VEL_SIGMA_MPS, GPS_VEL_SIGMA_SCALE * float(meas["vel_sigma_e_mps"]))
            sigma_n = max(MIN_VEL_SIGMA_MPS, GPS_VEL_SIGMA_SCALE * float(meas["vel_sigma_n_mps"]))
            R = np.diag([sigma_e ** 2, sigma_n ** 2])
            x, P, y, S = ekf_update_linear(x, P, z, H, R)
            innovations.append({"host_time_unix_s": t, "type": "vel", "e": y[0], "n": y[1]})

        if (
            USE_GPS_HEADING
            and bool(meas.get("heading_valid_bool", False))
            and np.isfinite(meas.get("heading_deg", np.nan))
        ):
            sigma_deg = meas.get("heading_sigma_deg", np.nan)
            if not np.isfinite(sigma_deg):
                sigma_deg = MIN_HEADING_SIGMA_DEG
            sigma_rad = np.deg2rad(max(MIN_HEADING_SIGMA_DEG, GPS_HEADING_SIGMA_SCALE * float(sigma_deg)))
            yaw_meas = heading_deg_to_rad(float(meas["heading_deg"]))
            x, P, y, S = ekf_update_yaw(x, P, yaw_meas, sigma_rad)
            innovations.append({"host_time_unix_s": t, "type": "yaw", "yaw_deg": np.rad2deg(y[0])})

        gps_index += 1

    states.append(
        {
            "host_time_unix_s": t,
            "plot_time_s": t + time_offset_tow_minus_host - float(gps["gps_tow_s"].iloc[0]),
            "east_m": x[0],
            "north_m": x[1],
            "vel_e_mps": x[2],
            "vel_n_mps": x[3],
            "speed_mps": float(np.hypot(x[2], x[3])),
            "yaw_deg": wrap_deg_signed(float(np.rad2deg(x[4]))),
            "gyro_bias_z_radps": x[5],
            "accel_bias_forward_mps2": x[6],
            "accel_bias_right_mps2": x[7],
        }
    )

est = pd.DataFrame(states)
innov = pd.DataFrame(innovations)
gps["plot_time_s"] = gps["gps_tow_s"] - float(gps["gps_tow_s"].iloc[0])
gps["speed_mps"] = np.sqrt(gps["vel_e_mps"] ** 2 + gps["vel_n_mps"] ** 2)
gps["yaw_deg"] = wrap_deg_signed(gps["heading_deg"] + YAW_OFFSET_DEG)

if not cv7_valid.empty:
    cv7_valid["plot_time_s"] = cv7_valid["gps_tow_s"] - float(gps["gps_tow_s"].iloc[0])
    cv7_valid["speed_mps"] = np.sqrt(cv7_valid["vel_e_mps"] ** 2 + cv7_valid["vel_n_mps"] ** 2)
    cv7_valid["yaw_deg"] = wrap_deg_signed(cv7_valid["yaw_deg"])

print("\nOffline EKF group:", STAMP)
print("IMU rows used:", len(imu), "GPS rows used:", len(gps), "EKF states:", len(est))
print("Local origin lat/lon:", f"{lat0:.10f}", f"{lon0:.10f}")
print("Initial yaw deg:", f"{np.rad2deg(init_yaw):.3f}")
print("Final estimated biases:")
print("  gyro z bias rad/s:", f"{est['gyro_bias_z_radps'].iloc[-1]:.6g}")
print("  accel forward bias m/s^2:", f"{est['accel_bias_forward_mps2'].iloc[-1]:.6g}")
print("  accel right bias m/s^2:", f"{est['accel_bias_right_mps2'].iloc[-1]:.6g}")

if not innov.empty:
    print("Innovation RMS:")
    for kind in sorted(innov["type"].unique()):
        part = innov[innov["type"] == kind]
        if kind in ("pos", "vel"):
            rms_e = float(np.sqrt(np.nanmean(part["e"] ** 2)))
            rms_n = float(np.sqrt(np.nanmean(part["n"] ** 2)))
            print(f"  {kind}: east {rms_e:.4g}, north {rms_n:.4g}")
        elif kind == "yaw":
            rms = float(np.sqrt(np.nanmean(part["yaw_deg"] ** 2)))
            print(f"  yaw: {rms:.4g} deg")

output_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(9, 8))
ax.scatter(gps["east_m"], gps["north_m"], s=18, c="green", label="Teensy GPS", alpha=0.85)
ax.scatter(est["east_m"].iloc[::DOWNSAMPLE_IMU_PLOT], est["north_m"].iloc[::DOWNSAMPLE_IMU_PLOT], s=10, c="purple", label="Offline EKF", alpha=0.85)
if not cv7_valid.empty:
    ax.scatter(cv7_valid["east_m"].iloc[::DOWNSAMPLE_IMU_PLOT], cv7_valid["north_m"].iloc[::DOWNSAMPLE_IMU_PLOT], s=10, c="red", label="CV7 EKF", alpha=0.65)
ax.set_title(f"Offline EKF local ENU overlay - {STAMP}")
ax.set_xlabel("east from shared origin (m)")
ax.set_ylabel("north from shared origin (m)")
ax.axis("equal")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
if SAVE_FIGURES:
    fig.savefig(output_dir / f"offline_ekf_position_{STAMP}.png", dpi=180, bbox_inches="tight")

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
axes[0].scatter(gps["plot_time_s"], gps["east_m"], s=18, c="green", label="GPS east")
axes[0].scatter(est["plot_time_s"].iloc[::DOWNSAMPLE_IMU_PLOT], est["east_m"].iloc[::DOWNSAMPLE_IMU_PLOT], s=10, c="purple", label="Offline EKF east")
if not cv7_valid.empty:
    axes[0].scatter(cv7_valid["plot_time_s"].iloc[::DOWNSAMPLE_IMU_PLOT], cv7_valid["east_m"].iloc[::DOWNSAMPLE_IMU_PLOT], s=10, c="red", label="CV7 EKF east")
axes[0].set_ylabel("east (m)")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best")

axes[1].scatter(gps["plot_time_s"], gps["north_m"], s=18, c="green", label="GPS north")
axes[1].scatter(est["plot_time_s"].iloc[::DOWNSAMPLE_IMU_PLOT], est["north_m"].iloc[::DOWNSAMPLE_IMU_PLOT], s=10, c="purple", label="Offline EKF north")
if not cv7_valid.empty:
    axes[1].scatter(cv7_valid["plot_time_s"].iloc[::DOWNSAMPLE_IMU_PLOT], cv7_valid["north_m"].iloc[::DOWNSAMPLE_IMU_PLOT], s=10, c="red", label="CV7 EKF north")
axes[1].set_ylabel("north (m)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="best")

axes[2].scatter(gps["plot_time_s"], gps["speed_mps"], s=18, c="green", label="GPS speed")
axes[2].scatter(est["plot_time_s"].iloc[::DOWNSAMPLE_IMU_PLOT], est["speed_mps"].iloc[::DOWNSAMPLE_IMU_PLOT], s=10, c="purple", label="Offline EKF speed")
if not cv7_valid.empty:
    axes[2].scatter(cv7_valid["plot_time_s"].iloc[::DOWNSAMPLE_IMU_PLOT], cv7_valid["speed_mps"].iloc[::DOWNSAMPLE_IMU_PLOT], s=10, c="red", label="CV7 EKF speed")
axes[2].set_xlabel("GPS TOW - first GPS sample (s)")
axes[2].set_ylabel("speed (m/s)")
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc="best")
fig.suptitle(f"Offline EKF position/speed time series - {STAMP}")
fig.tight_layout()
if SAVE_FIGURES:
    fig.savefig(output_dir / f"offline_ekf_time_series_{STAMP}.png", dpi=180, bbox_inches="tight")

fig, ax = plt.subplots(1, 1, figsize=(13, 4))
ax.scatter(gps.loc[gps["heading_valid_bool"], "plot_time_s"], gps.loc[gps["heading_valid_bool"], "yaw_deg"], s=18, c="green", label="GPS heading")
ax.scatter(est["plot_time_s"].iloc[::DOWNSAMPLE_IMU_PLOT], est["yaw_deg"].iloc[::DOWNSAMPLE_IMU_PLOT], s=10, c="purple", label="Offline EKF yaw")
if not cv7_valid.empty:
    ax.scatter(cv7_valid["plot_time_s"].iloc[::DOWNSAMPLE_IMU_PLOT], cv7_valid["yaw_deg"].iloc[::DOWNSAMPLE_IMU_PLOT], s=10, c="red", label="CV7 EKF yaw")
ax.set_title(f"Offline EKF yaw comparison - {STAMP}")
ax.set_xlabel("GPS TOW - first GPS sample (s)")
ax.set_ylabel("yaw/heading deg (-180 to 180)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
if SAVE_FIGURES:
    fig.savefig(output_dir / f"offline_ekf_yaw_{STAMP}.png", dpi=180, bbox_inches="tight")

if SHOW_FIGURES:
    plt.show()
else:
    plt.close("all")
