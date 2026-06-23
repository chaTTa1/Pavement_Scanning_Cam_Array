from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# Spyder-friendly configuration. Run this whole file directly.
if "__file__" in globals():
    DATA_DIR = Path(__file__).resolve().parent
elif (Path.cwd() / "IMU_EKF").exists():
    DATA_DIR = Path.cwd() / "IMU_EKF"
else:
    DATA_DIR = Path.cwd()
AUTO_FIND_LATEST_GROUP = True
STAMP = None  # Example: "20260615_221524". Ignored when AUTO_FIND_LATEST_GROUP is True.

SHOW_FIGURES = True
SAVE_FIGURES = True
USE_RELATIVE_TIME = True  # False: x axis is GPS TOW seconds. True: x axis starts at 0.
DOWNSAMPLE_EKF_FOR_PLOT = 5  # EKF is high-rate; increase if Spyder plotting feels slow.
POINT_SIZE = 18


def latest_complete_stamp(data_dir):
    ekf_files = sorted(data_dir.glob("cv7_ekf_fused_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    required_templates = [
        "cv7_ekf_fused_{stamp}.csv",
        "cv7_gps_raw_{stamp}.csv",
        "cv7_imu_sensor_{stamp}.csv",
        "cv7_teensy_gps_aid_{stamp}.csv",
    ]
    for ekf_file in ekf_files:
        stamp = ekf_file.stem.replace("cv7_ekf_fused_", "")
        if all((data_dir / t.format(stamp=stamp)).exists() for t in required_templates):
            return stamp
    raise FileNotFoundError("No complete CSV group found in " + str(data_dir))


def read_csv_file(path):
    if not path.exists():
        print("Missing:", path)
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {path.name}: {len(df)} rows")
    return df


def to_num(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_plot_time(df):
    if df.empty:
        df["plot_time_s"] = np.nan
        return df

    to_num(df, ["gps_tow_s", "sample_time_s", "host_time_unix_s"])
    if "gps_tow_s" in df.columns:
        plot_time = df["gps_tow_s"].copy()
    else:
        plot_time = pd.Series(np.nan, index=df.index, dtype=float)

    host_col = None
    if "sample_time_s" in df.columns and df["sample_time_s"].notna().any():
        host_col = "sample_time_s"
    elif "host_time_unix_s" in df.columns and df["host_time_unix_s"].notna().any():
        host_col = "host_time_unix_s"

    valid = plot_time.notna() & np.isfinite(plot_time)
    if host_col is not None and valid.sum() >= 2 and (~valid).any():
        host = df[host_col]
        order = np.argsort(host.to_numpy(dtype=float))
        host_sorted = host.iloc[order].to_numpy(dtype=float)
        tow_sorted = plot_time.iloc[order].to_numpy(dtype=float)
        valid_sorted = np.isfinite(host_sorted) & np.isfinite(tow_sorted)
        interp = np.interp(host_sorted, host_sorted[valid_sorted], tow_sorted[valid_sorted])
        filled_sorted = tow_sorted.copy()
        filled_sorted[~np.isfinite(filled_sorted)] = interp[~np.isfinite(filled_sorted)]
        filled = np.empty_like(filled_sorted)
        filled[order] = filled_sorted
        plot_time = pd.Series(filled, index=df.index)

    df["plot_time_s"] = plot_time
    return df


def fill_plot_time_from_host_reference(target, reference_frames, label):
    if target.empty or "host_time_unix_s" not in target.columns:
        return target

    to_num(target, ["host_time_unix_s", "plot_time_s"])
    offsets = []
    for ref in reference_frames:
        if ref.empty or "host_time_unix_s" not in ref.columns or "plot_time_s" not in ref.columns:
            continue
        to_num(ref, ["host_time_unix_s", "plot_time_s"])
        valid = (
            ref["host_time_unix_s"].replace([np.inf, -np.inf], np.nan).notna()
            & ref["plot_time_s"].replace([np.inf, -np.inf], np.nan).notna()
        )
        if valid.any():
            offsets.extend((ref.loc[valid, "plot_time_s"] - ref.loc[valid, "host_time_unix_s"]).to_list())

    if not offsets:
        print(f"No host-to-GPS-time reference available for {label}")
        return target

    offset = float(np.nanmedian(offsets))
    missing = (
        target["plot_time_s"].replace([np.inf, -np.inf], np.nan).isna()
        & target["host_time_unix_s"].replace([np.inf, -np.inf], np.nan).notna()
    )
    if missing.any():
        target.loc[missing, "plot_time_s"] = target.loc[missing, "host_time_unix_s"] + offset
        print(f"Filled {int(missing.sum())} {label} timestamps from host_time_unix_s using GPS TOW offset {offset:.6f}")
    return target


def finite_df(df, x_col, y_col):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame(columns=[x_col, y_col])
    to_num(df, [x_col, y_col])
    out = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    return out.sort_values(x_col)


def finite_count(df, y_col):
    if df.empty or y_col not in df.columns:
        return 0
    to_num(df, [y_col])
    return int(np.isfinite(df[y_col].to_numpy(dtype=float)).sum())


def wrap_deg_signed(values):
    return ((values + 180.0) % 360.0) - 180.0


def plot_series(ax, df, y_col, label, style="-", every=1):
    data = finite_df(df, "x_time_s", y_col)
    if every > 1 and len(data) > every:
        data = data.iloc[::every]
    if data.empty:
        print(f"No finite data for {label}: {y_col}")
        return
    ax.plot(data["x_time_s"], data[y_col], style, linewidth=1.2, label=label)


def plot_points(ax, df, y_col, label, marker="o", every=1):
    data = finite_df(df, "x_time_s", y_col)
    if every > 1 and len(data) > every:
        data = data.iloc[::every]
    if data.empty:
        print(f"No finite data for {label}: {y_col}")
        return
    ax.scatter(data["x_time_s"], data[y_col], s=18, marker=marker, alpha=0.85, label=label, zorder=5)


def plot_compare(ax, specs, title, ylabel, note_missing=True):
    plotted = 0
    missing = []
    for spec in specs:
        df = spec["df"]
        col = spec["col"]
        label = spec["label"]
        color = spec["color"]
        every = spec.get("every", 1)
        marker = spec.get("marker", "o")
        data = finite_df(df, "x_time_s", col)
        if every > 1 and len(data) > every:
            data = data.iloc[::every]
        if data.empty:
            print(f"No finite data for {label}: {col}")
            missing.append(label)
            continue
        ax.scatter(
            data["x_time_s"],
            data[col],
            s=POINT_SIZE,
            marker=marker,
            alpha=0.85,
            color=color,
            label=label,
            zorder=5,
        )
        plotted += 1
    setup_axis(ax, title, ylabel)
    if plotted == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
    elif note_missing and missing:
        ax.text(
            0.01,
            0.02,
            "No valid data: " + ", ".join(missing),
            transform=ax.transAxes,
            fontsize=8,
            color="0.35",
            va="bottom",
        )


def plot_xy_compare(ax, specs, title):
    plotted = 0
    missing = []
    for spec in specs:
        df = spec["df"]
        label = spec["label"]
        color = spec["color"]
        every = spec.get("every", 1)
        if df.empty or "east_m" not in df.columns or "north_m" not in df.columns:
            missing.append(label)
            continue
        data = df[["east_m", "north_m"]].replace([np.inf, -np.inf], np.nan).dropna()
        if every > 1 and len(data) > every:
            data = data.iloc[::every]
        if data.empty:
            print(f"No finite local XY data for {label}")
            missing.append(label)
            continue
        ax.scatter(
            data["east_m"],
            data["north_m"],
            s=POINT_SIZE,
            alpha=0.85,
            color=color,
            label=label,
            zorder=5,
        )
        plotted += 1
    ax.set_title(title)
    ax.set_xlabel("east from shared origin (m)")
    ax.set_ylabel("north from shared origin (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if plotted == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
    elif missing:
        ax.text(
            0.01,
            0.02,
            "No valid data: " + ", ".join(missing),
            transform=ax.transAxes,
            fontsize=8,
            color="0.35",
            va="bottom",
        )


def setup_axis(ax, title, ylabel):
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def save_fig(fig, name):
    if not SAVE_FIGURES:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print("Saved:", path)


def print_interpolated_difference(label, reference, measured, columns):
    if reference.empty or measured.empty:
        return

    ref = finite_df(reference, "plot_time_s", columns[0])
    meas = finite_df(measured, "plot_time_s", columns[0])
    for col in columns[1:]:
        ref_extra = finite_df(reference, "plot_time_s", col)
        meas_extra = finite_df(measured, "plot_time_s", col)
        ref = ref.merge(ref_extra, on="plot_time_s", how="inner")
        meas = meas.merge(meas_extra, on="plot_time_s", how="inner")

    if ref.empty or meas.empty:
        return

    t_ref = ref["plot_time_s"].to_numpy(dtype=float)
    t_meas = meas["plot_time_s"].to_numpy(dtype=float)
    mask = (t_meas >= t_ref.min()) & (t_meas <= t_ref.max())
    if mask.sum() < 2:
        return

    print(label)
    for col in columns:
        ref_interp = np.interp(t_meas[mask], t_ref, ref[col].to_numpy(dtype=float))
        err = meas.loc[mask, col].to_numpy(dtype=float) - ref_interp
        err = err[np.isfinite(err)]
        if len(err):
            print(
                f"  {col}: median error {np.nanmedian(err):.6g}, "
                f"RMSE {np.sqrt(np.nanmean(err ** 2)):.6g}, max abs {np.nanmax(np.abs(err)):.6g}"
            )


def add_local_xy(frames):
    lat_values = []
    lon_values = []
    for frame in frames:
        if frame.empty or "latitude_deg" not in frame.columns or "longitude_deg" not in frame.columns:
            continue
        lat_values.append(frame["latitude_deg"].replace([np.inf, -np.inf], np.nan).dropna())
        lon_values.append(frame["longitude_deg"].replace([np.inf, -np.inf], np.nan).dropna())

    if not lat_values or not lon_values:
        print("No latitude/longitude data available for local ENU conversion")
        return None, None

    lat0 = float(pd.concat(lat_values).median())
    lon0 = float(pd.concat(lon_values).median())
    lat0_rad = np.deg2rad(lat0)
    earth_radius_m = 6378137.0

    for frame in frames:
        if frame.empty or "latitude_deg" not in frame.columns or "longitude_deg" not in frame.columns:
            continue
        frame["east_m"] = np.deg2rad(frame["longitude_deg"] - lon0) * earth_radius_m * np.cos(lat0_rad)
        frame["north_m"] = np.deg2rad(frame["latitude_deg"] - lat0) * earth_radius_m

    print(f"Local ENU origin: lat0={lat0:.10f}, lon0={lon0:.10f}")
    return lat0, lon0


def print_position_error_m(label, reference, measured):
    needed = ["latitude_deg", "longitude_deg", "ellipsoid_height_m"]
    if reference.empty or measured.empty or any(col not in reference.columns or col not in measured.columns for col in needed):
        return

    ref = reference[["plot_time_s"] + needed].replace([np.inf, -np.inf], np.nan).dropna().sort_values("plot_time_s")
    meas = measured[["plot_time_s"] + needed].replace([np.inf, -np.inf], np.nan).dropna().sort_values("plot_time_s")
    if ref.empty or meas.empty:
        return

    t_ref = ref["plot_time_s"].to_numpy(dtype=float)
    t_meas = meas["plot_time_s"].to_numpy(dtype=float)
    mask = (t_meas >= t_ref.min()) & (t_meas <= t_ref.max())
    if mask.sum() < 2:
        return

    lat0 = float(np.nanmedian(ref["latitude_deg"]))
    earth_radius_m = 6378137.0
    dlat_m = (
        meas.loc[mask, "latitude_deg"].to_numpy(dtype=float)
        - np.interp(t_meas[mask], t_ref, ref["latitude_deg"].to_numpy(dtype=float))
    )
    dlat_m = np.deg2rad(dlat_m) * earth_radius_m
    dlon_m = (
        meas.loc[mask, "longitude_deg"].to_numpy(dtype=float)
        - np.interp(t_meas[mask], t_ref, ref["longitude_deg"].to_numpy(dtype=float))
    )
    dlon_m = np.deg2rad(dlon_m) * earth_radius_m * np.cos(np.deg2rad(lat0))
    dh_m = (
        meas.loc[mask, "ellipsoid_height_m"].to_numpy(dtype=float)
        - np.interp(t_meas[mask], t_ref, ref["ellipsoid_height_m"].to_numpy(dtype=float))
    )
    horiz_m = np.sqrt(dlat_m ** 2 + dlon_m ** 2)

    print(label)
    print(
        f"  horizontal: median {np.nanmedian(horiz_m):.6g} m, "
        f"RMSE {np.sqrt(np.nanmean(horiz_m ** 2)):.6g} m, max {np.nanmax(horiz_m):.6g} m"
    )
    print(
        f"  height: median error {np.nanmedian(dh_m):.6g} m, "
        f"RMSE {np.sqrt(np.nanmean(dh_m ** 2)):.6g} m, max abs {np.nanmax(np.abs(dh_m)):.6g} m"
    )


if AUTO_FIND_LATEST_GROUP:
    STAMP = latest_complete_stamp(DATA_DIR)
elif STAMP is None:
    raise ValueError("Set STAMP or enable AUTO_FIND_LATEST_GROUP")

paths = {
    "ekf": DATA_DIR / f"cv7_ekf_fused_{STAMP}.csv",
    "gps": DATA_DIR / f"cv7_gps_raw_{STAMP}.csv",
    "imu": DATA_DIR / f"cv7_imu_sensor_{STAMP}.csv",
    "teensy": DATA_DIR / f"cv7_teensy_gps_aid_{STAMP}.csv",
}

output_dir = DATA_DIR / f"csv_compare_plots_{STAMP}"

ekf = add_plot_time(read_csv_file(paths["ekf"]))
gps = add_plot_time(read_csv_file(paths["gps"]))
imu = add_plot_time(read_csv_file(paths["imu"]))
teensy = add_plot_time(read_csv_file(paths["teensy"]))

gps = fill_plot_time_from_host_reference(gps, [teensy, ekf, imu], "GPS raw")

numeric_cols = [
    "latitude_deg",
    "longitude_deg",
    "ellipsoid_height_m",
    "altitude_m",
    "geoid_separation_m",
    "height_m",
    "heading_deg",
    "heading_true_deg",
    "track_true_deg",
    "yaw_deg",
    "roll_deg",
    "pitch_deg",
    "vel_n_mps",
    "vel_e_mps",
    "vel_d_mps",
    "speed_mps",
    "ground_speed_mps",
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
    "gyro_x_radps",
    "gyro_y_radps",
    "gyro_z_radps",
    "num_sats",
    "num_sv",
    "heading_valid",
]
for frame in [ekf, gps, imu, teensy]:
    to_num(frame, numeric_cols)

if not gps.empty:
    if "heading_true_deg" in gps.columns:
        gps["heading_true_signed_deg"] = wrap_deg_signed(gps["heading_true_deg"])
    if "heading_deg" in gps.columns:
        gps["heading_signed_deg"] = wrap_deg_signed(gps["heading_deg"])
    if "track_true_deg" in gps.columns:
        gps["track_true_signed_deg"] = wrap_deg_signed(gps["track_true_deg"])
if not teensy.empty and "heading_deg" in teensy.columns:
    teensy["heading_signed_deg"] = wrap_deg_signed(teensy["heading_deg"])
if not ekf.empty and "yaw_deg" in ekf.columns:
    ekf["yaw_signed_deg"] = wrap_deg_signed(ekf["yaw_deg"])

if not gps.empty:
    has_alt_geoid = {"altitude_m", "geoid_separation_m"}.issubset(gps.columns)
    if has_alt_geoid:
        gps_ellipsoid_height = gps["altitude_m"] + gps["geoid_separation_m"]
        if "ellipsoid_height_m" in gps.columns:
            gps["ellipsoid_height_m"] = gps["ellipsoid_height_m"].combine_first(gps_ellipsoid_height)
        else:
            gps["ellipsoid_height_m"] = gps_ellipsoid_height
        print("GPS ellipsoid_height_m filled from GGA altitude_m + geoid_separation_m where needed")

gps_pos = gps[gps["latitude_deg"].notna() | gps["longitude_deg"].notna()].copy()
gps_hdt = gps[(gps.get("message_type", "") == "HDT") | gps["heading_true_deg"].notna() | gps["heading_deg"].notna()].copy()
gps_track = gps[gps["track_true_deg"].notna()].copy()

if "message_type" in gps.columns:
    gps_gga = gps[gps["message_type"] == "GGA"].copy()
    gps_rmc = gps[gps["message_type"] == "RMC"].copy()
    gps_vtg = gps[gps["message_type"] == "VTG"].copy()
    gps_gst = gps[gps["message_type"] == "GST"].copy()
else:
    gps_gga = pd.DataFrame()
    gps_rmc = pd.DataFrame()
    gps_vtg = pd.DataFrame()
    gps_gst = pd.DataFrame()

if not gps_gga.empty:
    gps_pos_for_plot = gps_gga.copy()
else:
    gps_pos_for_plot = gps_pos.copy()

if not teensy.empty and "heading_valid" in teensy.columns:
    teensy_heading = teensy[(teensy["heading_valid"] == 1) & teensy["heading_deg"].notna()].copy()
else:
    teensy_heading = pd.DataFrame()

if not teensy.empty:
    teensy["speed_2d_mps"] = np.sqrt(teensy["vel_n_mps"] ** 2 + teensy["vel_e_mps"] ** 2)
    if "height_m" in teensy.columns:
        teensy["ellipsoid_height_m"] = teensy["height_m"]
if not ekf.empty:
    ekf["speed_2d_mps"] = np.sqrt(ekf["vel_n_mps"] ** 2 + ekf["vel_e_mps"] ** 2)
if not gps.empty:
    gps["speed_2d_mps"] = np.sqrt(gps["vel_n_mps"] ** 2 + gps["vel_e_mps"] ** 2)
    gps["yaw_deg"] = gps["heading_true_deg"].combine_first(gps["heading_deg"]).combine_first(gps["track_true_deg"])
    gps["yaw_signed_deg"] = wrap_deg_signed(gps["yaw_deg"])
gps_speed = gps[gps["ground_speed_mps"].notna()].copy() if "ground_speed_mps" in gps.columns else pd.DataFrame()
if not imu.empty:
    imu["accel_norm_g"] = np.sqrt(imu["accel_x_g"] ** 2 + imu["accel_y_g"] ** 2 + imu["accel_z_g"] ** 2)
if not teensy.empty and "heading_deg" in teensy.columns:
    teensy["yaw_deg"] = teensy["heading_deg"]
    teensy["yaw_signed_deg"] = teensy["heading_signed_deg"]

all_times = []
for frame in [ekf, gps, imu, teensy]:
    if "plot_time_s" in frame.columns:
        all_times.append(frame["plot_time_s"].replace([np.inf, -np.inf], np.nan).dropna())
t0 = min([s.min() for s in all_times if len(s)]) if all_times else 0.0

for frame in [
    ekf,
    gps,
    imu,
    teensy,
    gps_pos,
    gps_pos_for_plot,
    gps_hdt,
    gps_track,
    gps_gga,
    gps_rmc,
    gps_vtg,
    gps_gst,
    gps_speed,
    teensy_heading,
]:
    if not frame.empty and "plot_time_s" in frame.columns:
        frame["x_time_s"] = frame["plot_time_s"] - t0 if USE_RELATIVE_TIME else frame["plot_time_s"]

add_local_xy([gps_pos_for_plot, teensy, ekf])

x_label = "GPS TOW - first sample (s)" if USE_RELATIVE_TIME else "GPS TOW (s)"

print("\nCSV group:", STAMP)
print("Output directory:", output_dir)
if not gps_hdt.empty:
    print("GPS HDT heading rows:", len(gps_hdt), "finite:", finite_count(gps_hdt, "heading_true_deg"))
if "message_type" in gps.columns:
    print("GPS raw NMEA message counts:")
    print(gps["message_type"].value_counts(dropna=False).sort_index())
if not teensy.empty and "heading_valid" in teensy.columns:
    print("Teensy heading_valid counts:")
    print(teensy["heading_valid"].value_counts(dropna=False).sort_index())
    print("Teensy valid finite heading rows:", finite_count(teensy_heading, "heading_deg"))

print_interpolated_difference(
    "GPS raw GGA -> Teensy position comparison after time/height conversion:",
    gps_pos_for_plot,
    teensy,
    ["latitude_deg", "longitude_deg", "ellipsoid_height_m"],
)
print_position_error_m(
    "GPS raw GGA -> Teensy position error in meters after time/height conversion:",
    gps_pos_for_plot,
    teensy,
)
print_interpolated_difference(
    "GPS raw speed -> Teensy speed comparison after time conversion:",
    gps_speed.rename(columns={"ground_speed_mps": "speed_compare_mps"}),
    teensy.assign(speed_compare_mps=teensy["speed_2d_mps"] if "speed_2d_mps" in teensy.columns else np.nan),
    ["speed_compare_mps"],
)

GPS_COLOR = "blue"
TEENSY_COLOR = "green"
IMU_EKF_COLOR = "red"

fig, ax = plt.subplots(1, 1, figsize=(9, 8))
plot_xy_compare(
    ax,
    [
        {"df": gps_pos_for_plot, "label": "GPS raw GGA", "color": GPS_COLOR},
        {"df": teensy, "label": "Teensy output", "color": TEENSY_COLOR},
        {"df": ekf, "label": "IMU EKF", "color": IMU_EKF_COLOR, "every": DOWNSAMPLE_EKF_FOR_PLOT},
    ],
    "Local ENU position overlay, equal meter scale",
)
fig.suptitle(f"Same-coordinate position overlay - {STAMP}")
fig.tight_layout()
save_fig(fig, f"local_enu_position_overlay_{STAMP}.png")


fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
plot_compare(
    axes[0],
    [
        {"df": gps_pos_for_plot, "col": "north_m", "label": "GPS raw GGA", "color": GPS_COLOR},
        {"df": teensy, "col": "north_m", "label": "Teensy output", "color": TEENSY_COLOR},
        {"df": ekf, "col": "north_m", "label": "IMU EKF", "color": IMU_EKF_COLOR, "every": DOWNSAMPLE_EKF_FOR_PLOT},
    ],
    "North position comparison converted from latitude",
    "north from shared origin (m)",
)
plot_compare(
    axes[1],
    [
        {"df": gps_pos_for_plot, "col": "east_m", "label": "GPS raw GGA", "color": GPS_COLOR},
        {"df": teensy, "col": "east_m", "label": "Teensy output", "color": TEENSY_COLOR},
        {"df": ekf, "col": "east_m", "label": "IMU EKF", "color": IMU_EKF_COLOR, "every": DOWNSAMPLE_EKF_FOR_PLOT},
    ],
    "East position comparison converted from longitude",
    "east from shared origin (m)",
)
plot_compare(
    axes[2],
    [
        {"df": gps_pos_for_plot, "col": "ellipsoid_height_m", "label": "GPS raw GGA ellipsoid", "color": GPS_COLOR},
        {"df": teensy, "col": "ellipsoid_height_m", "label": "Teensy output ellipsoid", "color": TEENSY_COLOR},
        {"df": ekf, "col": "ellipsoid_height_m", "label": "IMU EKF", "color": IMU_EKF_COLOR, "every": DOWNSAMPLE_EKF_FOR_PLOT},
    ],
    "Altitude / height comparison",
    "height (m)",
)

fig.suptitle(f"Position and height comparison - {STAMP}")
fig.tight_layout()
save_fig(fig, f"position_height_{STAMP}.png")


fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
plot_compare(
    axes[0],
    [
        {"df": gps_hdt, "col": "heading_true_signed_deg", "label": "GPS raw yaw/heading", "color": GPS_COLOR},
        {"df": teensy_heading, "col": "heading_signed_deg", "label": "Teensy valid yaw/heading", "color": TEENSY_COLOR},
        {"df": ekf, "col": "yaw_signed_deg", "label": "IMU EKF yaw", "color": IMU_EKF_COLOR, "every": DOWNSAMPLE_EKF_FOR_PLOT},
    ],
    "Yaw / heading comparison, signed to CV7 range",
    "deg (-180 to 180)",
)
plot_compare(
    axes[1],
    [
        {"df": gps, "col": "pitch_deg", "label": "GPS raw pitch", "color": GPS_COLOR},
        {"df": teensy, "col": "pitch_deg", "label": "Teensy pitch", "color": TEENSY_COLOR},
        {"df": ekf, "col": "pitch_deg", "label": "IMU EKF pitch", "color": IMU_EKF_COLOR, "every": DOWNSAMPLE_EKF_FOR_PLOT},
    ],
    "Pitch comparison",
    "deg",
)
plot_compare(
    axes[2],
    [
        {"df": gps, "col": "roll_deg", "label": "GPS raw roll", "color": GPS_COLOR},
        {"df": teensy, "col": "roll_deg", "label": "Teensy roll", "color": TEENSY_COLOR},
        {"df": ekf, "col": "roll_deg", "label": "IMU EKF roll", "color": IMU_EKF_COLOR, "every": DOWNSAMPLE_EKF_FOR_PLOT},
    ],
    "Roll comparison",
    "deg",
)

fig.suptitle(f"Attitude comparison - {STAMP}")
fig.tight_layout()
save_fig(fig, f"attitude_heading_{STAMP}.png")


fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
plot_compare(
    axes[0],
    [
        {"df": gps_speed, "col": "ground_speed_mps", "label": "GPS raw speed", "color": GPS_COLOR},
        {"df": teensy, "col": "speed_2d_mps", "label": "Teensy speed", "color": TEENSY_COLOR},
        {"df": ekf, "col": "speed_2d_mps", "label": "IMU EKF speed", "color": IMU_EKF_COLOR, "every": DOWNSAMPLE_EKF_FOR_PLOT},
    ],
    "Horizontal speed comparison",
    "m/s",
)
plot_compare(
    axes[1],
    [
        {"df": gps, "col": "vel_n_mps", "label": "GPS raw vel N", "color": GPS_COLOR},
        {"df": teensy, "col": "vel_n_mps", "label": "Teensy vel N", "color": TEENSY_COLOR},
        {"df": ekf, "col": "vel_n_mps", "label": "IMU EKF vel N", "color": IMU_EKF_COLOR, "every": DOWNSAMPLE_EKF_FOR_PLOT},
    ],
    "North velocity comparison",
    "m/s",
)
plot_compare(
    axes[2],
    [
        {"df": gps, "col": "vel_e_mps", "label": "GPS raw vel E", "color": GPS_COLOR},
        {"df": teensy, "col": "vel_e_mps", "label": "Teensy vel E", "color": TEENSY_COLOR},
        {"df": ekf, "col": "vel_e_mps", "label": "IMU EKF vel E", "color": IMU_EKF_COLOR, "every": DOWNSAMPLE_EKF_FOR_PLOT},
    ],
    "East velocity comparison",
    "m/s",
)
plot_compare(
    axes[3],
    [
        {"df": gps, "col": "vel_d_mps", "label": "GPS raw vel D", "color": GPS_COLOR},
        {"df": teensy, "col": "vel_d_mps", "label": "Teensy vel D", "color": TEENSY_COLOR},
        {"df": ekf, "col": "vel_d_mps", "label": "IMU EKF vel D", "color": IMU_EKF_COLOR, "every": DOWNSAMPLE_EKF_FOR_PLOT},
    ],
    "Down velocity comparison",
    "m/s",
)

fig.suptitle(f"Velocity comparison - {STAMP}")
fig.tight_layout()
save_fig(fig, f"velocity_{STAMP}.png")


fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
plot_compare(
    axes[0],
    [
        {"df": gps, "col": "accel_x_g", "label": "GPS raw accel X", "color": GPS_COLOR},
        {"df": teensy, "col": "accel_x_g", "label": "Teensy accel X", "color": TEENSY_COLOR},
        {"df": imu, "col": "accel_x_g", "label": "IMU accel X", "color": IMU_EKF_COLOR},
    ],
    "Acceleration X comparison",
    "g",
)
plot_compare(
    axes[1],
    [
        {"df": gps, "col": "accel_y_g", "label": "GPS raw accel Y", "color": GPS_COLOR},
        {"df": teensy, "col": "accel_y_g", "label": "Teensy accel Y", "color": TEENSY_COLOR},
        {"df": imu, "col": "accel_y_g", "label": "IMU accel Y", "color": IMU_EKF_COLOR},
    ],
    "Acceleration Y comparison",
    "g",
)
plot_compare(
    axes[2],
    [
        {"df": gps, "col": "accel_z_g", "label": "GPS raw accel Z", "color": GPS_COLOR},
        {"df": teensy, "col": "accel_z_g", "label": "Teensy accel Z", "color": TEENSY_COLOR},
        {"df": imu, "col": "accel_z_g", "label": "IMU accel Z", "color": IMU_EKF_COLOR},
    ],
    "Acceleration Z comparison",
    "g",
)
plot_compare(
    axes[3],
    [
        {"df": gps, "col": "accel_norm_g", "label": "GPS raw accel norm", "color": GPS_COLOR},
        {"df": teensy, "col": "accel_norm_g", "label": "Teensy accel norm", "color": TEENSY_COLOR},
        {"df": imu, "col": "accel_norm_g", "label": "IMU accel norm", "color": IMU_EKF_COLOR},
    ],
    "Acceleration norm comparison",
    "g",
)

fig.suptitle(f"Acceleration comparison - {STAMP}")
fig.tight_layout()
save_fig(fig, f"acceleration_{STAMP}.png")


fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
plot_compare(
    axes[0],
    [
        {"df": gps, "col": "gyro_x_radps", "label": "GPS raw gyro X", "color": GPS_COLOR},
        {"df": teensy, "col": "gyro_x_radps", "label": "Teensy gyro X", "color": TEENSY_COLOR},
        {"df": imu, "col": "gyro_x_radps", "label": "IMU gyro X", "color": IMU_EKF_COLOR},
    ],
    "Gyro X comparison",
    "rad/s",
)
plot_compare(
    axes[1],
    [
        {"df": gps, "col": "gyro_y_radps", "label": "GPS raw gyro Y", "color": GPS_COLOR},
        {"df": teensy, "col": "gyro_y_radps", "label": "Teensy gyro Y", "color": TEENSY_COLOR},
        {"df": imu, "col": "gyro_y_radps", "label": "IMU gyro Y", "color": IMU_EKF_COLOR},
    ],
    "Gyro Y comparison",
    "rad/s",
)
plot_compare(
    axes[2],
    [
        {"df": gps, "col": "gyro_z_radps", "label": "GPS raw gyro Z", "color": GPS_COLOR},
        {"df": teensy, "col": "gyro_z_radps", "label": "Teensy gyro Z", "color": TEENSY_COLOR},
        {"df": imu, "col": "gyro_z_radps", "label": "IMU gyro Z", "color": IMU_EKF_COLOR},
    ],
    "Gyro Z comparison",
    "rad/s",
)

fig.suptitle(f"Gyro comparison - {STAMP}")
fig.tight_layout()
save_fig(fig, f"gyro_{STAMP}.png")


fig, axes = plt.subplots(1, 1, figsize=(13, 4), sharex=True)
plot_compare(
    axes,
    [
        {"df": gps, "col": "num_sats", "label": "GPS raw num_sats", "color": GPS_COLOR},
        {"df": teensy, "col": "num_sats", "label": "Teensy num_sats", "color": TEENSY_COLOR},
        {"df": ekf, "col": "num_sats", "label": "IMU EKF num_sats", "color": IMU_EKF_COLOR},
    ],
    "Satellite count comparison",
    "count",
)
fig.suptitle(f"Satellite comparison - {STAMP}")
fig.tight_layout()
save_fig(fig, f"satellites_{STAMP}.png")


if SHOW_FIGURES:
    plt.show()
else:
    plt.close("all")
