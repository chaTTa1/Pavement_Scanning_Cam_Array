# -*- coding: utf-8 -*-
"""
Plot GPS/IMU Dead Reckoning Results
====================================
Reads position.csv and gps_raw.csv produced by the fusion code.

For Spyder IDE: configure the settings below, then Run (F5).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys


# ─────────────────────────────────────────────
# ★★★ CONFIGURE THESE FOR YOUR DATA ★★★
# ─────────────────────────────────────────────

POSITION_CSV = r"D:\Ryan\GitHub\paper\Pavement_Scanning_Cam_Array\Pavement_Scanning_Cam_Array\test_code\log\circle_rtk\position.csv"     
GPS_RAW_CSV = r"D:\Ryan\GitHub\paper\Pavement_Scanning_Cam_Array\Pavement_Scanning_Cam_Array\test_code\log\circle_rtk\gps_raw.csv"       

SAVE_PLOTS = False                  # True = save PNGs, False = show in Spyder
OUTPUT_DIR = "plots"                # folder for saved PNGs

# Crop (set to None to use all data)
TIME_START = None                   # start time in seconds (None = beginning)
TIME_DURATION = None                # duration in seconds (None = all)

# ─────────────────────────────────────────────
# Make matplotlib work nicely in Spyder
# ─────────────────────────────────────────────
# In Spyder: Tools → Preferences → IPython Console → Graphics
#   → Backend: "Automatic" or "Qt5" for interactive plots
#   → Backend: "Inline" for plots in the console

# Close any leftover figures from previous runs
plt.close("all")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_position_csv(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        print(f"  Current directory: {os.getcwd()}")
        print(f"  Files here: {os.listdir('.')}")
        return None, None, None

    df = pd.read_csv(filepath)

    required = ["time", "source", "lat", "lon"]
    for col in required:
        if col not in df.columns:
            print(f"[ERROR] Missing column '{col}' in {filepath}")
            print(f"  Available columns: {list(df.columns)}")
            return None, None, None

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    for col in ["alt", "ve", "vn", "vu", "yaw", "pitch", "roll",
                "drift_m", "imu_dt", "gps_age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time", "lat", "lon"])

    t0 = df["time"].iloc[0]
    df["t_rel"] = df["time"] - t0
    df["datetime"] = df["time"].apply(
        lambda t: datetime.fromtimestamp(t))

    df_gps = df[df["source"] == "gps"].copy()
    df_dr = df[df["source"] == "dr"].copy()

    print(f"Loaded {filepath}:")
    print(f"  Total rows: {len(df)}")
    print(f"  GPS rows:   {len(df_gps)}")
    print(f"  DR rows:    {len(df_dr)}")
    print(f"  Time span:  {df['t_rel'].iloc[-1]:.1f} seconds")
    print(f"  Columns:    {list(df.columns)}")

    return df, df_gps, df_dr


def load_gps_raw_csv(filepath):
    if not os.path.exists(filepath):
        print(f"[INFO] {filepath} not found, skipping GPS quality plot")
        return None

    df = pd.read_csv(filepath)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    for col in ["lat", "lon", "alt", "fix_quality", "num_sats",
                "hdop", "drift_at_reset_m", "vel_e", "vel_n", "vel_u"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time"])

    if len(df) > 0:
        t0 = df["time"].iloc[0]
        df["t_rel"] = df["time"] - t0
        df["datetime"] = df["time"].apply(
            lambda t: datetime.fromtimestamp(t))

    print(f"Loaded {filepath}: {len(df)} rows")
    return df


def crop_data(df, df_gps, df_dr, df_gps_raw, start, duration):
    if start is None:
        t_start = df["t_rel"].iloc[0]
    else:
        t_start = start

    if duration is None:
        t_end = df["t_rel"].iloc[-1]
    else:
        t_end = t_start + duration

    mask = (df["t_rel"] >= t_start) & (df["t_rel"] <= t_end)
    df = df[mask].copy()
    df_gps = df_gps[(df_gps["t_rel"] >= t_start)
                    & (df_gps["t_rel"] <= t_end)].copy()
    df_dr = df_dr[(df_dr["t_rel"] >= t_start)
                  & (df_dr["t_rel"] <= t_end)].copy()

    if df_gps_raw is not None and len(df_gps_raw) > 0:
        df_gps_raw = df_gps_raw[
            (df_gps_raw["t_rel"] >= t_start)
            & (df_gps_raw["t_rel"] <= t_end)].copy()

    print(f"Cropped to {t_start:.1f}s - {t_end:.1f}s: "
          f"{len(df)} rows ({len(df_gps)} GPS, {len(df_dr)} DR)")

    return df, df_gps, df_dr, df_gps_raw


# ─────────────────────────────────────────────
# PLOT 1: 2D TRAJECTORY MAP
# ─────────────────────────────────────────────

def plot_trajectory(df_gps, df_dr):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.canvas.manager.set_window_title("1 - Trajectory Map")

    # DR points colored by GPS age
    if len(df_dr) > 0 and "gps_age" in df_dr.columns:
        has_age = df_dr["gps_age"].notna()
        if has_age.any():
            sc = ax.scatter(
                df_dr.loc[has_age, "lon"],
                df_dr.loc[has_age, "lat"],
                c=df_dr.loc[has_age, "gps_age"],
                cmap="YlOrRd", s=2, alpha=0.6,
                label="DR predictions", zorder=2)
            plt.colorbar(sc, ax=ax, label="GPS age (seconds)",
                        shrink=0.8, pad=0.02)
        else:
            ax.scatter(df_dr["lon"], df_dr["lat"],
                      c="orange", s=2, alpha=0.6,
                      label="DR predictions", zorder=2)
    elif len(df_dr) > 0:
        ax.scatter(df_dr["lon"], df_dr["lat"],
                  c="orange", s=2, alpha=0.6,
                  label="DR predictions", zorder=2)

    # GPS ground truth
    if len(df_gps) > 0:
        ax.plot(df_gps["lon"], df_gps["lat"],
               "b-", linewidth=1.5, alpha=0.7,
               label="GPS ground truth", zorder=3)
        ax.scatter(df_gps["lon"], df_gps["lat"],
                  c="blue", s=15, alpha=0.9,
                  edgecolors="white", linewidths=0.5, zorder=4)

        # Start / End markers
        ax.scatter(df_gps["lon"].iloc[0], df_gps["lat"].iloc[0],
                  c="green", s=150, marker="^",
                  edgecolors="black", linewidths=1.5,
                  label="Start", zorder=5)
        ax.scatter(df_gps["lon"].iloc[-1], df_gps["lat"].iloc[-1],
                  c="red", s=150, marker="s",
                  edgecolors="black", linewidths=1.5,
                  label="End", zorder=5)

    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.set_title("Trajectory: GPS Ground Truth vs Dead Reckoning",
                fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(useOffset=False, style="plain")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# PLOT 2: LAT/LON VS TIME
# ─────────────────────────────────────────────

def plot_latlon_vs_time(df_gps, df_dr):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.canvas.manager.set_window_title("2 - Lat/Lon vs Time")

    if len(df_dr) > 0:
        ax1.plot(df_dr["t_rel"], df_dr["lat"],
                ".", color="orange", markersize=1, alpha=0.4,
                label="DR")
    if len(df_gps) > 0:
        ax1.plot(df_gps["t_rel"], df_gps["lat"],
                "b.-", markersize=4, linewidth=1, label="GPS")
    ax1.set_ylabel("Latitude (°)", fontsize=11)
    ax1.set_title("Position vs Time", fontsize=14)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(useOffset=False, axis="y", style="plain")

    if len(df_dr) > 0:
        ax2.plot(df_dr["t_rel"], df_dr["lon"],
                ".", color="orange", markersize=1, alpha=0.4,
                label="DR")
    if len(df_gps) > 0:
        ax2.plot(df_gps["t_rel"], df_gps["lon"],
                "b.-", markersize=4, linewidth=1, label="GPS")
    ax2.set_ylabel("Longitude (°)", fontsize=11)
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(useOffset=False, axis="y", style="plain")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# PLOT 3: VELOCITY VS TIME
# ─────────────────────────────────────────────

def plot_velocity(df_gps, df_dr):
    has_vel = all(c in df_dr.columns for c in ["ve", "vn", "vu"])
    if not has_vel or len(df_dr) == 0:
        print("  [SKIP] No velocity columns")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.canvas.manager.set_window_title("3 - Velocity")

    # Components
    if len(df_dr) > 0:
        ax1.plot(df_dr["t_rel"], df_dr["ve"],
                ".", color="coral", markersize=1, alpha=0.4,
                label="DR Ve (east)")
        ax1.plot(df_dr["t_rel"], df_dr["vn"],
                ".", color="orange", markersize=1, alpha=0.4,
                label="DR Vn (north)")
    if len(df_gps) > 0 and "ve" in df_gps.columns:
        ax1.plot(df_gps["t_rel"], df_gps["ve"],
                "r.", markersize=4, label="GPS Ve")
        ax1.plot(df_gps["t_rel"], df_gps["vn"],
                "b.", markersize=4, label="GPS Vn")
    ax1.set_ylabel("Velocity (m/s)", fontsize=11)
    ax1.set_title("Velocity Components", fontsize=14)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2D speed
    if len(df_dr) > 0:
        speed_dr = np.sqrt(df_dr["ve"] ** 2 + df_dr["vn"] ** 2)
        ax2.plot(df_dr["t_rel"], speed_dr,
                ".", color="orange", markersize=1, alpha=0.4,
                label="DR speed")
    if len(df_gps) > 0 and "ve" in df_gps.columns:
        speed_gps = np.sqrt(df_gps["ve"] ** 2 + df_gps["vn"] ** 2)
        ax2.plot(df_gps["t_rel"], speed_gps,
                "b.", markersize=4, label="GPS speed")
    ax2.set_ylabel("Speed (m/s)", fontsize=11)
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_title("2D Speed", fontsize=14)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # km/h secondary axis
    ax2_right = ax2.twinx()
    y_min, y_max = ax2.get_ylim()
    ax2_right.set_ylim(y_min * 3.6, y_max * 3.6)
    ax2_right.set_ylabel("Speed (km/h)", fontsize=11)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# PLOT 4: DRIFT BETWEEN GPS RESETS
# ─────────────────────────────────────────────

def plot_drift(df_gps, df_dr, df_gps_raw):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.canvas.manager.set_window_title("4 - Drift Analysis")

    # Top: continuous drift
    ax1 = axes[0]
    if len(df_dr) > 0 and "drift_m" in df_dr.columns:
        ax1.plot(df_dr["t_rel"], df_dr["drift_m"],
                ".", color="orange", markersize=1, alpha=0.5,
                label="DR drift from anchor")
    ax1.set_ylabel("Drift (meters)", fontsize=11)
    ax1.set_title("IMU Drift Between GPS Resets", fontsize=14)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Bottom: drift at each reset
    ax2 = axes[1]
    if (df_gps_raw is not None and len(df_gps_raw) > 0
            and "drift_at_reset_m" in df_gps_raw.columns):
        drift = df_gps_raw["drift_at_reset_m"].dropna()
        t = df_gps_raw.loc[drift.index, "t_rel"]

        ax2.bar(t, drift, width=0.08, color="coral",
                alpha=0.7, label="Drift at GPS reset")
        ax2.axhline(y=drift.mean(), color="red",
                    linestyle="--", linewidth=1,
                    label=f"Mean: {drift.mean():.4f} m")
        ax2.axhline(y=drift.median(), color="darkred",
                    linestyle=":", linewidth=1,
                    label=f"Median: {drift.median():.4f} m")
        ax2.legend(loc="best", fontsize=9)

        stats_text = (
            f"Count: {len(drift)}\n"
            f"Mean: {drift.mean():.4f} m\n"
            f"Median: {drift.median():.4f} m\n"
            f"Max: {drift.max():.4f} m\n"
            f"Std: {drift.std():.4f} m"
        )
        ax2.text(0.98, 0.95, stats_text,
                transform=ax2.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat",
                         alpha=0.8))
    else:
        ax2.text(0.5, 0.5, "No gps_raw.csv data",
                transform=ax2.transAxes,
                ha="center", va="center", fontsize=14)

    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Drift at reset (meters)", fontsize=11)
    ax2.set_title("Drift at Each GPS Reset", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# PLOT 5: IMU TIMING ANALYSIS
# ─────────────────────────────────────────────

def plot_imu_timing(df_dr):
    if "imu_dt" not in df_dr.columns or len(df_dr) == 0:
        print("  [SKIP] No imu_dt column")
        return None

    dt = df_dr["imu_dt"].dropna()
    if len(dt) == 0:
        print("  [SKIP] No valid imu_dt values")
        return None

    dt_us = dt * 1e6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.canvas.manager.set_window_title("5 - IMU Timing")

    # dt vs time
    t = df_dr.loc[dt.index, "t_rel"]
    ax1.plot(t, dt_us, ".", markersize=1, alpha=0.3, color="steelblue")
    ax1.axhline(y=2500, color="red", linestyle="--",
                linewidth=1, label="Expected 2500 µs")
    ax1.set_xlabel("Time (seconds)", fontsize=11)
    ax1.set_ylabel("IMU dt (µs)", fontsize=11)
    ax1.set_title("IMU Sample Interval vs Time", fontsize=14)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    median_dt = dt_us.median()
    ax1.set_ylim(median_dt - 500, median_dt + 500)

    # Histogram
    ax2.hist(dt_us, bins=100, color="steelblue",
             alpha=0.7, edgecolor="white", linewidth=0.5)
    ax2.axvline(x=2500, color="red", linestyle="--",
                linewidth=1.5, label="Expected 2500 µs")
    ax2.axvline(x=dt_us.mean(), color="green", linestyle="-",
                linewidth=1.5,
                label=f"Mean: {dt_us.mean():.1f} µs")
    ax2.set_xlabel("IMU dt (µs)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("IMU dt Distribution", fontsize=14)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    stats_text = (
        f"Mean: {dt_us.mean():.1f} µs\n"
        f"Std: {dt_us.std():.1f} µs\n"
        f"Min: {dt_us.min():.1f} µs\n"
        f"Max: {dt_us.max():.1f} µs\n"
        f"Jitter: ±{dt_us.std():.1f} µs"
    )
    ax2.text(0.98, 0.95, stats_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow",
                     alpha=0.8))

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# PLOT 6: GPS QUALITY
# ─────────────────────────────────────────────

def plot_gps_quality(df_gps_raw):
    if df_gps_raw is None or len(df_gps_raw) == 0:
        print("  [SKIP] No gps_raw.csv data")
        return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.canvas.manager.set_window_title("6 - GPS Quality")

    t = df_gps_raw["t_rel"]

    # Satellites
    ax1 = axes[0]
    if "num_sats" in df_gps_raw.columns:
        sats = df_gps_raw["num_sats"].dropna()
        ax1.plot(t.loc[sats.index], sats,
                "g.-", markersize=4, linewidth=1)
        ax1.set_ylabel("Satellites", fontsize=11)
        ax1.set_title("GPS Quality Over Time", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(sats.max() + 2, 15))

    # HDOP
    ax2 = axes[1]
    if "hdop" in df_gps_raw.columns:
        hdop = df_gps_raw["hdop"].dropna()
        ax2.plot(t.loc[hdop.index], hdop,
                "m.-", markersize=4, linewidth=1)
        ax2.set_ylabel("HDOP", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color="green", linestyle="--",
                    alpha=0.5, label="Excellent (<1.0)")
        ax2.axhline(y=2.0, color="orange", linestyle="--",
                    alpha=0.5, label="Good (<2.0)")
        ax2.axhline(y=5.0, color="red", linestyle="--",
                    alpha=0.5, label="Poor (>5.0)")
        ax2.legend(loc="best", fontsize=9)

    # Fix quality
    ax3 = axes[2]
    if "fix_quality" in df_gps_raw.columns:
        fq = df_gps_raw["fix_quality"].dropna()
        fix_labels = {
            0: "Invalid", 1: "GPS", 2: "DGPS",
            4: "RTK Fix", 5: "RTK Float",
        }
        colors = {
            0: "red", 1: "orange", 2: "yellow",
            4: "green", 5: "lightgreen",
        }
        for fq_val in sorted(fq.unique()):
            mask = fq == fq_val
            label = fix_labels.get(int(fq_val), f"Type {int(fq_val)}")
            color = colors.get(int(fq_val), "gray")
            ax3.scatter(t.loc[fq.index[mask]], fq[mask],
                       c=color, s=20, label=label, zorder=3)

        ax3.set_ylabel("Fix Quality", fontsize=11)
        ax3.set_xlabel("Time (seconds)", fontsize=11)
        ax3.set_yticks(sorted(fq.unique()))
        ax3.set_yticklabels([
            fix_labels.get(int(v), str(int(v)))
            for v in sorted(fq.unique())
        ])
        ax3.legend(loc="best", fontsize=9)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# PLOT 7: ZOOMED TRAJECTORY (single GPS interval)
# ─────────────────────────────────────────────

def plot_zoomed_interval(df_gps, df_dr):
    if len(df_gps) < 5 or len(df_dr) < 10:
        print("  [SKIP] Not enough data for zoomed plot")
        return None

    # Pick 5 GPS intervals from the middle
    mid = len(df_gps) // 2
    start_idx = max(0, mid - 2)
    end_idx = min(len(df_gps), mid + 3)

    gps_slice = df_gps.iloc[start_idx:end_idx]
    t_start = gps_slice["t_rel"].iloc[0] - 0.05
    t_end = gps_slice["t_rel"].iloc[-1] + 0.05

    dr_slice = df_dr[
        (df_dr["t_rel"] >= t_start)
        & (df_dr["t_rel"] <= t_end)
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.canvas.manager.set_window_title("7 - Zoomed Interval")

    # Zoomed 2D trajectory
    if len(dr_slice) > 0:
        ax1.plot(dr_slice["lon"], dr_slice["lat"],
                "o-", color="orange", markersize=3,
                linewidth=1, alpha=0.7,
                label="DR predictions")
    ax1.plot(gps_slice["lon"], gps_slice["lat"],
            "bs-", markersize=10, linewidth=2,
            label="GPS ground truth")

    # Red arrows: last DR → next GPS (shows reset jump)
    for i in range(len(gps_slice) - 1):
        t_gps = gps_slice["t_rel"].iloc[i + 1]
        dr_before = dr_slice[dr_slice["t_rel"] < t_gps]
        if len(dr_before) > 0:
            last_dr = dr_before.iloc[-1]
            gps_pt = gps_slice.iloc[i + 1]
            ax1.annotate(
                "", xy=(gps_pt["lon"], gps_pt["lat"]),
                xytext=(last_dr["lon"], last_dr["lat"]),
                arrowprops=dict(
                    arrowstyle="->", color="red",
                    linewidth=1.5, alpha=0.7))

    ax1.set_xlabel("Longitude (°)", fontsize=11)
    ax1.set_ylabel("Latitude (°)", fontsize=11)
    ax1.set_title(
        f"Zoomed: {end_idx - start_idx} GPS Intervals "
        f"(red arrows = GPS reset jumps)", fontsize=14)
    ax1.legend(loc="best")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(useOffset=False, style="plain")

    # Drift sawtooth
    if "drift_m" in dr_slice.columns:
        ax2.plot(dr_slice["t_rel"], dr_slice["drift_m"],
                ".-", color="orange", markersize=3, linewidth=1)
        for _, row in gps_slice.iterrows():
            ax2.axvline(x=row["t_rel"], color="blue",
                       linestyle="--", alpha=0.5, linewidth=1)
        ax2.set_ylabel("Drift from anchor (m)", fontsize=11)
        ax2.set_title("Drift Sawtooth Pattern "
                      "(resets to 0 at each blue line)", fontsize=14)
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# ★★★ RUN EVERYTHING ★★★
# ─────────────────────────────────────────────

print("=" * 60)
print("GPS/IMU Dead Reckoning — Plot Results")
print("=" * 60)
print(f"Working directory: {os.getcwd()}")
print()

# ── Load data ──
df, df_gps, df_dr = load_position_csv(POSITION_CSV)

if df is None:
    raise SystemExit("Cannot continue without position.csv")

df_gps_raw = load_gps_raw_csv(GPS_RAW_CSV)

# ── Crop if configured ──
if TIME_START is not None or TIME_DURATION is not None:
    df, df_gps, df_dr, df_gps_raw = crop_data(
        df, df_gps, df_dr, df_gps_raw,
        TIME_START, TIME_DURATION)

# ── Output directory ──
if SAVE_PLOTS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving plots to: {OUTPUT_DIR}/")

# ── Generate all plots ──
print("\nGenerating plots...")
figures = {}

print("  [1/7] Trajectory map")
figures["1_trajectory"] = plot_trajectory(df_gps, df_dr)

print("  [2/7] Lat/Lon vs time")
figures["2_latlon_time"] = plot_latlon_vs_time(df_gps, df_dr)

print("  [3/7] Velocity")
figures["3_velocity"] = plot_velocity(df_gps, df_dr)

print("  [4/7] Drift analysis")
figures["4_drift"] = plot_drift(df_gps, df_dr, df_gps_raw)

print("  [5/7] IMU timing")
figures["5_imu_timing"] = plot_imu_timing(df_dr)

print("  [6/7] GPS quality")
figures["6_gps_quality"] = plot_gps_quality(df_gps_raw)

print("  [7/7] Zoomed interval")
figures["7_zoomed"] = plot_zoomed_interval(df_gps, df_dr)

# ── Save or show ──
if SAVE_PLOTS:
    for name, fig in figures.items():
        if fig is not None:
            path = os.path.join(OUTPUT_DIR, f"{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
    plt.close("all")
    print(f"\nAll plots saved to {OUTPUT_DIR}/")
else:
    # Count valid figures
    valid = sum(1 for f in figures.values() if f is not None)
    print(f"\n{valid} plots generated. "
          f"They should appear in Spyder's Plots pane.")
    plt.show()

print("\nDone!")