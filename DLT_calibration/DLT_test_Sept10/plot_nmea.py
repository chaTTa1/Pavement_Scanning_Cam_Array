# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:41:35 2026

@author: alexc


Plot 3 CSV files (camera_antenna_point, gps_imu_antenna, gps_main) on a shared 
local East-North-Up (ENU) coordinate frame using the RTK aux antenna mean as origin.
"""

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Input CSV File Paths
# -----------------------------------------------------------------------------
csv_targets_path = Path("camera_antenna_point.csv")
csv_aux_path = Path("gps_imu_antenna.csv")
csv_main_path = Path("gps_main.csv")

output_plot_path = Path("csv_positions_enu.png")


# -----------------------------------------------------------------------------
# WGS84 Constants
# -----------------------------------------------------------------------------
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


# -----------------------------------------------------------------------------
# Robust CSV Coordinate Loader
# -----------------------------------------------------------------------------
def load_csv_coordinates(csv_path):
    """Flexibly loads CSV files and extracts (Lat, Lon, Alt) or (East, North, Up).

    Returns:
        coords: N x 3 numpy array
        is_geodetic: True if coordinates are Lat/Lon degrees, False if metric meters
    """
    if not csv_path.exists():
        print(f"[WARNING] File not found: {csv_path}")
        return None, True

    df = pd.read_csv(csv_path)
    col_map = {c.lower().strip().replace(" ", "_"): c for c in df.columns}

    # 1. Look for Geodetic Lat / Lon / Alt columns
    lat_candidates = ["lat", "latitude", "gps_lat", "lat_deg"]
    lon_candidates = ["lon", "longitude", "long", "gps_lon", "lon_deg"]
    alt_candidates = [
        "alt",
        "altitude",
        "elevation",
        "height",
        "msl",
        "h",
        "z",
        "ellipsoid_h",
    ]

    lat_col = next((col_map[c] for c in lat_candidates if c in col_map), None)
    lon_col = next((col_map[c] for c in lon_candidates if c in col_map), None)
    alt_col = next((col_map[c] for c in alt_candidates if c in col_map), None)

    if lat_col and lon_col:
        z_vals = (
            df[alt_col].to_numpy()
            if alt_col
            else np.zeros(len(df), dtype=float)
        )
        coords = np.column_stack(
            [
                df[lat_col].to_numpy(dtype=float),
                df[lon_col].to_numpy(dtype=float),
                z_vals,
            ]
        )
        print(
            f"Loaded {csv_path.name}: {len(coords)} rows (Geodetic Lat/Lon/Alt)"
        )
        return coords, True

    # 2. Look for Cartesian / Metric ENU columns
    x_candidates = ["east", "easting", "x", "target_x", "enu_x"]
    y_candidates = ["north", "northing", "y", "target_y", "enu_y"]
    z_candidates = [
        "up",
        "elevation",
        "z",
        "target_z",
        "height",
        "enu_z",
        "alt",
    ]

    x_col = next((col_map[c] for c in x_candidates if c in col_map), None)
    y_col = next((col_map[c] for c in y_candidates if c in col_map), None)
    z_col = next((col_map[c] for c in z_candidates if c in col_map), None)

    if x_col and y_col:
        z_vals = (
            df[z_col].to_numpy()
            if z_col
            else np.zeros(len(df), dtype=float)
        )
        coords = np.column_stack(
            [
                df[x_col].to_numpy(dtype=float),
                df[y_col].to_numpy(dtype=float),
                z_vals,
            ]
        )
        print(f"Loaded {csv_path.name}: {len(coords)} rows (Metric X/Y/Z)")
        return coords, False

    # 3. Fallback to first numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        z_vals = (
            df[numeric_cols[2]].to_numpy(dtype=float)
            if len(numeric_cols) >= 3
            else np.zeros(len(df), dtype=float)
        )
        coords = np.column_stack(
            [
                df[numeric_cols[0]].to_numpy(dtype=float),
                df[numeric_cols[1]].to_numpy(dtype=float),
                z_vals,
            ]
        )
        is_geo = (
            -90.0 <= np.min(coords[:, 0]) <= 90.0
            and -180.0 <= np.min(coords[:, 1]) <= 180.0
        )
        print(
            f"Loaded {csv_path.name}: {len(coords)} rows (Positional Fallback)"
        )
        return coords, is_geo

    raise ValueError(f"Could not parse coordinate columns in {csv_path.name}")


# -----------------------------------------------------------------------------
# Geodetic <-> ECEF <-> ENU Conversions
# -----------------------------------------------------------------------------
def geodetic_to_ecef(geo):
    lat = np.radians(geo[:, 0])
    lon = np.radians(geo[:, 1])
    h = geo[:, 2]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (N + h) * cos_lat * np.cos(lon)
    y = (N + h) * cos_lat * np.sin(lon)
    z = (N * (1.0 - WGS84_E2) + h) * sin_lat
    return np.column_stack([x, y, z])


def ecef_to_geodetic_single(x, y, z):
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(20):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        h = p / math.cos(lat) - N
        lat_new = math.atan2(z, p * (1.0 - WGS84_E2 * N / (N + h)))
        if abs(lat_new - lat) < 1e-13:
            lat = lat_new
            break
        lat = lat_new
    sin_lat = math.sin(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    h = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), h


def mean_geodetic(geo):
    ecef = geodetic_to_ecef(geo)
    mean = np.mean(ecef, axis=0)
    return ecef_to_geodetic_single(mean[0], mean[1], mean[2])


def geodetic_to_enu(geo, ref_geo):
    ref_ecef = geodetic_to_ecef(
        np.asarray(ref_geo, dtype=float).reshape(1, 3)
    )[0]
    lat = math.radians(ref_geo[0])
    lon = math.radians(ref_geo[1])
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    R = np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ]
    )
    ecef = geodetic_to_ecef(geo)
    return (R @ (ecef - ref_ecef).T).T


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    # Load all 3 CSV datasets
    aux_raw, aux_is_geo = load_csv_coordinates(csv_aux_path)
    main_raw, main_is_geo = load_csv_coordinates(csv_main_path)
    targets_raw, targets_is_geo = load_csv_coordinates(csv_targets_path)

    # Establish ENU Reference Origin using Aux (RTK) mean geodetic location
    if aux_is_geo:
        ref_geo = mean_geodetic(aux_raw)
        print(
            f"\nENU origin set to Aux RTK mean: lat={ref_geo[0]:.9f}, lon={ref_geo[1]:.9f}, h={ref_geo[2]:.4f} m"
        )
    else:
        ref_geo = (0, 0, 0)
        print("\nAux data is already metric; using local origin.")

    # Convert datasets to ENU meters
    aux_enu = geodetic_to_enu(aux_raw, ref_geo) if aux_is_geo else aux_raw
    main_enu = (
        geodetic_to_enu(main_raw, ref_geo)
        if (main_raw is not None and main_is_geo)
        else main_raw
    )

    if targets_raw is not None:
        targets_enu = (
            geodetic_to_enu(targets_raw, ref_geo)
            if targets_is_geo
            else targets_raw
        )
    else:
        targets_enu = None

    # Calculate empirical centroids
    aux_mean = np.mean(aux_enu, axis=0)
    main_mean = np.mean(main_enu, axis=0) if main_enu is not None else None

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(15, 6.5))

    # --- Top-Down View (East vs North) ---
    # 1. Aux Antenna Log
    ax_top.scatter(
        aux_enu[:, 0],
        aux_enu[:, 1],
        s=12,
        c="tab:blue",
        alpha=0.4,
        label=f"Aux RTK Log (n={len(aux_enu)})",
    )
    ax_top.scatter(
        aux_mean[0],
        aux_mean[1],
        s=150,
        marker="X",
        c="blue",
        edgecolor="k",
        linewidth=1.2,
        zorder=5,
        label="Aux Mean",
    )

    # 2. Main Antenna Log
    if main_enu is not None:
        ax_top.scatter(
            main_enu[:, 0],
            main_enu[:, 1],
            s=12,
            c="tab:orange",
            alpha=0.4,
            label=f"Main Log (n={len(main_enu)})",
        )
        ax_top.scatter(
            main_mean[0],
            main_mean[1],
            s=150,
            marker="X",
            c="orange",
            edgecolor="k",
            linewidth=1.2,
            zorder=5,
            label="Main Mean",
        )

    # 3. Camera Ground Targets
    if targets_enu is not None:
        ax_top.scatter(
            targets_enu[:, 0],
            targets_enu[:, 1],
            s=80,
            c="red",
            marker="o",
            edgecolor="k",
            zorder=6,
            label=f"Targets (n={len(targets_enu)})",
        )
        for i, (e, n, u) in enumerate(targets_enu):
            ax_top.annotate(
                f" #{i+1}",
                (e, n),
                fontsize=8,
                color="darkred",
                textcoords="offset points",
                xytext=(3, 3),
            )

    ax_top.set_xlabel("East (m)")
    ax_top.set_ylabel("North (m)")
    ax_top.set_title("Top-Down View (East vs North)\nOrigin = Aux RTK Mean")
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.grid(True, linestyle=":", alpha=0.7)
    ax_top.legend(loc="best", fontsize=9)

    # --- Side Profile View (East vs Up) ---
    ax_side.scatter(
        aux_enu[:, 0], aux_enu[:, 2], s=12, c="tab:blue", alpha=0.4, label="Aux"
    )
    ax_side.scatter(
        aux_mean[0],
        aux_mean[2],
        s=150,
        marker="X",
        c="blue",
        edgecolor="k",
        linewidth=1.2,
        zorder=5,
    )

    if main_enu is not None:
        ax_side.scatter(
            main_enu[:, 0],
            main_enu[:, 2],
            s=12,
            c="tab:orange",
            alpha=0.4,
            label="Main",
        )
        ax_side.scatter(
            main_mean[0],
            main_mean[2],
            s=150,
            marker="X",
            c="orange",
            edgecolor="k",
            linewidth=1.2,
            zorder=5,
        )

    if targets_enu is not None:
        ax_side.scatter(
            targets_enu[:, 0],
            targets_enu[:, 2],
            s=80,
            c="red",
            marker="o",
            edgecolor="k",
            zorder=6,
            label="Targets",
        )
        for i, (e, n, u) in enumerate(targets_enu):
            ax_side.annotate(
                f" #{i+1}",
                (e, u),
                fontsize=8,
                color="darkred",
                textcoords="offset points",
                xytext=(3, 3),
            )

    ax_side.set_xlabel("East (m)")
    ax_side.set_ylabel("Up / Elevation (m)")
    ax_side.set_title("Side Profile View (East vs Up)")
    ax_side.grid(True, linestyle=":", alpha=0.7)
    ax_side.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    print(f"\nPlot saved to: {output_plot_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()