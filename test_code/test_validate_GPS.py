

# -*- coding: utf-8 -*-
"""
Modified on Thu Apr 16 11:05:04 2026
@author: Desktop
"""

from pathlib import Path
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt

# Try a proper projected CRS first
use_pyproj = True
try:
    from pyproj import Transformer
except Exception:
    use_pyproj = False

# List of log files
files = [
    Path(r"D:\Ryan\GitHub\paper\Pavement_Scanning_Cam_Array\Pavement_Scanning_Cam_Array\Data_logs\position_log_04_15_1.csv"),
    Path(r"D:\Ryan\GitHub\paper\Pavement_Scanning_Cam_Array\Pavement_Scanning_Cam_Array\Data_logs\position_log_04_15_2.csv"),
    Path(r"D:\Ryan\GitHub\paper\Pavement_Scanning_Cam_Array\Pavement_Scanning_Cam_Array\Data_logs\position_log_04_15_3.csv"),
]

def load_latlon(file_path: Path) -> pd.DataFrame:
    rows = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()

    normalized = []
    for row in rows[1:]:  # skip header
        row = row + [""] * (8 - len(row))
        normalized.append(row[:8])

    cols = ["timestamp", "lat", "lon", "alt", "heading_deg", "x", "y", "z"]
    df = pd.DataFrame(normalized, columns=cols)
    for c in ["lat", "lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["lat", "lon"]).copy()

# Load all valid points
all_dfs = []
valid_files = []
for fp in files:
    df = load_latlon(fp)
    if not df.empty:
        df["source_file"] = fp.name
        all_dfs.append(df)
        valid_files.append(fp)

if not all_dfs:
    print("No valid data found.")
    exit()

all_points = pd.concat(all_dfs, ignore_index=True)

# Common reference for local metric display
lat0 = all_points["lat"].mean()
lon0 = all_points["lon"].mean()

# Convert to metric coordinates
if use_pyproj:
    # Dayton / Springboro area is in UTM zone 16N (EPSG:32616)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
    e0, n0 = transformer.transform(lon0, lat0)

    def to_meters(lat_series, lon_series):
        east, north = transformer.transform(lon_series.to_numpy(), lat_series.to_numpy())
        return east - e0, north - n0

    method_used = "UTM Zone 16N"
else:
    # Fallback local tangent plane approximation
    R = 6378137.0
    lat0_rad = math.radians(lat0)

    def to_meters(lat_series, lon_series):
        x_m = (lon_series - lon0) * math.pi / 180.0 * R * math.cos(lat0_rad)
        y_m = (lat_series - lat0) * math.pi / 180.0 * R
        return x_m.to_numpy(), y_m.to_numpy()

    method_used = "Local equirectangular approximation"

# 1. Individual plots for each file
for df, fp in zip(all_dfs, valid_files):
    # 1. Remove rows where both Lat and Lon are identical to the previous row
    # keep='first' ensures we keep the first instance of a coordinate and drop subsequent repeats
    df_clean = df.drop_duplicates(subset=['lat', 'lon'], keep='first').copy()
    
    # 2. Calculate coordinates using the cleaned data
    x_m, y_m = to_meters(df_clean["lat"], df_clean["lon"])
    
    # Print the count to see the difference
    print(f"File: {fp.name}")
    print(f"  Original points: {len(df)}")
    print(f"  Unique GPS points: {len(df_clean)}")
    
    # 3. Plot the cleaned data
    plt.figure(figsize=(7, 7))
    plt.plot(x_m, y_m, marker="o", markersize=2, linewidth=1)
    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    plt.title(f"{fp.stem} - Cleaned Track")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

# 2. Combined overlay plot for comparison
plt.figure(figsize=(10, 8))
for df, fp in zip(all_dfs, valid_files):
    x_m, y_m = to_meters(df["lat"], df["lon"])
    plt.plot(x_m, y_m, marker="o", markersize=2, linewidth=1, label=fp.stem)

plt.xlabel("x (meters)")
plt.ylabel("y (meters)")
plt.title(f"Combined Track Overlay ({method_used})")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()

# Summary Output to Console
print(f"--- Summary Statistics ---")
print(f"Conversion method used: {method_used}")
print(f"Common origin: {lat0:.8f}, {lon0:.8f}\n")

for df, fp in zip(all_dfs, valid_files):
    x_m, y_m = to_meters(df["lat"], df["lon"])
    print(f"File: {fp.name}")
    print(f"  Points: {len(df)}")
    print(f"  X Range: {x_m.min():.2f} to {x_m.max():.2f} m")
    print(f"  Y Range: {y_m.min():.2f} to {y_m.max():.2f} m\n")
