# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:44:54 2026

@author: Desktop
"""

import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# Load data
df = pd.read_csv("position.csv")

gps = df[df["source"] == "gps"].copy()
ekf = df[df["source"] == "ekf"].copy()

print(f"GPS points: {len(gps)}")
print(f"EKF points: {len(ekf)}")
print(f"Duration: {df['time'].iloc[-1] - df['time'].iloc[0]:.1f}s")

# Use first GPS point as reference for meter conversion
ref_lat = gps["lat"].iloc[0]
ref_lon = gps["lon"].iloc[0]

def to_meters(lat, lon):
    e = geodesic((ref_lat, ref_lon), (ref_lat, lon)).meters
    e *= 1.0 if lon >= ref_lon else -1.0
    n = geodesic((ref_lat, ref_lon), (lat, ref_lon)).meters
    n *= 1.0 if lat >= ref_lat else -1.0
    return e, n

# Convert GPS to meters
gps_e, gps_n = zip(*[to_meters(r.lat, r.lon) for _, r in gps.iterrows()])
gps["east_m"] = gps_e
gps["north_m"] = gps_n

# Convert EKF to meters
ekf_e, ekf_n = zip(*[to_meters(r.lat, r.lon) for _, r in ekf.iterrows()])
ekf["east_m"] = ekf_e
ekf["north_m"] = ekf_n

# Time relative to start
t0 = df["time"].iloc[0]
gps["t"] = gps["time"] - t0
ekf["t"] = ekf["time"] - t0

# ── Plot 1: 2D Trajectory (East vs North) ──
fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(ekf["east_m"], ekf["north_m"],
        'b-', linewidth=0.5, label="EKF Fused", alpha=0.5)
ax.plot(gps["east_m"], gps["north_m"],
        'ro', markersize=5, label="Raw GPS", alpha=0.7)

ax.plot(gps["east_m"].iloc[0], gps["north_m"].iloc[0],
        'g^', markersize=15, label="Start", zorder=5)
ax.plot(gps["east_m"].iloc[-1], gps["north_m"].iloc[-1],
        'ks', markersize=15, label="End", zorder=5)

ax.set_xlabel("East (m)")
ax.set_ylabel("North (m)")
ax.set_title("GPS vs EKF Trajectory")
ax.legend()
ax.grid(True)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("trajectory_2d.png", dpi=150)
plt.show()

# ── Plot 2: Lat/Lon/Alt over time ──
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(gps["t"], gps["lat"], 'ro', markersize=2, label="GPS", alpha=0.5)
axes[0].plot(ekf["t"], ekf["lat"], 'b-', linewidth=0.5, label="EKF", alpha=0.5)
axes[0].set_ylabel("Latitude")
axes[0].set_title("Position Over Time")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(gps["t"], gps["lon"], 'ro', markersize=2, label="GPS", alpha=0.5)
axes[1].plot(ekf["t"], ekf["lon"], 'b-', linewidth=0.5, label="EKF", alpha=0.5)
axes[1].set_ylabel("Longitude")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(gps["t"], gps["alt"], 'ro', markersize=2, label="GPS", alpha=0.5)
axes[2].plot(ekf["t"], ekf["alt"], 'b-', linewidth=0.5, label="EKF", alpha=0.5)
axes[2].set_ylabel("Altitude (m)")
axes[2].set_xlabel("Time (s)")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("position_over_time.png", dpi=150)
plt.show()

# ── Plot 3: East/North in meters over time ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(gps["t"], gps["east_m"], 'ro', markersize=2, label="GPS", alpha=0.5)
axes[0].plot(ekf["t"], ekf["east_m"], 'b-', linewidth=0.5, label="EKF", alpha=0.5)
axes[0].set_ylabel("East (m)")
axes[0].set_title("Local Position Over Time")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(gps["t"], gps["north_m"], 'ro', markersize=2, label="GPS", alpha=0.5)
axes[1].plot(ekf["t"], ekf["north_m"], 'b-', linewidth=0.5, label="EKF", alpha=0.5)
axes[1].set_ylabel("North (m)")
axes[1].set_xlabel("Time (s)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("local_position_over_time.png", dpi=150)
plt.show()

# ── Plot 4: GPS vs EKF difference ──
# Interpolate EKF to GPS timestamps for comparison
from scipy.interpolate import interp1d

if len(ekf) > 2 and len(gps) > 2:
    ekf_lat_interp = interp1d(ekf["time"], ekf["lat"],
                               bounds_error=False, fill_value="extrapolate")
    ekf_lon_interp = interp1d(ekf["time"], ekf["lon"],
                               bounds_error=False, fill_value="extrapolate")
    ekf_alt_interp = interp1d(ekf["time"], ekf["alt"],
                               bounds_error=False, fill_value="extrapolate")

    gps["ekf_lat_at_gps_time"] = ekf_lat_interp(gps["time"])
    gps["ekf_lon_at_gps_time"] = ekf_lon_interp(gps["time"])
    gps["ekf_alt_at_gps_time"] = ekf_alt_interp(gps["time"])

    # Compute distance between GPS and EKF at each GPS timestamp
    gps["error_m"] = [
        geodesic((r.lat, r.lon),
                 (r.ekf_lat_at_gps_time, r.ekf_lon_at_gps_time)).meters
        for _, r in gps.iterrows()
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(gps["t"], gps["error_m"], 'r-', linewidth=1)
    axes[0].set_ylabel("Horizontal Error (m)")
    axes[0].set_title("EKF vs GPS Difference")
    axes[0].grid(True)

    axes[1].plot(gps["t"], gps["alt"] - gps["ekf_alt_at_gps_time"],
                 'b-', linewidth=1)
    axes[1].set_ylabel("Altitude Difference (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("ekf_vs_gps_error.png", dpi=150)
    plt.show()

    print(f"\nError statistics:")
    print(f"  Horizontal: mean={gps['error_m'].mean():.3f}m, "
          f"max={gps['error_m'].max():.3f}m, "
          f"std={gps['error_m'].std():.3f}m")
    print(f"  Altitude:   mean={abs(gps['alt'] - gps['ekf_alt_at_gps_time']).mean():.3f}m")