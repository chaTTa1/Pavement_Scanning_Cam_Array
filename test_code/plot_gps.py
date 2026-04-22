# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:45:32 2026

@author: rqy19
"""

import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import numpy as np

# Load data
df = pd.read_csv("position_3.csv")

gps = df[df["source"] == "gps"].copy()
ekf = df[df["source"] == "ekf"].copy()

print(f"GPS points: {len(gps)}")
print(f"EKF points: {len(ekf)}")
print(f"Duration: {df['time'].iloc[-1] - df['time'].iloc[0]:.1f}s")

# Time relative to start
t0 = df["time"].iloc[0]
gps["t"] = gps["time"] - t0
ekf["t"] = ekf["time"] - t0

# Convert to local meters using first GPS point as reference
if not gps.empty:
    ref_lat = gps["lat"].iloc[0]
    ref_lon = gps["lon"].iloc[0]
else:
    ref_lat = ekf["lat"].iloc[0]
    ref_lon = ekf["lon"].iloc[0]

def to_meters(lat, lon):
    e = geodesic((ref_lat, ref_lon), (ref_lat, lon)).meters
    e *= 1.0 if lon >= ref_lon else -1.0
    n = geodesic((ref_lat, ref_lon), (lat, ref_lon)).meters
    n *= 1.0 if lat >= ref_lat else -1.0
    return e, n

if not gps.empty:
    gps_e, gps_n = zip(*[to_meters(r.lat, r.lon) for _, r in gps.iterrows()])
    gps["east_m"] = gps_e
    gps["north_m"] = gps_n

if not ekf.empty:
    ekf_e, ekf_n = zip(*[to_meters(r.lat, r.lon) for _, r in ekf.iterrows()])
    ekf["east_m"] = ekf_e
    ekf["north_m"] = ekf_n

# ── Plot 1: 2D Trajectory ──
fig, ax = plt.subplots(figsize=(10, 10))

if not ekf.empty:
    ax.plot(ekf["east_m"], ekf["north_m"],
            'b-', linewidth=0.5, label="EKF Fused", alpha=0.5)

if not gps.empty:
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

if not gps.empty:
    axes[0].plot(gps["t"], gps["lat"], 'ro', markersize=2, label="GPS", alpha=0.5)
if not ekf.empty:
    axes[0].plot(ekf["t"], ekf["lat"], 'b-', linewidth=0.5, label="EKF", alpha=0.5)
axes[0].set_ylabel("Latitude")
axes[0].set_title("Position Over Time")
axes[0].legend()
axes[0].grid(True)

if not gps.empty:
    axes[1].plot(gps["t"], gps["lon"], 'ro', markersize=2, label="GPS", alpha=0.5)
if not ekf.empty:
    axes[1].plot(ekf["t"], ekf["lon"], 'b-', linewidth=0.5, label="EKF", alpha=0.5)
axes[1].set_ylabel("Longitude")
axes[1].legend()
axes[1].grid(True)

if not gps.empty:
    axes[2].plot(gps["t"], gps["alt"], 'ro', markersize=2, label="GPS", alpha=0.5)
if not ekf.empty:
    axes[2].plot(ekf["t"], ekf["alt"], 'b-', linewidth=0.5, label="EKF", alpha=0.5)
axes[2].set_ylabel("Altitude (m)")
axes[2].set_xlabel("Time (s)")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("position_over_time.png", dpi=150)
plt.show()

# ── Plot 3: East/North over time ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

if not gps.empty:
    axes[0].plot(gps["t"], gps["east_m"], 'ro', markersize=2, label="GPS", alpha=0.5)
if not ekf.empty:
    axes[0].plot(ekf["t"], ekf["east_m"], 'b-', linewidth=0.5, label="EKF", alpha=0.5)
axes[0].set_ylabel("East (m)")
axes[0].set_title("Local Position Over Time")
axes[0].legend()
axes[0].grid(True)

if not gps.empty:
    axes[1].plot(gps["t"], gps["north_m"], 'ro', markersize=2, label="GPS", alpha=0.5)
if not ekf.empty:
    axes[1].plot(ekf["t"], ekf["north_m"], 'b-', linewidth=0.5, label="EKF", alpha=0.5)
axes[1].set_ylabel("North (m)")
axes[1].set_xlabel("Time (s)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("local_position_over_time.png", dpi=150)
plt.show()

# ── Plot 4: GPS vs EKF error ──
if not ekf.empty and not gps.empty and len(ekf) > 2 and len(gps) > 2:
    from scipy.interpolate import interp1d

    ekf_lat_interp = interp1d(ekf["time"], ekf["lat"],
                               bounds_error=False, fill_value="extrapolate")
    ekf_lon_interp = interp1d(ekf["time"], ekf["lon"],
                               bounds_error=False, fill_value="extrapolate")
    ekf_alt_interp = interp1d(ekf["time"], ekf["alt"],
                               bounds_error=False, fill_value="extrapolate")

    gps["ekf_lat"] = ekf_lat_interp(gps["time"])
    gps["ekf_lon"] = ekf_lon_interp(gps["time"])
    gps["ekf_alt"] = ekf_alt_interp(gps["time"])

    gps["error_m"] = [
        geodesic((r.lat, r.lon), (r.ekf_lat, r.ekf_lon)).meters
        for _, r in gps.iterrows()
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(gps["t"], gps["error_m"], 'r-', linewidth=1)
    axes[0].set_ylabel("Horizontal Error (m)")
    axes[0].set_title("EKF vs GPS Difference")
    axes[0].grid(True)

    axes[1].plot(gps["t"], gps["alt"] - gps["ekf_alt"], 'b-', linewidth=1)
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
    alt_diff = abs(gps["alt"] - gps["ekf_alt"])
    print(f"  Altitude:   mean={alt_diff.mean():.3f}m, "
          f"max={alt_diff.max():.3f}m")

# ── Summary ──
print(f"\nData summary:")
print(f"  GPS points:  {len(gps)}")
print(f"  EKF points:  {len(ekf)}")
if not gps.empty:
    print(f"  GPS rate:    {len(gps) / (gps['t'].iloc[-1] - gps['t'].iloc[0]):.1f} Hz"
          if len(gps) > 1 else "")
if not ekf.empty:
    print(f"  EKF rate:    {len(ekf) / (ekf['t'].iloc[-1] - ekf['t'].iloc[0]):.1f} Hz"
          if len(ekf) > 1 else "")