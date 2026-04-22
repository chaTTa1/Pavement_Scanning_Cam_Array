# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:16:27 2026

@author: rqy19
"""

import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic


# Load data
df = pd.read_csv(r"C:\Users\rqy19\Documents\github\paper\test_data\sensor_log.csv")

# Get GPS points
gps = df[df["source"] == "gps_gga"].dropna(subset=["gps_lat", "gps_lon"]).copy()

# Get EKF points
ekf = df[df["source"] == "imu_raw"].dropna(subset=["ekf_lat", "ekf_lon"]).copy()

if gps.empty:
    print("No GPS data found")
else:
    # Use first GPS point as reference
    ref_lat = gps["gps_lat"].iloc[0]
    ref_lon = gps["gps_lon"].iloc[0]

    def to_local_meters(lat, lon, ref_lat, ref_lon):
        e = geodesic((ref_lat, ref_lon), (ref_lat, lon)).meters
        e *= 1.0 if lon >= ref_lon else -1.0
        n = geodesic((ref_lat, ref_lon), (lat, ref_lon)).meters
        n *= 1.0 if lat >= ref_lat else -1.0
        return e, n

    # Convert GPS to meters
    gps_e, gps_n = zip(*[to_local_meters(r.gps_lat, r.gps_lon, ref_lat, ref_lon)
                          for _, r in gps.iterrows()])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Raw GPS
    ax.plot(gps_e, gps_n, 'ro', markersize=6, label="Raw GPS", alpha=0.7)

    # EKF fused
    if not ekf.empty:
        ekf_e, ekf_n = zip(*[to_local_meters(r.ekf_lat, r.ekf_lon, ref_lat, ref_lon)
                              for _, r in ekf.iterrows()])
        ax.plot(ekf_e, ekf_n, 'b-', linewidth=1, label="EKF Fused", alpha=0.5)

    # Start/end markers
    ax.plot(gps_e[0], gps_n[0], 'g^', markersize=15, label="Start")
    ax.plot(gps_e[-1], gps_n[-1], 'ks', markersize=15, label="End")

    ax.set_xlabel("East (meters)")
    ax.set_ylabel("North (meters)")
    ax.set_title("GPS vs EKF Position (Local ENU)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig("position_map_meters.png", dpi=150)
    plt.show()
    
#%%

import pandas as pd

df = pd.read_csv(r"D:\Ryan\GitHub\paper\Pavement_Scanning_Cam_Array\Pavement_Scanning_Cam_Array\test_code\sensor_log.csv")
gps = df[df["source"] == "gps_gga"].copy()

print(f"Total GGA sentences: {len(gps)}")
print(f"\nGPS data:")
print(gps[["timestamp", "gps_lat", "gps_lon", "gps_alt", 
            "gps_fix_quality", "gps_num_sats", "gps_hdop", "gps_valid"]].to_string())