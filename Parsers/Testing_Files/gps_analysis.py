import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from pyproj import Proj, Transformer

# === Load CSV ===
# Change filename as needed
filename = 'gps_filter_validation4.csv'
df = pd.read_csv(
    filename,
    usecols=["timestamp", "lat", "lon", "alt"],
    engine="python"  # Use Python engine for more flexibility
)

# Drop rows with missing lat/lon
df = df.dropna(subset=['lat', 'lon'])

# === Convert columns to numeric and drop rows with NaN values ===
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
df['alt'] = pd.to_numeric(df['alt'], errors='coerce')
df = df.dropna(subset=['lat', 'lon', 'alt'])

coords = list(zip(df['lat'], df['lon']))
if not coords:
    raise ValueError("No valid GPS data found after cleaning. Check your CSV file.")

# === Calculate Total Distance Traveled ===
total_distance = 0.0
for i in range(1, len(coords)):
    total_distance += geodesic(coords[i-1], coords[i]).meters

print(f"Total distance traveled: {total_distance:.2f} meters")

# === Convert to Local Coordinates (ENU) ===
# Use pyproj to convert lat/lon to UTM (local planar coordinates)
# Reference point: first coordinate
ref_lat, ref_lon = coords[0]
proj_utm = Proj(proj='utm', zone=16, ellps='WGS84', south=False)  # Zone may need adjustment!
transformer = Transformer.from_crs("epsg:4326", proj_utm.srs, always_xy=True)

# Convert all points
eastings, northings = [], []
for lat, lon in coords:
    e, n = transformer.transform(lon, lat)
    eastings.append(e)
    northings.append(n)

# Shift to local ENU (origin at first point)
eastings = np.array(eastings)
northings = np.array(northings)
eastings -= eastings[0]
northings -= northings[0]

# === Plot Trajectory in Local Coordinates ===
plt.figure(figsize=(8,6))
plt.plot(eastings, northings, marker='o', markersize=2, linewidth=1)
plt.xlabel('East (meters)')
plt.ylabel('North (meters)')
plt.title('Trajectory in Local ENU Coordinates')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot Altitude vs Time ===
df['timestamp'] = pd.to_datetime(df['timestamp'])
plt.figure(figsize=(8,4))
plt.plot(df['timestamp'], df['alt'])
plt.xlabel('Time')
plt.ylabel('Altitude (meters)')
plt.title('Altitude vs Time')
plt.grid(True)
plt.tight_layout()
plt.show()