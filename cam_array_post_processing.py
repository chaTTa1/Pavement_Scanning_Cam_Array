import os
import pandas as pd
from PIL import Image
import piexif
import numpy as np
from scipy.spatial.transform import Rotation as R
from geopy.distance import geodesic
from geopy.point import Point
from exif import Image as ExifImage
import glob

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

def propagate_gps_with_imu(imu_csv, gps_csv, output_csv=None):
    gps_df = pd.read_csv(gps_csv)
    imu_df = pd.read_csv(imu_csv)

    # Use gps_time as the time reference for IMU, and TOW for GPS
    gps_df.sort_values('TOW', inplace=True)
    imu_df.sort_values('gps_time', inplace=True)

    result_rows = []

    for i in range(len(gps_df) - 1):
        t_start = gps_df.iloc[i]['TOW']
        t_end = gps_df.iloc[i + 1]['TOW']
        heading = gps_df.iloc[i]['Heading'] if 'Heading' in gps_df.columns else None

        segment = imu_df[(imu_df['gps_time'] >= t_start) & (imu_df['gps_time'] < t_end)]
        if segment.empty:
            continue

        # Use new IMU CSV column names
        accel_data = segment[['accel_x', 'accel_y', 'accel_z']].to_numpy()
        quat_data = segment[['q2', 'q3', 'q4', 'q1']].to_numpy()  # scipy expects [x, y, z, w]
        timestamps = segment['gps_time'].to_numpy()
        dt = np.diff(timestamps, prepend=timestamps[0])  # seconds

        velocity = np.zeros((len(accel_data), 3))
        position = np.zeros((len(accel_data), 3))
        gravity = np.array([0, 0, 9.81])

        for j in range(1, len(accel_data)):
            r = R.from_quat(quat_data[j])
            accel_world = r.apply(accel_data[j])
            linear_accel = accel_world - gravity

            prev_r = R.from_quat(quat_data[j - 1])
            prev_world = prev_r.apply(accel_data[j - 1]) - gravity

            velocity[j] = velocity[j - 1] + 0.5 * (linear_accel + prev_world) * dt[j]
            position[j] = position[j - 1] + 0.5 * (velocity[j] + velocity[j - 1]) * dt[j]

        start_lat = gps_df.iloc[i]['Latitude']
        start_lon = gps_df.iloc[i]['Longitude']
        start_alt = gps_df.iloc[i]['Altitude']

        for j, pos in enumerate(position):
            x, y, z = pos
            dest = geodesic(meters=np.sqrt(x**2 + y**2)).destination(Point(start_lat, start_lon), np.degrees(np.arctan2(y, x)))
            result_rows.append({
                'gps_time': timestamps[j],
                'latitude': dest.latitude,
                'longitude': dest.longitude,
                'altitude': start_alt + z,
                'heading': heading
            })

    output_df = pd.DataFrame(result_rows)
    if output_csv is None:
        output_csv = os.path.join(SCRIPTS_DIR, "propagated_gps_imu_output.csv")
    output_df.to_csv(output_csv, index=False)
    return output_df

def imu_aided_gps_interpolation(imu_csv, gps_csv, output_csv=None):
    gps_df = pd.read_csv(gps_csv)
    gps_df['timestamp'] = gps_df['TOW'] + gps_df['WNc'] * 604800

    imu_df = pd.read_csv(imu_csv)
    imu_df['timestamp'] = imu_df['timestamp'] / 1e6  # microseconds to seconds

    result_rows = []
    gravity = np.array([0, 0, 9.81])

    gps_index = 0
    last_gps_pos = np.array([gps_df.iloc[0]['Latitude'], gps_df.iloc[0]['Longitude'], gps_df.iloc[0]['Altitude']])
    last_velocity = np.zeros(3)
    last_timestamp = gps_df.iloc[0]['timestamp']
    last_heading = gps_df.iloc[0]['Heading'] if 'Heading' in gps_df.columns else None

    for i, imu_row in imu_df.iterrows():
        imu_time = imu_row['timestamp']

        while gps_index + 1 < len(gps_df) and imu_time > gps_df.iloc[gps_index + 1]['timestamp']:
            gps_index += 1
            last_gps_pos = np.array([
                gps_df.iloc[gps_index]['Latitude'],
                gps_df.iloc[gps_index]['Longitude'],
                gps_df.iloc[gps_index]['Altitude']
            ])
            last_velocity = np.zeros(3)
            last_timestamp = gps_df.iloc[gps_index]['timestamp']
            last_heading = gps_df.iloc[gps_index]['Heading'] if 'Heading' in gps_df.columns else None

        dt = imu_time - last_timestamp
        if dt <= 0:
            continue

        # Use new IMU CSV quaternion and accel columns
        quat = [imu_row['q2'], imu_row['q3'], imu_row['q4'], imu_row['q1']]
        accel = np.array([imu_row['accel_x'], imu_row['accel_y'], imu_row['accel_z']])
        r = R.from_quat(quat)
        accel_world = r.apply(accel) - gravity

        velocity = last_velocity + accel_world * dt
        displacement = 0.5 * (last_velocity + velocity) * dt

        start_point = Point(last_gps_pos[0], last_gps_pos[1])
        flat_distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
        bearing = np.degrees(np.arctan2(displacement[1], displacement[0])) % 360
        new_point = geodesic(meters=flat_distance).destination(start_point, bearing)
        new_alt = last_gps_pos[2] + displacement[2]

        result_rows.append({
            'timestamp': imu_time,
            'latitude': new_point.latitude,
            'longitude': new_point.longitude,
            'altitude': new_alt,
            'heading': last_heading
        })

        last_velocity = velocity
        last_timestamp = imu_time

    output_df = pd.DataFrame(result_rows)
    if output_csv is None:
        output_csv = os.path.join(SCRIPTS_DIR, "imu_aided_gps_interpolation_output.csv")
    output_df.to_csv(output_csv, index=False)
    return output_df

def parse_mosaic_txt_data(file_path, output_csv=None):
    data = []
    with open(file_path, 'r', encoding='latin-1') as file:
        for line in file:
            if line.startswith("SBF PVTGeodetic2"):
                parts = line.strip().split(',')
                if len(parts) >= 10:
                    try:
                        tow = float(parts[1])
                        wnc = int(parts[2])
                        lat = float(parts[7])
                        lon = float(parts[8])
                        alt = float(parts[9])
                        hdt = float(parts[14])
                        # Append the parsed data to the list
                        data.append([tow, wnc, lat, lon, alt, hdt])
                    except ValueError:
                        continue
    df = pd.DataFrame(data, columns=["TOW", "WNc", "Latitude", "Longitude", "Altitude", "Heading"])
    if output_csv is None:
        output_csv = os.path.join(SCRIPTS_DIR, "mosaic_parsed_output06_24_test2.csv")
    df.to_csv(output_csv, index=False)
    return df

def debug_imu_gps_time_ranges(imu_csv, gps_csv):
    imu_df = pd.read_csv(imu_csv)
    gps_df = pd.read_csv(gps_csv)

    print("IMU time range:", imu_df['timestamp'].min(), imu_df['timestamp'].max())
    print("GPS time range:", gps_df['timestamp'].min(), gps_df['timestamp'].max())
    print("IMU columns:", imu_df.columns)
    print("GPS columns:", gps_df.columns)\
    

def add_EXIF_to_images(image_directory, cam_prop_gps_csv):
    """
    Adds EXIF data to all JPG images in the given directory, using different EXIF blocks
    depending on whether the directory name contains 'left', 'mid', or 'right'.
    Matches image timestamps to closest TOW in cam_prop_gps_csv and writes GPS EXIF tags.
    """
    gps_df = pd.read_csv(cam_prop_gps_csv)
    # Determine camera type from directory name
    dir_lower = os.path.basename(image_directory).lower()
    if "left" in dir_lower:
        cam_lat_col, cam_lon_col = 'leftlat', 'leftlon'
    elif "mid" in dir_lower or "middle" in dir_lower:
        cam_lat_col, cam_lon_col = 'midlat', 'midlon'
    elif "right" in dir_lower:
        cam_lat_col, cam_lon_col = 'rightlat', 'rightlon'
    else:
        print("Directory name does not contain 'left', 'mid', or 'right'. No EXIF added.")
        return

    # Load GPS CSV
    gps_df = pd.read_csv(cam_prop_gps_csv)
    gps_df['timestamp'] = gps_df['TOW'] + gps_df['Wnc'] * 604800  # GPS time in seconds

    # Process all JPG images in the directory
    image_files = glob.glob(os.path.join(image_directory, '*.jpg'))
    for file_path in image_files:
        # Extract timestamp from filename (assuming format: ..._<timestamp>.jpg)
        base = os.path.splitext(os.path.basename(file_path))[0]
        # Try to extract timestamp (customize this if your naming is different)
        try:
            img_timestamp = float(base.split('_')[-1]) / 1e6  # microseconds to seconds
        except Exception:
            print(f"Could not extract timestamp from {file_path}, skipping.")
            continue

        # Find closest GPS row
        idx = (gps_df['timestamp'] - img_timestamp).abs().idxmin()
        gps_row = gps_df.iloc[idx]

        # Prepare EXIF GPS tags
        lat = gps_row[cam_lat_col]
        lon = gps_row[cam_lon_col]
        alt = gps_row['altitude']
 
        lat_ref = "N" if lat >= 0 else "S"
        lon_ref = "E" if lon >= 0 else "W"

        # Write EXIF
        with open(file_path, 'rb') as image_file:
            img = ExifImage(image_file)
            img.make = "Flir"
            img.model = "Blackfly S"
            img.focal_length = "6"
            img.gps_latitude = abs(lat)
            img.gps_latitude_ref = lat_ref
            img.gps_longitude = abs(lon)
            img.gps_longitude_ref = lon_ref
            img.gps_altitude = float(alt)
            img.gps_altitude_ref = 0  # 0 = above sea level

        # Save with _EXIF appended to filename
        new_path = os.path.join(image_directory, f"{base}_EXIF.jpg")
        with open(new_path, 'wb') as new_image_file:
            new_image_file.write(img.get_file())

def gps_converted(input_gps_csv, output_csv=None, camisleft_flag=False, l_dist=0.6, r_dist=0.4, m_dist=0.1):
    """
    Converts GPS/heading data to per-camera GPS coordinates for left, mid, right cameras.
    Args:
        input_gps_csv (str): Path to input GPS CSV with columns [TOW, WNc, Latitude, Longitude, Altitude, Heading]
        output_csv (str or None): Path to output CSV. If None, saves as 'gps_converted_output.csv' in script directory.
        camisleft_flag (bool): If True, mid camera is to left of GPS; else to right
        l_dist, r_dist, m_dist (float): Distances from GPS to left, right, and mid cameras (meters)
    """

    SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

    gps_values = pd.read_csv(input_gps_csv)
    new_gps_values = []

    def deg2rad(deg):
        return np.radians(deg)

    def lat_offset(dist, heading):
        return (dist * np.sin(deg2rad(heading))) / 111139

    def lon_offset(dist, heading, lat):
        return (dist * np.cos(deg2rad(heading))) / (111139 * np.cos(np.radians(lat)))

    for i, row in gps_values.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        alt = row['Altitude']
        head = row['Heading']
        TOW = row['TOW']
        WNc = row['WNc']

        # Left and right camera positions
        if 0 < head < 180 and head != 90:
            left_lat = lat + lat_offset(l_dist, head)
            right_lat = lat - lat_offset(r_dist, head)
        elif 180 < head < 360 and head != 270:
            left_lat = lat - lat_offset(l_dist, head - 180)
            right_lat = lat + lat_offset(r_dist, head - 180)
        elif head == 0 or head == 360:
            left_lat = lat
            right_lat = lat
        elif head == 180:
            left_lat = lat
            right_lat = lat
        elif head == 90:
            left_lat = lat + (l_dist / 111139)
            right_lat = lat - (r_dist / 111139)
        elif head == 270:
            left_lat = lat - (l_dist / 111139)
            right_lat = lat + (r_dist / 111139)
        else:
            left_lat = lat
            right_lat = lat

        # Longitude for left/right
        if (head > 270 or head < 90) and head not in [0, 360]:
            left_lon = lon - lon_offset(l_dist, head, left_lat)
            right_lon = lon + lon_offset(r_dist, head, right_lat)
        elif 90 < head < 270 and head != 180:
            left_lon = lon + lon_offset(l_dist, head, left_lat)
            right_lon = lon - lon_offset(r_dist, head, right_lat)
        elif head == 0 or head == 360:
            left_lon = lon - (l_dist / (111139 * np.cos(np.radians(left_lat))))
            right_lon = lon + (r_dist / (111139 * np.cos(np.radians(right_lat))))
        elif head == 180:
            left_lon = lon + (l_dist / (111139 * np.cos(np.radians(left_lat))))
            right_lon = lon - (r_dist / (111139 * np.cos(np.radians(right_lat))))
        elif head == 90:
            left_lon = lon
            right_lon = lon
        elif head == 270:
            left_lon = lon
            right_lon = lon
        else:
            left_lon = lon
            right_lon = lon

        # Mid camera position
        if camisleft_flag:
            if 0 < head < 180 and head != 90:
                mid_lat = lat + lat_offset(l_dist, head)
            elif 180 < head < 360 and head != 270:
                mid_lat = lat - lat_offset(l_dist, head - 180)
            elif head == 0 or head == 360:
                mid_lat = lat
            elif head == 180:
                mid_lat = lat
            elif head == 90:
                mid_lat = lat + (l_dist / 111139)
            elif head == 270:
                mid_lat = lat - (l_dist / 111139)
            else:
                mid_lat = lat

            if (head > 270 or head < 90) and head not in [0, 360]:
                mid_lon = lon - lon_offset(l_dist, head, mid_lat)
            elif 90 < head < 270 and head != 180:
                mid_lon = lon + lon_offset(l_dist, head, mid_lat)
            elif head == 0 or head == 360:
                mid_lon = lon - (l_dist / (111139 * np.cos(np.radians(mid_lat))))
            elif head == 180:
                mid_lon = lon + (l_dist / (111139 * np.cos(np.radians(mid_lat))))
            elif head == 90 or head == 270:
                mid_lon = lon
            else:
                mid_lon = lon
        else:
            if 0 < head < 180 and head != 90:
                mid_lat = lat - lat_offset(r_dist, head)
            elif 180 < head < 360 and head != 270:
                mid_lat = lat + lat_offset(r_dist, head - 180)
            elif head == 0 or head == 360:
                mid_lat = lat
            elif head == 180:
                mid_lat = lat
            elif head == 90:
                mid_lat = lat - (r_dist / 111139)
            elif head == 270:
                mid_lat = lat + (r_dist / 111139)
            else:
                mid_lat = lat

            if (head > 270 or head < 90) and head not in [0, 360]:
                mid_lon = lon + lon_offset(r_dist, head, mid_lat)
            elif 90 < head < 270 and head != 180:
                mid_lon = lon - lon_offset(r_dist, head, mid_lat)
            elif head == 0 or head == 360:
                mid_lon = lon + (r_dist / (111139 * np.cos(np.radians(mid_lat))))
            elif head == 180:
                mid_lon = lon - (r_dist / (111139 * np.cos(np.radians(mid_lat))))
            elif head == 90 or head == 270:
                mid_lon = lon
            else:
                mid_lon = lon

        new_gps_values.append([
            left_lat, left_lon, mid_lat, mid_lon, right_lat, right_lon, alt, head, TOW, WNc
        ])

    columns = [
        'leftlat', 'leftlon', 'midlat', 'midlon', 'rightlat', 'rightlon',
        'altitude', 'heading', 'TOW', 'WNc'
    ]

    if output_csv is None:
        output_csv = os.path.join(SCRIPTS_DIR, "gps_converted_output.csv")

    pd.DataFrame(new_gps_values, columns=columns).to_csv(output_csv, index=False)

