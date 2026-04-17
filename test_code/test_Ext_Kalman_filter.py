# -*- coding: utf-8 -*-
"""
EKF Sensor Fusion: GPS + IMU with Camera Position Broadcasting
Corrected version — April 17, 2026
"""

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from scipy.spatial.transform import Rotation as R
import serial
import pynmea2
import struct
import time
from geopy.distance import geodesic
from geopy.point import Point
import pandas as pd
import os
import socket
import json
import subprocess

# ============================================================
# CONFIGURATION
# ============================================================
IMU_PORT = 'COM3'       # Jetson Nano: /dev/ttyUSB0 or /dev/ttyACM0
IMU_BAUD = 115200
GPS_PORT = 'COM10'      # Jetson Nano: /dev/ttyUSB1
GPS_BAUD = 115200
GRAVITY_MAG = 9.80665   # m/s²

# IMU timestamp divisor: set to 1e6 for microseconds, 1e3 for milliseconds
IMU_TIMESTAMP_DIVISOR = 1e6

# Set True if your IMU outputs acceleration in g-units (not m/s²)
IMU_ACCEL_IN_G = True

# Setup UDP broadcast socket
UDP_IP = "192.168.1.255"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# Camera hardware configuration (lateral offsets from IMU center)
# Positive = right of heading, Negative = left of heading
CAMERA_FORWARD_OFFSET = 0.01   # All cameras same forward distance (m)
CAMERA_LATERAL_OFFSETS = {
    "left":  -0.6,   # 0.6m to the left
    "mid":    0.0,   # centered
    "right":  0.4,   # 0.4m to the right
}

# Map roles to (username, IP) for each Jetson Nano
jetsons = {
    "left":  ("ryan4", "192.168.1.12"),
    "mid":   ("ryan5", "192.168.1.11"),
    "right": ("ryan6", "192.168.1.10"),
}


# ============================================================
# FAST LOGGER (Batched CSV writes to reduce SSD I/O)
# ============================================================
class FastLogger:
    def __init__(self, filename='position_log.csv', batch_size=400):
        self.filename = filename
        self.batch_size = batch_size
        self.buffer = []
        self.header_written = False

    def log(self, timestamp, gps_pos, gps_error=None):
        data = {
            'timestamp': timestamp,
            'lat': gps_pos[0],
            'lon': gps_pos[1],
            'alt': gps_pos[2],
            'lat_err': gps_error[0] if gps_error else None,
            'lon_err': gps_error[1] if gps_error else None,
            'alt_err': gps_error[2] if gps_error else None
        }
        self.buffer.append(data)

        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        df.to_csv(
            self.filename,
            mode='a',
            header=not self.header_written,
            index=False
        )
        self.header_written = True
        self.buffer = []


# ============================================================
# SSH SCRIPT MANAGEMENT
# ============================================================
ssh_processes = {}


def launch_all_jetson_scripts():
    """SSH into each Jetson Nano and start the camera capture script."""
    print("[INFO] Launching blkfly.py scripts on all Jetsons...")
    for role, (username, ip) in jetsons.items():
        try:
            proc = subprocess.Popen(
                ["ssh", f"{username}@{ip}",
                 f"DEVICE_ROLE={role} python3 ~/Alex/Blackfly/blkfly.py"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            ssh_processes[role] = proc
            print(f"  [OK] Launched {role} on {username}@{ip}")
        except Exception as exc:
            print(f"  [ERROR] Could not launch {role} on {ip}: {exc}")


def stop_all_jetson_scripts():
    """SSH into each Jetson Nano and kill the camera capture script."""
    print("[INFO] Stopping blkfly.py scripts on Jetsons...")
    for role, (username, ip) in jetsons.items():
        try:
            subprocess.run(
                ["ssh", f"{username}@{ip}", "pkill -f blkfly.py"],
                check=True, capture_output=True, text=True
            )
            print(f"  [OK] Stopped {role} on {ip}")
        except Exception as exc:
            print(f"  [ERROR] Could not stop {role} on {ip}: {exc}")


# ============================================================
# GPS ↔ LOCAL COORDINATE CONVERSION (WGS84 Ellipsoid)
# ============================================================
def gps_to_local_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """Convert geodetic (lat, lon, alt) to local East-North-Up (meters)
    relative to a reference point. Uses WGS84 ellipsoid radii."""
    d_lat = np.radians(lat - ref_lat)
    d_lon = np.radians(lon - ref_lon)
    ref_lat_rad = np.radians(ref_lat)

    e2 = 0.00669437999014  # WGS84 first eccentricity squared
    a = 6378137.0          # WGS84 semi-major axis

    sin_ref = np.sin(ref_lat_rad)
    N = a / np.sqrt(1 - e2 * sin_ref ** 2)                    # Prime vertical radius
    M = a * (1 - e2) / (1 - e2 * sin_ref ** 2) ** 1.5        # Meridional radius

    east  = d_lon * (N + alt) * np.cos(ref_lat_rad)
    north = d_lat * (M + alt)
    up    = alt - ref_alt

    return np.array([east, north, up])


def local_enu_to_gps(e, n, u, ref_lat, ref_lon, ref_alt):
    """Convert local East-North-Up (meters) back to geodetic (lat, lon, alt)."""
    ref_lat_rad = np.radians(ref_lat)

    e2 = 0.00669437999014
    a = 6378137.0

    sin_ref = np.sin(ref_lat_rad)
    N = a / np.sqrt(1 - e2 * sin_ref ** 2)
    M = a * (1 - e2) / (1 - e2 * sin_ref ** 2) ** 1.5

    alt = ref_alt + u
    d_lat = n / (M + alt)
    d_lon = e / ((N + alt) * np.cos(ref_lat_rad))

    lat = ref_lat + np.degrees(d_lat)
    lon = ref_lon + np.degrees(d_lon)

    return lat, lon, alt


# ============================================================
# EKF STATE TRANSITION & JACOBIAN
# ============================================================
# State vector (16 elements):
#   x[0:3]   = position (East, North, Up) in meters
#   x[3:6]   = velocity (m/s)
#   x[6:10]  = quaternion [qx, qy, qz, qw]  (scipy convention)
#   x[10:13] = gyroscope bias (rad/s)
#   x[13:16] = accelerometer bias (m/s²)

def normalize_quat_in_state(x):
    """Normalize the quaternion portion of the state vector in-place."""
    q = x[6:10]
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        x[6:10] = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        x[6:10] = q / norm
    return x


def fx(x, dt, u):
    """State transition function.
    
    Args:
        x:  State vector (16,)
        dt: Time step (seconds)
        u:  Control input [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
    
    Returns:
        x_new: Predicted state vector (16,)
    """
    pos  = x[0:3].copy()
    vel  = x[3:6].copy()
    quat = x[6:10].copy()
    b_g  = x[10:13].copy()
    b_a  = x[13:16].copy()

    # Correct sensor readings with estimated biases
    gyro = u[0:3] - b_g
    acc  = u[3:6] - b_a

    # Normalize quaternion before use
    quat = quat / np.linalg.norm(quat)

    # Rotate body-frame acceleration to world frame
    r = R.from_quat(quat)  # scipy: [x, y, z, w]
    acc_world = r.apply(acc)
    acc_world[2] -= GRAVITY_MAG  # Remove gravity (world Z = Up)

    # Constant acceleration kinematic equations
    new_pos = pos + vel * dt + 0.5 * acc_world * (dt ** 2)
    new_vel = vel + acc_world * dt

    # Propagate orientation
    delta_r = R.from_rotvec(gyro * dt)
    new_r = r * delta_r
    new_quat = new_r.as_quat()  # [x, y, z, w]

    # Biases modeled as random walk (constant between updates)
    x_new = np.zeros(16)
    x_new[0:3]   = new_pos
    x_new[3:6]   = new_vel
    x_new[6:10]  = new_quat
    x_new[10:13] = b_g
    x_new[13:16] = b_a

    return x_new


def compute_F_jacobian(x, dt, u):
    """Numerical Jacobian of fx() with respect to state x.
    
    Uses central differences for better accuracy than forward differences.
    """
    n = len(x)
    F = np.zeros((n, n))
    eps = 1e-7

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps

        f_plus  = fx(x_plus, dt, u)
        f_minus = fx(x_minus, dt, u)

        F[:, i] = (f_plus - f_minus) / (2.0 * eps)

    return F


# ============================================================
# EKF MEASUREMENT MODEL
# ============================================================
def hx(x):
    """Measurement function: GPS observes position directly."""
    return x[0:3]


def H_jacobian(x):
    """Jacobian of hx(): GPS measures [East, North, Up]."""
    H = np.zeros((3, 16))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    return H


# ============================================================
# SENSOR FUSION EKF CLASS
# ============================================================
class SensorFusionEKF:
    def __init__(self):
        self.ekf = ExtendedKalmanFilter(dim_x=16, dim_z=3)

        # --- Initial state ---
        self.ekf.x = np.zeros(16)
        self.ekf.x[9] = 1.0  # Identity quaternion w-component [x,y,z,w]

        # --- Initial covariance ---
        self.ekf.P = np.eye(16) * 100.0
        self.ekf.P[6:10, 6:10] = np.eye(4) * 1.0    # Quaternion: smaller initial uncertainty
        self.ekf.P[10:13, 10:13] = np.eye(3) * 0.01  # Gyro bias
        self.ekf.P[13:16, 13:16] = np.eye(3) * 0.1   # Accel bias

        # --- Process noise ---
        self.ekf.Q = np.eye(16) * 0.001
        self.ekf.Q[0:3, 0:3]   = np.eye(3) * 0.01    # Position process noise
        self.ekf.Q[3:6, 3:6]   = np.eye(3) * 0.1     # Velocity process noise
        self.ekf.Q[6:10, 6:10] = np.eye(4) * 0.001   # Quaternion process noise
        self.ekf.Q[10:13, 10:13] = np.eye(3) * 1e-5   # Gyro bias drift
        self.ekf.Q[13:16, 13:16] = np.eye(3) * 1e-4   # Accel bias drift

        # --- Measurement noise (default, overridden by GST) ---
        self.ekf.R = np.eye(3) * 1.0

        # --- State tracking ---
        self.reference_gps = None
        self.last_imu_time = None
        self.last_valid_heading = 0.0

    def set_reference_gps(self, lat, lon, alt):
        """Set the GPS reference point (origin of local ENU frame)."""
        self.reference_gps = (lat, lon, alt if alt is not None else 0.0)

    def update_imu(self, gyro, accel, timestamp):
        """Process one IMU sample: propagate state and covariance.
        
        Args:
            gyro:  np.array [gx, gy, gz] in rad/s
            accel: np.array [ax, ay, az] in m/s² (convert before calling if needed)
            timestamp: IMU hardware timestamp (raw units, divided by IMU_TIMESTAMP_DIVISOR)
        """
        if self.last_imu_time is None:
            self.last_imu_time = timestamp
            return

        dt = (timestamp - self.last_imu_time) / IMU_TIMESTAMP_DIVISOR
        self.last_imu_time = timestamp

        # Guard against bad timestamps
        if dt <= 0 or dt > 1.0:
            print(f"[WARN] Bad IMU dt={dt:.6f}s, skipping sample.")
            return

        u = np.concatenate([gyro, accel])

        # Step 1: Manually propagate the state
        self.ekf.x = fx(self.ekf.x, dt, u)

        # Step 2: Normalize quaternion to prevent drift
        self.ekf.x = normalize_quat_in_state(self.ekf.x)

        # Step 3: Compute the Jacobian and set it
        self.ekf.F = compute_F_jacobian(self.ekf.x, dt, u)

        # Step 4: Propagate covariance (P = F @ P @ F.T + Q)
        self.ekf.predict()

    def update_gps(self, lat, lon, alt, gps_error=None):
        """Fuse a GPS measurement into the EKF.
        
        Args:
            lat, lon: Geodetic coordinates (degrees)
            alt: Altitude (meters), can be None
            gps_error: Tuple (lat_err, lon_err, alt_err) from GST sentence (meters, 1-sigma)
        """
        if self.reference_gps is None:
            return

        alt = alt if alt is not None else self.reference_gps[2]

        # Dynamic R matrix from GPS-reported accuracy (GST sentence)
        if gps_error and all(e is not None for e in gps_error):
            self.ekf.R = np.diag([
                max(gps_error[1], 0.1) ** 2,  # lon_err → East variance
                max(gps_error[0], 0.1) ** 2,  # lat_err → North variance
                max(gps_error[2], 0.1) ** 2   # alt_err → Up variance
            ])
        else:
            self.ekf.R = np.eye(3) * 2.5  # Default ~1.6m 1-sigma

        try:
            # Convert GPS to local ENU
            z = gps_to_local_enu(
                lat, lon, alt,
                self.reference_gps[0], self.reference_gps[1], self.reference_gps[2]
            )

            # Outlier rejection: discard if > 500m from reference
            if np.linalg.norm(z[:2]) > 500:
                print(f"[WARN] GPS outlier rejected: {z[:2]} meters from reference.")
                return

            # EKF measurement update
            self.ekf.update(z, HJacobian=H_jacobian, Hx=hx)

            # Re-normalize quaternion after update
            self.ekf.x = normalize_quat_in_state(self.ekf.x)

        except Exception as exc:
            print(f"[ERROR] GPS update failed: {exc}")
            # Inflate position covariance to reduce trust in prediction
            self.ekf.P[0:3, 0:3] += np.eye(3) * 10.0

    def get_gps_position(self):
        """Convert current EKF state back to geodetic coordinates.
        
        Returns:
            (lat, lon, alt) or None if no reference is set.
        """
        if self.reference_gps is None:
            return None

        e, n, u = self.ekf.x[0:3]

        try:
            lat, lon, alt = local_enu_to_gps(
                e, n, u,
                self.reference_gps[0], self.reference_gps[1], self.reference_gps[2]
            )
            return lat, lon, alt
        except Exception as exc:
            print(f"[ERROR] ENU→GPS conversion failed: {exc}")
            return None

    def get_heading_deg(self):
        """Extract heading from the EKF state.
        
        Uses velocity direction when moving, quaternion orientation when stationary.
        Returns heading in degrees [0, 360).
        """
        vx, vy = self.ekf.x[3], self.ekf.x[4]  # East, North velocity
        speed = np.sqrt(vx ** 2 + vy ** 2)

        if speed > 0.3:  # Moving: use velocity direction
            heading = np.degrees(np.arctan2(vx, vy)) % 360
            self.last_valid_heading = heading
        else:
            # Stationary: use quaternion yaw
            quat = self.ekf.x[6:10]
            norm = np.linalg.norm(quat)
            if norm > 1e-10:
                r = R.from_quat(quat / norm)
                # ZYX Euler: first angle is yaw (heading)
                heading = r.as_euler('ZYX', degrees=True)[0] % 360
                self.last_valid_heading = heading
            else:
                heading = self.last_valid_heading

        return heading


# ============================================================
# SERIAL PARSING
# ============================================================
def parse_packet(cmd_id, payload):
    """Parse a TM151 IMU binary packet.
    
    CMD 41 (RAW): timestamp(u32), ax, ay, az, gx, gy, gz (float32 each)
    Payload must be at least 28 bytes (4 + 6*4).
    """
    if cmd_id == 41 and len(payload) >= 28:
        # '<I fff fff' = uint32 + 6 floats = 4 + 24 = 28 bytes
        vals = struct.unpack('<I fff fff', payload[:28])
        timestamp = vals[0]
        imu_data = {
            'accel_x': vals[1], 'accel_y': vals[2], 'accel_z': vals[3],
            'gyro_x':  vals[4], 'gyro_y':  vals[5], 'gyro_z':  vals[6],
        }
        return timestamp, imu_data
    return None, None


def read_imu_packet(ser):
    """Read one complete IMU packet from the serial buffer.
    
    Protocol: [0xAA 0x55] [length: 1 byte] [payload: length bytes] [CRC: 2 bytes]
    Returns (timestamp, imu_data_dict) or (None, None).
    """
    if ser.in_waiting < 10:
        return None, None

    max_attempts = 64  # Prevent infinite loop on corrupted data
    attempts = 0

    while ser.in_waiting > 0 and attempts < max_attempts:
        attempts += 1
        byte1 = ser.read(1)
        if byte1 != b'\xAA':
            continue
        byte2 = ser.read(1)
        if byte2 != b'\x55':
            continue

        length_b = ser.read(1)
        if not length_b:
            break
        length = length_b[0]

        if length > 128:  # Sanity check
            continue

        payload = ser.read(length)
        _crc = ser.read(2)  # CRC (not validated here — add if needed)

        if len(payload) < 4:
            continue

        header = struct.unpack('<I', payload[:4])[0]
        cmd_id = header & 0x7F
        return parse_packet(cmd_id, payload[4:])

    return None, None


def read_gps_line(ser):
    """Read and parse one NMEA sentence from the GPS serial port.
    
    Returns a dict with 'type' key ('GGA' or 'GST') or None.
    """
    try:
        line = ser.readline().decode('ascii', errors='ignore').strip()
    except Exception:
        return None

    if not line.startswith('$'):
        return None

    try:
        msg = pynmea2.parse(line)

        if isinstance(msg, pynmea2.types.talker.GGA):
            lat = float(msg.latitude) if msg.latitude else None
            lon = float(msg.longitude) if msg.longitude else None
            alt = float(msg.altitude) if msg.altitude else None
            if lat is not None and lon is not None:
                return {'type': 'GGA', 'lat': lat, 'lon': lon, 'alt': alt}

        elif isinstance(msg, pynmea2.types.talker.GST):
            return {
                'type': 'GST',
                'lat_err': float(msg.data[5]) if len(msg.data) > 5 and msg.data[5] else None,
                'lon_err': float(msg.data[6]) if len(msg.data) > 6 and msg.data[6] else None,
                'alt_err': float(msg.data[7]) if len(msg.data) > 7 and msg.data[7] else None,
            }

    except pynmea2.ParseError:
        pass
    except (ValueError, IndexError):
        pass

    return None


# ============================================================
# CAMERA POSITION BROADCASTING
# ============================================================
def broadcast_all_camera_positions(fused_pos, heading_deg):
    """Compute each camera's GPS position and broadcast via UDP.
    
    Cameras are offset laterally (perpendicular to heading) and
    slightly forward along the heading direction.
    
    Args:
        fused_pos: (lat, lon, alt) of the IMU/vehicle center
        heading_deg: Current heading in degrees [0, 360)
    """
    lat, lon, alt = fused_pos
    heading_rad = np.radians(heading_deg)
    perp_rad = heading_rad + np.pi / 2  # 90° clockwise = right

    positions = {}
    for name, lateral_dist in CAMERA_LATERAL_OFFSETS.items():
        # Forward component (along heading)
        e_fwd = CAMERA_FORWARD_OFFSET * np.sin(heading_rad)
        n_fwd = CAMERA_FORWARD_OFFSET * np.cos(heading_rad)

        # Lateral component (perpendicular to heading)
        e_lat = lateral_dist * np.sin(perp_rad)
        n_lat = lateral_dist * np.cos(perp_rad)

        # Total offset in ENU
        total_e = e_fwd + e_lat
        total_n = n_fwd + n_lat
        total_dist = np.sqrt(total_e ** 2 + total_n ** 2)

        if total_dist > 0.001:
            bearing = np.degrees(np.arctan2(total_e, total_n)) % 360
            ref_point = Point(lat, lon)
            dest = geodesic(meters=total_dist).destination(ref_point, bearing)
            positions[name] = {
                "lat": dest.latitude,
                "lon": dest.longitude,
                "alt": alt
            }
        else:
            positions[name] = {"lat": lat, "lon": lon, "alt": alt}

    message = {**positions, "timestamp": time.time()}

    try:
        sock.sendto(json.dumps(message).encode(), (UDP_IP, UDP_PORT))
    except Exception as exc:
        print(f"[WARN] UDP broadcast failed: {exc}")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    ekf_fusion = SensorFusionEKF()
    data_logger = FastLogger(batch_size=400)

    # Launch camera scripts on remote Jetsons
    launch_all_jetson_scripts()

    # Open serial ports
    imu_ser = None
    gps_ser = None
    try:
        imu_ser = serial.Serial(IMU_PORT, IMU_BAUD, timeout=0.01)
        gps_ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=0.01)
        print(f"[OK] IMU on {IMU_PORT} @ {IMU_BAUD}")
        print(f"[OK] GPS on {GPS_PORT} @ {GPS_BAUD}")
    except serial.SerialException as exc:
        print(f"[FATAL] Failed to open serial ports: {exc}")
        stop_all_jetson_scripts()
        return

    # --- Wait for initial GPS fix ---
    print("Waiting for initial GPS fix (timeout: 30s)...")
    start_time = time.time()
    while ekf_fusion.reference_gps is None and (time.time() - start_time) < 30:
        if gps_ser.in_waiting > 0:
            gps_fix = read_gps_line(gps_ser)
            if gps_fix and gps_fix.get('type') == 'GGA':
                lat = gps_fix.get('lat')
                lon = gps_fix.get('lon')
                alt = gps_fix.get('alt')
                if lat and lon and lat != 0.0 and lon != 0.0:
                    ekf_fusion.set_reference_gps(lat, lon, alt)
                    print(f"[OK] Reference GPS set: {ekf_fusion.reference_gps}")
        time.sleep(0.01)

    if ekf_fusion.reference_gps is None:
        print("[FATAL] No GPS fix within 30s. Exiting.")
        imu_ser.close()
        gps_ser.close()
        stop_all_jetson_scripts()
        return

    # --- Main sensor fusion loop ---
    print("Starting sensor fusion and broadcasting loop...")
    print("Press Ctrl+C to stop.\n")
    last_gps_error = None
    loop_count = 0

    try:
        while True:
            try:
                # ---- 1. IMU Update (high rate: ~400 Hz) ----
                imu_timestamp, imu_data = read_imu_packet(imu_ser)
                if imu_data is not None:
                    gyro = np.array([
                        imu_data['gyro_x'],
                        imu_data['gyro_y'],
                        imu_data['gyro_z']
                    ])
                    accel = np.array([
                        imu_data['accel_x'],
                        imu_data['accel_y'],
                        imu_data['accel_z']
                    ])

                    # Convert g → m/s² if needed
                    if IMU_ACCEL_IN_G:
                        accel *= GRAVITY_MAG

                    ekf_fusion.update_imu(gyro, accel, imu_timestamp)

                # ---- 2. GPS Update (low rate: ~1-10 Hz) ----
                if gps_ser.in_waiting > 0:
                    gps_fix = read_gps_line(gps_ser)
                    if gps_fix is not None:
                        if gps_fix['type'] == 'GGA':
                            ekf_fusion.update_gps(
                                gps_fix['lat'],
                                gps_fix['lon'],
                                gps_fix['alt'],
                                gps_error=last_gps_error
                            )
                        elif gps_fix['type'] == 'GST':
                            last_gps_error = (
                                gps_fix['lat_err'],
                                gps_fix['lon_err'],
                                gps_fix['alt_err']
                            )

                # ---- 3. Output: Broadcast & Log ----
                gps_pos = ekf_fusion.get_gps_position()
                if gps_pos is not None:
                    heading_deg = ekf_fusion.get_heading_deg()

                    # UDP broadcast to camera Jetsons
                    broadcast_all_camera_positions(gps_pos, heading_deg)

                    # Log to CSV (batched)
                    data_logger.log(pd.Timestamp.now(), gps_pos, gps_error=last_gps_error)

                    # Periodic status print
                    loop_count += 1
                    if loop_count % 2000 == 0:
                        print(f"  [STATUS] Pos=({gps_pos[0]:.7f}, {gps_pos[1]:.7f}, {gps_pos[2]:.2f}) "
                              f"Hdg={heading_deg:.1f}° "
                              f"Vel={np.linalg.norm(ekf_fusion.ekf.x[3:6]):.2f} m/s")

                # Prevent 100% CPU usage when serial buffers are empty
                time.sleep(0.0005)

            except serial.SerialException as exc:
                print(f"[ERROR] Serial error: {exc}. Attempting to continue...")
                time.sleep(1.0)
            except Exception as exc:
                print(f"[ERROR] Loop error: {exc}")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[INFO] Shutdown initiated by user (Ctrl+C).")

    finally:
        print("Flushing log buffer and closing connections...")
        data_logger.flush()
        stop_all_jetson_scripts()
        if imu_ser and imu_ser.is_open:
            imu_ser.close()
        if gps_ser and gps_ser.is_open:
            gps_ser.close()
        sock.close()
        print("[OK] Shutdown complete.")


if __name__ == "__main__":
    main()