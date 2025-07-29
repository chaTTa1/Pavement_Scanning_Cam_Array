import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from scipy.spatial.transform import Rotation as R
import serial
import pynmea2
import struct
import time
from geopy.distance import geodesic
from geopy.point import Point
import socket
import json
import subprocess
import pandas as pd
import os

# Setup UDP broadcast socket
UDP_IP = "192.168.1.255"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# Camera hardware configuration
m_dist = 0.01
l_dist = 0.6
r_dist = 0.4
CAMISLEFT_FLAG = False

# Map roles to (username, IP) for each Jetson Nano
jetsons = {
    "left":  ("ryan4", "192.168.1.12"),
    "mid":   ("ryan5", "192.168.1.11"),
    "right": ("ryan6", "192.168.1.10"),
}

# --- SSH Process Management ---
ssh_processes = {}
print("[INFO] Launching blkfly.py scripts on all Jetsons...")
for role, (username, ip) in jetsons.items():
    try:
        proc = subprocess.Popen(
            [
                "ssh", f"{username}@{ip}",
                f"DEVICE_ROLE={role} python3 ~/Alex/Blackfly/blkfly.py"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        ssh_processes[role] = proc
        print(f"[INFO] Launched {role} blkfly.py on {username}@{ip}")
    except Exception as e:
        print(f"[ERROR] Could not launch {role} on {ip}: {e}")

def stop_all_jetson_scripts():
    print("[INFO] Stopping blkfly.py scripts on Jetsons...")
    for role, (username, ip) in jetsons.items():
        try:
            subprocess.run(
                ["ssh", f"{username}@{ip}", "pkill -f blkfly.py"],
                check=True,
                capture_output=True, text=True
            )
            print(f"[INFO] Stopped {role} blkfly.py on {ip}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to stop {role} on {ip}: {e.stderr}")
        except Exception as e:
            print(f"[ERROR] Could not stop {role} on {ip}: {e}")

# --- EKF Helper Functions ---
def quaternion_propagate(q, omega, dt):
    r = R.from_quat(q)
    delta_r = R.from_rotvec(omega * dt)
    new_r = r * delta_r
    return new_r.as_quat()

def fx(x, dt, u):
    pos, vel, quat, b_g, b_a = x[0:3], x[3:6], x[6:10], x[10:13], x[13:16]
    gyro = u[:3] - b_g
    acc = u[3:6] - b_a
    quat = quat / np.linalg.norm(quat)
    r = R.from_quat(quat)
    acc_world = r.apply(acc)
    acc_world[2] -= 9.81
    new_pos = pos + vel * dt + 0.5 * acc_world * dt**2
    new_vel = vel + acc_world * dt
    new_quat = quaternion_propagate(quat, gyro, dt)
    x_new = np.zeros(16)
    x_new[0:3], x_new[3:6], x_new[6:10], x_new[10:13], x_new[13:16] = new_pos, new_vel, new_quat, b_g, b_a
    return x_new

def hx(x):
    return x[0:3]

def H_jacobian(x):
    H = np.zeros((3, 16))
    H[:, :3] = np.eye(3)
    return H

# --- Logging ---
def log_position_and_error(timestamp, gps_pos, gps_error=None):
    data = {'timestamp': timestamp, 'lat': gps_pos[0], 'lon': gps_pos[1], 'alt': gps_pos[2]}
    if gps_error:
        data.update({'lat_err': gps_error[0], 'lon_err': gps_error[1], 'alt_err': gps_error[2]})
    df = pd.DataFrame([data])
    df.to_csv('position_log.csv', mode='a', header=not os.path.exists('position_log.csv'), index=False)

# --- Sensor Fusion Class ---
class SensorFusionEKF:
    def __init__(self):
        self.ekf = ExtendedKalmanFilter(dim_x=16, dim_z=3)
        self.ekf.x = np.zeros(16)
        self.ekf.x[6] = 1.0
        self.ekf.P = np.eye(16) * 100
        self.ekf.Q = np.eye(16) * 0.1
        self.ekf.Q[0:3, 0:3], self.ekf.Q[3:6, 3:6] = np.eye(3) * 0.01, np.eye(3) * 0.1
        self.ekf.Q[6:10, 6:10], self.ekf.Q[10:13, 10:13] = np.eye(4) * 0.01, np.eye(3) * 0.001
        self.ekf.Q[13:16, 13:16] = np.eye(3) * 0.01
        self.ekf.R = np.eye(3) * 1.0
        self.reference_gps = None
        self.last_imu_time = None
        self.dt = 0.01

    def set_reference_gps(self, lat, lon, alt):
        self.reference_gps = (lat, lon, alt if alt is not None else 0.0)

    def update_imu(self, gyro, accel, timestamp):
        if self.last_imu_time is None:
            self.last_imu_time = timestamp
            return
        dt = (timestamp - self.last_imu_time) / 1e6
        self.last_imu_time, self.dt = timestamp, dt
        u = np.concatenate([gyro, accel])
        self.ekf.predict(fx=fx, dt=self.dt, u=u)

    def update_gps(self, lat, lon, alt):
        if self.reference_gps is None: return
        alt = alt if alt is not None else self.reference_gps[2]
        try:
            e = geodesic((self.reference_gps[0], self.reference_gps[1]), (self.reference_gps[0], lon)).meters * (1 if lon >= self.reference_gps[1] else -1)
            n = geodesic((self.reference_gps[0], self.reference_gps[1]), (lat, self.reference_gps[1])).meters * (1 if lat >= self.reference_gps[0] else -1)
            if abs(e) > 1000 or abs(n) > 1000:
                print(f"Warning: Large ENU coordinates (e={e:.1f}, n={n:.1f}), skipping update")
                return
            u_coord = alt - self.reference_gps[2]
            z = np.array([e, n, u_coord])
            self.ekf.update(z, HJacobian=H_jacobian, Hx=hx)
        except Exception as e:
            print(f"GPS update error: {e}")
            self.ekf.P[0:3, 0:3] = np.eye(3) * 100

    def get_gps_position(self):
        if self.reference_gps is None: return None
        x, y, z = self.ekf.x[0:3]
        try:
            flat_distance = np.sqrt(x**2 + y**2)
            bearing = np.degrees(np.arctan2(x, y)) % 360
            ref_point = Point(self.reference_gps[0], self.reference_gps[1])
            dest = geodesic(meters=flat_distance).destination(ref_point, bearing)
            return dest.latitude, dest.longitude, self.reference_gps[2] + z
        except Exception as e:
            print(f"Position conversion error: {e}")
            return None

# --- Serial & Parsing ---
IMU_PORT, GPS_PORT = 'COM3', 'COM10'
IMU_BAUD, GPS_BAUD = 115200, 115200

def parse_imu_packet(ser):
    while True:
        if ser.read() != b'\xAA' or ser.read() != b'\x55': continue
        length = ser.read(1)
        if not length: continue
        payload = ser.read(length[0])
        if len(payload) < 4: continue
        header = struct.unpack('<I', payload[:4])[0]
        cmd_id = header & 0x7F
        if cmd_id == 41:
            vals = struct.unpack('<Iffff' 'fff' 'fff', payload[4:])
            return vals[0], {
                'accel_x': vals[1], 'accel_y': vals[2], 'accel_z': vals[3],
                'gyro_x': vals[4], 'gyro_y': vals[5], 'gyro_z': vals[6]
            }
        return None, {}

def read_gps_line(ser):
    line = ser.readline().decode('ascii', errors='ignore').strip()
    if not line.startswith('$'): return None
    try:
        msg = pynmea2.parse(line)
        if isinstance(msg, pynmea2.types.talker.GGA):
            return {'type': 'GGA', 'lat': float(msg.latitude), 'lon': float(msg.longitude), 'alt': float(msg.altitude) if msg.altitude else None}
        elif isinstance(msg, pynmea2.types.talker.GST):
            return {'type': 'GST', 'lat_err': float(msg.data[5]), 'lon_err': float(msg.data[6]), 'alt_err': float(msg.data[7])}
    except pynmea2.ParseError:
        return None

# --- Camera Position Calculation and Broadcasting ---
def convert_camera_gps(lat, lon, alt, heading_deg):
    heading = np.radians(heading_deg)
    ref_point = Point(lat, lon)
    positions = {}
    offsets = {"left": l_dist, "mid": m_dist, "right": r_dist}
    for name, dist in offsets.items():
        e, n = dist * np.sin(heading), dist * np.cos(heading)
        dest = geodesic(meters=np.sqrt(e**2 + n**2)).destination(ref_point, (np.degrees(np.arctan2(e, n)) % 360))
        positions[name] = {"lat": dest.latitude, "lon": dest.longitude, "alt": alt}
    return positions

def broadcast_all_camera_positions(fused_pos, heading_deg):
    lat, lon, alt = fused_pos
    positions = convert_camera_gps(lat, lon, alt, heading_deg)
    message = {**positions, "timestamp": time.time()}
    sock.sendto(json.dumps(message).encode(), (UDP_IP, UDP_PORT))

# --- Main Execution ---
def main():
    ekf_fusion = SensorFusionEKF()
    try:
        imu_ser = serial.Serial(IMU_PORT, IMU_BAUD, timeout=0.01)
        gps_ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=0.01)
    except serial.SerialException as e:
        print(f"Failed to open serial ports: {e}")
        return

    print("Waiting for initial GPS fix to set reference...")
    while ekf_fusion.reference_gps is None:
        gps_data = read_gps_line(gps_ser)
        if gps_data and gps_data.get('type') == 'GGA' and gps_data.get('lat', 0) != 0.0:
            ekf_fusion.set_reference_gps(gps_data['lat'], gps_data['lon'], gps_data['alt'])
            print(f"Reference GPS set: {ekf_fusion.reference_gps}")

    print("Starting sensor fusion and broadcasting loop...")
    last_gps_error = None
    try:
        while True:
            # IMU Update
            imu_timestamp, imu_data = parse_imu_packet(imu_ser)
            if imu_data:
                gyro = np.array([imu_data.get(f'gyro_{axis}', 0) for axis in 'xyz'])
                accel = np.array([imu_data.get(f'accel_{axis}', 0) for axis in 'xyz'])
                ekf_fusion.update_imu(gyro, accel, imu_timestamp)

            # GPS Update
            gps_data = read_gps_line(gps_ser)
            if gps_data:
                if gps_data['type'] == 'GGA':
                    ekf_fusion.update_gps(gps_data['lat'], gps_data['lon'], gps_data['alt'])
                elif gps_data['type'] == 'GST':
                    last_gps_error = (gps_data['lat_err'], gps_data['lon_err'], gps_data['alt_err'])

            # Get fused position and broadcast
            fused_pos = ekf_fusion.get_gps_position()
            if fused_pos:
                # Use EKF bearing for heading
                x, y, _ = ekf_fusion.ekf.x[0:3]
                heading_deg = np.degrees(np.arctan2(x, y)) % 360
                print(f"EKF GPS: lat={fused_pos[0]:.8f}, lon={fused_pos[1]:.8f}, alt={fused_pos[2]:.2f}")
                broadcast_all_camera_positions(fused_pos, heading_deg)
                log_position_and_error(pd.Timestamp.now(), fused_pos, gps_error=last_gps_error)
                last_gps_error = None # Reset error after logging

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] Shutdown initiated by user.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in the main loop: {e}")
    finally:
        stop_all_jetson_scripts()
        imu_ser.close()
        gps_ser.close()
        print("Serial ports closed. Shutdown complete.")

if __name__ == "__main__":
    main()