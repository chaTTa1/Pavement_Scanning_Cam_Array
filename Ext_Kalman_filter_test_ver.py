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
import signal

# Setup UDP broadcast socket
UDP_IP = "192.168.1.255"  
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

m_dist = 0.1
l_dist = 0.6
r_dist = 0.4
CAMISLEFT_FLAG = False

# Map roles to (username, IP) for each Jetson Nano
jetsons = {
    "left":  ("ryan4", "192.168.1.12"),
    "mid":   ("ryan5", "192.168.1.11"),
    "right": ("ryan6", "192.168.1.10"),
}
ssh_processes = {}

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

def broadcast_all_camera_positions(left, mid, right, alt):
    message = {
        "left": {"lat": left[0], "lon": left[1], "alt": alt},
        "mid": {"lat": mid[0], "lon": mid[1], "alt": alt},
        "right": {"lat": right[0], "lon": right[1], "alt": alt}
    }
    sock.sendto(json.dumps(message).encode(), (UDP_IP, UDP_PORT))

def stop_all_jetson_scripts():
    print("[INFO] Stopping blkfly.py scripts on Jetsons...")
    for role, (username, ip) in jetsons.items():
        try:
            subprocess.run(
                ["ssh", f"{username}@{ip}", "pkill -f blkfly.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"[INFO] Stopped {role} blkfly.py on {ip}")
        except Exception as e:
            print(f"[ERROR] Could not stop {role} on {ip}: {e}")

def quaternion_propagate(q, omega, dt):
    """
    Propagate quaternion with angular velocity omega over dt
    """
    r = R.from_quat(q)
    delta_r = R.from_rotvec(omega * dt)
    new_r = r * delta_r
    return new_r.as_quat()


def fx(x, u, dt):
    """
    State transition function
    x: state vector [pos, vel, quat, gyro_bias, acc_bias]
    u: control input [gyro_raw, acc_raw]
    """
    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    b_g = x[10:13]
    b_a = x[13:16]

    gyro = u[0] - b_g
    acc = u[1] - b_a

    quat = quat / np.linalg.norm(quat)
    r = R.from_quat(quat)
    acc_world = r.apply(acc)
    acc_world[2] -= 9.81  # remove gravity

    new_pos = pos + vel * dt + 0.5 * acc_world * dt**2
    new_vel = vel + acc_world * dt
    new_quat = quaternion_propagate(quat, gyro, dt)

    # Bias random walk (allow filter to adapt)
    new_b_g = b_g  # Optionally: + np.random.normal(0, 0.001, 3) * dt
    new_b_a = b_a  # Optionally: + np.random.normal(0, 0.01, 3) * dt

    x_new = np.zeros(16)
    x_new[0:3] = new_pos
    x_new[3:6] = new_vel
    x_new[6:10] = new_quat
    x_new[10:13] = new_b_g
    x_new[13:16] = new_b_a
    return x_new


def hx(x):
    return x[0:3]  # GPS measures position only


def H_jacobian(x):
    H = np.zeros((3, 16))
    H[:, :3] = np.eye(3)
    return H


# Initialize EKF
ekf = ExtendedKalmanFilter(dim_x=16, dim_z=3)
ekf.x = np.zeros(16)
ekf.x[6] = 1.0  # Initialize quaternion to identity
ekf.P = np.eye(16) * 100
ekf.Q = np.eye(16) * 10.0
ekf.Q[10:13, 10:13] = np.eye(3) * 0.01  # gyro bias
ekf.Q[13:16, 13:16] = np.eye(3) * 2.0  # or even higher
ekf.R = np.eye(3) * 0.0001


# --- Serial port setup ---
IMU_PORT = '/dev/ttyACM0'
IMU_BAUD = 115200
GPS_PORT = '/dev/ttyACM2'
GPS_BAUD = 115200

imu_ser = serial.Serial(IMU_PORT, IMU_BAUD, timeout=0.01)
gps_ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=0.01)

last_imu_time = None

# --- IMU packet parsing (copied from IMU_to_csv.py) ---
CMD_QUATERNION = 32
CMD_RPY = 35
CMD_RAW = 41

def parse_packet(cmd_id, payload):
    if cmd_id == CMD_QUATERNION:
        timestamp, q1, q2, q3, q4 = struct.unpack('<Iffff', payload)
        return timestamp, {'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4}
    elif cmd_id == CMD_RPY:
        timestamp, roll, pitch, yaw = struct.unpack('<Ifff', payload)
        return timestamp, {'roll': roll, 'pitch': pitch, 'yaw': yaw}
    elif cmd_id == CMD_RAW:
        vals = struct.unpack('<Ifff fff fff', payload)
        return vals[0], {
            'accel_x': vals[1], 'accel_y': vals[2], 'accel_z': vals[3],
            'gyro_x': vals[4], 'gyro_y': vals[5], 'gyro_z': vals[6],
            'mag_x': vals[7], 'mag_y': vals[8], 'mag_z': vals[9]
        }
    else:
        return None, {}

def read_imu_packet(ser):
    while True:
        if ser.read() != b'\xAA' or ser.read() != b'\x55':
            continue
        length = ser.read(1)
        if not length:
            continue
        length = length[0]
        payload = ser.read(length)
        crc = ser.read(2)
        if len(payload) < 4:
            continue
        header = struct.unpack('<I', payload[:4])[0]
        cmd_id = header & 0x7F
        timestamp, data = parse_packet(cmd_id, payload[4:])
        if timestamp is None:
            continue
        return timestamp, data

def read_gps_line(ser):
    """
    Parse GPS data exactly as in Real_Time_Mosaic_Parser.py.
    Returns (lat, lon, alt) if GGA sentence, else None.
    """
    line = ser.readline().decode('ascii', errors='ignore').strip()
    if not line.startswith('$'):
        return None
    try:
        msg = pynmea2.parse(line)

        if isinstance(msg, pynmea2.types.talker.GGA):
            # For debugging, you can uncomment the next line:
            print(f"[GGA] TOW: {msg.timestamp} | Lat: {msg.latitude}° | Lon: {msg.longitude}° | Alt: {msg.altitude} {msg.altitude_units}")
            return float(msg.latitude), float(msg.longitude), float(msg.altitude)

        elif isinstance(msg, pynmea2.types.talker.HDT):
            # print(f"[HDT] Heading: {msg.heading}°")
            return None

        elif isinstance(msg, pynmea2.types.talker.RMC):
            # print(f"[RMC] Time: {msg.timestamp}, Date: {msg.datestamp}")
            return None

    except pynmea2.ParseError:
        return None

reference_gps = None
reference_gps_samples = []
reference_gps_time = None
REFERENCE_GPS_SAMPLE_COUNT = 10
REFERENCE_GPS_STATIONARY_SECONDS = 2.0
REFERENCE_GPS_POSITION_TOL = 0.00001  

# Wait for first valid GPS fix to set reference_gps
while reference_gps is None:
    gps_fix = read_gps_line(gps_ser)
    if gps_fix is not None:
        reference_gps = gps_fix
        print(f"Reference GPS set: {reference_gps}")

def convert_camera_gps(lat, lon, alt, heading_deg, camisleft_flag):
    """
    Converts main GPS + heading to left/mid/right camera GPS positions.
    """
    # Camera offsets (meters)
    l_dist = 0.6
    r_dist = 0.4
    m_dist = 0.05

    # Convert heading to radians
    heading = np.radians(heading_deg)

    # Calculate ENU offsets for each camera
    # Left camera
    left_e = l_dist * np.sin(heading)
    left_n = l_dist * np.cos(heading)
    # Mid camera
    mid_e = m_dist * np.sin(heading)
    mid_n = m_dist * np.cos(heading)
    # Right camera
    right_e = r_dist * np.sin(heading)
    right_n = r_dist * np.cos(heading)

    # Reference point
    ref_point = Point(lat, lon)

    # Calculate GPS positions using geopy
    left_point = geodesic(meters=np.sqrt(left_e**2 + left_n**2)).destination(ref_point, (np.degrees(np.arctan2(left_e, left_n)) % 360))
    mid_point = geodesic(meters=np.sqrt(mid_e**2 + mid_n**2)).destination(ref_point, (np.degrees(np.arctan2(mid_e, mid_n)) % 360))
    right_point = geodesic(meters=np.sqrt(right_e**2 + right_n**2)).destination(ref_point, (np.degrees(np.arctan2(right_e, right_n)) % 360))
    #print(left_point, mid_point, right_point)
    return (
        (left_point.latitude, left_point.longitude),
        (mid_point.latitude, mid_point.longitude),
        (right_point.latitude, right_point.longitude)
    )

def broadcast_gps(lat, lon, alt, heading_deg):
    """
    Broadcasts all three camera GPS positions.
    """
    left, mid, right = convert_camera_gps(lat, lon, alt, heading_deg, CAMISLEFT_FLAG)
    broadcast_all_camera_positions(left, mid, right, alt)


# --- Main fusion loop ---
try:    
    while True:
        loop_start = time.perf_counter()

        # --- Read IMU ---
        imu_packet = read_imu_packet(imu_ser)
        if imu_packet is None:
            continue
        imu_timestamp, imu_data = imu_packet

        # Compose IMU input for EKF (match field names from IMU_to_csv.py)
        gyro = np.array([
            imu_data.get('gyro_x', 0),
            imu_data.get('gyro_y', 0),
            imu_data.get('gyro_z', 0)
        ])
        acc = np.array([
            imu_data.get('accel_x', 0),
            imu_data.get('accel_y', 0),
            imu_data.get('accel_z', 0)
        ])
        u = [gyro, acc]

        # Time step
        if last_imu_time is None:
            last_imu_time = imu_timestamp
            continue
        dt = (imu_timestamp - last_imu_time) / 1e6
        last_imu_time = imu_timestamp

        # EKF predict (dead-reckoning)
        ekf.x = fx(ekf.x, u, dt)

        # --- GPS update ---
        gps_fix = read_gps_line(gps_ser)
        if gps_fix is not None and reference_gps is not None:
            lat, lon, alt = gps_fix
            # Convert to ENU
            e = geodesic((reference_gps[0], reference_gps[1]), (reference_gps[0], lon)).meters * (1 if lon >= reference_gps[1] else -1)
            n = geodesic((reference_gps[0], reference_gps[1]), (lat, reference_gps[1])).meters * (1 if lat >= reference_gps[0] else -1)
            u_ = alt - reference_gps[2]
            # Directly set EKF position to GPS
            ekf.x[0:3] = [e, n, u_]

        # --- Output current state as GPS ---
        if reference_gps is not None:
            x, y, z_ = ekf.x[0:3]  # EKF position in meters (ENU)
            flat_distance = np.sqrt(x**2 + y**2)
            bearing = np.degrees(np.arctan2(x, y)) % 360  # East-North
            ref_point = Point(reference_gps[0], reference_gps[1])
            dest = geodesic(meters=flat_distance).destination(ref_point, bearing)
            alt_out = reference_gps[2] + z_
            heading_deg = bearing  # Use EKF bearing as heading
            print(f"EKF GPS: lat={dest.latitude:.8f}, lon={dest.longitude:.8f}, alt={alt_out:.2f}")
            broadcast_gps(dest.latitude, dest.longitude, alt_out, heading_deg)

        # Remove timing printouts
        # loop_end = time.perf_counter()
        # loop_time = loop_end - loop_start
        # predict_time = predict_end - predict_start
        # if update_time is not None:
        #     print(f"Loop: {loop_time1000:.2f} ms | Predict: {predict_time1000:.2f} ms | Update: {update_time*1000:.2f} ms")
        # else:
        #     print(f"Loop: {loop_time1000:.2f} ms | Predict: {predict_time1000:.2f} ms")

        # Sleep to avoid busy loop (optional)
        time.sleep(0.001)
except KeyboardInterrupt:
    print("\n[INFO] Caught Ctrl+C. Shutting down...")
    stop_all_jetson_scripts()
