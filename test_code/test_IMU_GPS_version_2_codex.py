# -*- coding: utf-8 -*-
"""
GPS/IMU Sensor Fusion using Extended Kalman Filter
===================================================
Hardware:
  - SYD Dynamics TransducerM TM151 AHRS 9-Axis IMU at 400 Hz
    Binary EasyProfile protocol
    Accel: includes gravity (reads ~1g on Z when stationary, in g-units)
    Gyro: rad/s, Timestamp: microseconds
  - GPS at ~5-10 Hz (NMEA: GGA, GST, HDT, RMC, VTG)

Architecture:
  - Thread 1: IMU reader → priority queue (by wall-clock arrival)
  - Thread 2: GPS reader → same priority queue
  - Main thread: dequeue in order → EKF predict/update

Modes:
  - GPS+IMU: Full EKF fusion, logs position.csv
  - IMU-only: No EKF. Logs raw sensor data + integrated velocity
              to IMU_only.csv.

Usage:
  python fusion.py                          # auto-detect ports
  python fusion.py --list-ports             # show available ports
  python fusion.py --imu-port COM5          # override IMU port
  python fusion.py --gps-port /dev/ttyUSB0  # override GPS port
  python fusion.py --no-gps                 # IMU-only mode (raw logging)
  python fusion.py --no-imu                 # GPS-only mode
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.linalg as linalg
import serial
import serial.tools.list_ports
import pynmea2
import struct
import time
import csv
import os
import sys
import argparse
import threading
import queue
from geopy.distance import geodesic
from geopy.point import Point


# ─────────────────────────────────────────────
# PLATFORM DETECTION & PORT CONFIGURATION
# ─────────────────────────────────────────────

def detect_os():
    if sys.platform.startswith("win"):
        return "Windows"
    elif sys.platform.startswith("linux"):
        return "Linux"
    else:
        print(f"[WARNING] Unknown platform '{sys.platform}', "
              f"using Linux defaults")
        return "Linux"


def get_default_ports(os_name):
    if os_name == "Windows":
        return "COM12", "COM8"
    else:
        return "/dev/ttyACM2", "/dev/ttyACM0"


def list_serial_ports():
    try:
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("  No serial ports found.")
            return
        for p in ports:
            if p.vid is not None:
                print(
                    f"  {p.device:20s} | "
                    f"VID:PID={p.vid:04X}:{p.pid:04X} | "
                    f"{p.description}")
            else:
                print(
                    f"  {p.device:20s} | "
                    f"VID:PID=None         | "
                    f"{p.description}")
    except Exception as exc:
        print(f"  Error listing ports: {exc}")


def find_septentrio_nmea_port(timeout=3.0):
    try:
        ports = serial.tools.list_ports.comports()
    except Exception:
        return None

    septentrio_ports = []
    for p in ports:
        if p.vid == 0x152A and p.pid == 0x85C0:
            septentrio_ports.append(p.device)

    if not septentrio_ports:
        return None

    print(f"Found {len(septentrio_ports)} Septentrio ports: "
          f"{septentrio_ports}")

    for port in septentrio_ports:
        print(f"  Testing {port} for NMEA data...")
        try:
            ser = serial.Serial(port, 115200, timeout=0.5)
            start = time.time()
            nmea_count = 0

            while time.time() - start < timeout:
                try:
                    raw = ser.readline()
                    if raw:
                        line = raw.decode(
                            "ascii", errors="ignore").strip()
                        if line.startswith("$") and (
                                "GGA" in line or "GST" in line
                                or "HDT" in line or "RMC" in line
                                or "GSV" in line):
                            nmea_count += 1
                            if nmea_count >= 3:
                                ser.close()
                                print(f"  -> {port} is NMEA port "
                                      f"({nmea_count} sentences)")
                                return port
                except Exception:
                    pass

            ser.close()
            print(f"  -> {port}: only {nmea_count} NMEA sentences")

        except serial.SerialException as exc:
            print(f"  -> {port}: could not open ({exc})")

    print("  No Septentrio NMEA port found")
    return None


def auto_detect_ports(os_name):
    IMU_USB_IDS = [(0x0483, 0x5740)]
    imu_port = None
    gps_port = None

    try:
        ports = serial.tools.list_ports.comports()
    except Exception:
        return None, None

    print("Scanning serial ports...")
    for p in ports:
        vid = p.vid
        pid = p.pid

        if vid is not None:
            print(
                f"  {p.device:20s} | "
                f"VID:PID={vid:04X}:{pid:04X} | "
                f"{p.description}")
        else:
            print(
                f"  {p.device:20s} | "
                f"VID:PID=None         | "
                f"{p.description}")
            continue

        for known_vid, known_pid in IMU_USB_IDS:
            if vid == known_vid and pid == known_pid:
                if imu_port is None:
                    imu_port = p.device
                    print(f"  -> Detected IMU: {p.device}")

    gps_port = find_septentrio_nmea_port(timeout=3.0)

    if gps_port is None:
        OTHER_GPS_IDS = [
            (0x1546, 0x01A8), (0x10C4, 0xEA60),
            (0x067B, 0x2303), (0x0403, 0x6001),
        ]
        for p in ports:
            if p.vid is None:
                continue
            for known_vid, known_pid in OTHER_GPS_IDS:
                if p.vid == known_vid and p.pid == known_pid:
                    if gps_port is None:
                        gps_port = p.device
                        print(f"  -> Detected GPS: {p.device}")

    return imu_port, gps_port


def resolve_ports(os_name, args):
    default_imu, default_gps = get_default_ports(os_name)
    auto_imu, auto_gps = auto_detect_ports(os_name)

    imu_port = args.imu_port or auto_imu or default_imu
    gps_port = args.gps_port or auto_gps or default_gps

    imu_source = ("command-line" if args.imu_port
                  else "auto-detected" if auto_imu
                  else "default")
    gps_source = ("command-line" if args.gps_port
                  else "auto-detected" if auto_gps
                  else "default")

    print(f"IMU port: {imu_port} ({imu_source})")
    print(f"GPS port: {gps_port} ({gps_source})")

    return imu_port, gps_port


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPS/IMU Sensor Fusion EKF")
    parser.add_argument("--imu-port", type=str, default=None)
    parser.add_argument("--gps-port", type=str, default=None)
    parser.add_argument("--imu-baud", type=int, default=None)
    parser.add_argument("--gps-baud", type=int, default=None)
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--no-gps", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--skip-crc", action="store_true",
                        help="Disable CRC checking on IMU packets")
    parser.add_argument("--print-every", type=float, default=0.25,
                        help="Console print interval in seconds")
    return parser.parse_args()


# ─────────────────────────────────────────────
# RESOLVE CONFIGURATION
# ─────────────────────────────────────────────

_args = parse_args()

if _args.list_ports:
    print("Available serial ports:")
    list_serial_ports()
    sys.exit(0)

OS_NAME = detect_os()

IMU_ENABLED = not _args.no_imu
GPS_ENABLED = not _args.no_gps
PRINT_EVERY = _args.print_every

IMU_PORT, GPS_PORT = resolve_ports(OS_NAME, _args)

# ─── IMU settings ───
ACCEL_INPUT_IS_G = True
IMU_ACCEL_IS_GRAVITY_FREE = False
GRAVITY_MPS2 = 9.81
IMU_BAUD = _args.imu_baud if _args.imu_baud else 115200

# ─── GPS settings ───
GPS_BAUD = _args.gps_baud if _args.gps_baud else 115200
UERE_SIGMA_M = 4.0
MIN_FIX_QUALITY = 1
MAX_RAW_GPS_JUMP_M = 15.0

# ─── GPS velocity settings ───
GPS_VEL_RMC_SIGMA_MPS = 0.1
GPS_VEL_FDIFF_SIGMA_MPS = 2.0
GPS_VEL_FDIFF_MAX_DT = 1.5
GPS_VEL_FDIFF_MIN_DT = 0.05
GPS_VEL_VERTICAL_SIGMA_SCALE = 3.0

# ─── EKF safety ───
MAX_EKF_POSITION_RADIUS_M = 1000.0
MAX_EKF_ALTITUDE_OFFSET_M = 200.0
MAX_VALID_IMU_DT = 0.02

# ─── Debug ───
PRINT_RAW_GPS_DEBUG = False
PRINT_ALL_NMEA_LINES = False

# ─── Process noise ───
ACCEL_NOISE_DENSITY = 0.05
GYRO_NOISE_DENSITY = 0.005
ACCEL_BIAS_RANDOM_WALK = 0.001
GYRO_BIAS_RANDOM_WALK = 0.0001
POSITION_PROCESS_NOISE = 0.001

# ─── IMU protocol ───
IMU_SYNC_BYTE_1 = 0xAA
IMU_SYNC_BYTE_2 = 0x55
IMU_CMD_RAW = 41
IMU_CMD_QUAT = 32
IMU_CMD_RPY = 35
IMU_CMD_GRAVITY = 36
IMU_CRC_BYTES = 2

# ─── Heading ───
DEFAULT_HEADING_SIGMA_DEG = 5.0

# ─── Quaternion measurement ───
QUAT_MEASUREMENT_SIGMA_RAD = 0.01

# ─── Gravity alignment ───
GRAVITY_INIT_SAMPLES = 200

# ─── GPS timeout ───
GPS_FIX_TIMEOUT_S = 30

# ─── Queue ───
SENSOR_QUEUE_MAXSIZE = 2000

print(f"OS: {OS_NAME}")
print(f"IMU: {'ENABLED' if IMU_ENABLED else 'DISABLED'} | "
      f"{IMU_PORT} @ {IMU_BAUD}")
print(f"GPS: {'ENABLED' if GPS_ENABLED else 'DISABLED'} | "
      f"{GPS_PORT} @ {GPS_BAUD}")
print(f"CRC check: {'off' if _args.skip_crc else 'on'}")
print(f"Print interval: {PRINT_EVERY}s")


# ─────────────────────────────────────────────
# CRC16 MODBUS
# ─────────────────────────────────────────────

def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


# ─────────────────────────────────────────────
# QUATERNION & ANGLE UTILITIES
# ─────────────────────────────────────────────

def normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


def quaternion_propagate(q, omega, dt):
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-12:
        return q.copy()
    r = R.from_quat(q)
    delta_r = R.from_rotvec(omega * dt)
    new_r = r * delta_r
    return new_r.as_quat()


def normalize_angle(angle_rad):
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def heading_deg_to_yaw_rad(heading_deg):
    scipy_yaw_deg = (360.0 - heading_deg) % 360.0
    return normalize_angle(np.deg2rad(scipy_yaw_deg))


def quaternion_to_yaw(q):
    q = normalize_quaternion(q)
    r = R.from_quat(q)
    yaw = r.as_euler("zyx", degrees=False)[0]
    return normalize_angle(yaw)


def angle_residual(z, z_pred):
    return np.array([
        normalize_angle(float(z[0]) - float(z_pred[0]))])


def quaternion_residual(q_meas, q_pred):
    q_meas = normalize_quaternion(q_meas)
    q_pred = normalize_quaternion(q_pred)
    r_meas = R.from_quat(q_meas)
    r_pred = R.from_quat(q_pred)
    r_err = r_meas * r_pred.inv()
    return r_err.as_rotvec()


def initialize_orientation_from_accel(accel_avg_mps2):
    ax, ay, az = accel_avg_mps2
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))
    r = R.from_euler('zyx', [0.0, pitch, roll])
    return r.as_quat()


def course_speed_to_enu_velocity(course_deg, speed_mps):
    course_rad = np.deg2rad(course_deg)
    ve = speed_mps * np.sin(course_rad)
    vn = speed_mps * np.cos(course_rad)
    return ve, vn


def format_vec(v, precision=5):
    return "[" + ", ".join(
        f"{x:.{precision}f}" for x in v) + "]"


# ─────────────────────────────────────────────
# NMEA UTILITIES
# ─────────────────────────────────────────────

def valid_nmea_checksum(line):
    try:
        if not line.startswith("$") or "*" not in line:
            return False
        data, checksum = line.strip().split("*", 1)
        calc = 0
        for c in data[1:]:
            calc ^= ord(c)
        return calc == int(checksum[:2], 16)
    except Exception:
        return False


def nmea_to_decimal_degrees(value_str, direction, is_latitude):
    if value_str is None or value_str == "":
        return None
    value_str = value_str.strip()
    if len(value_str) < 6:
        return None
    try:
        value = float(value_str)
    except ValueError:
        return None

    degrees = int(value // 100)
    minutes = value - degrees * 100
    decimal = degrees + minutes / 60.0

    if direction in ("S", "W"):
        decimal = -decimal

    if is_latitude and not (-90.0 <= decimal <= 90.0):
        return None
    if not is_latitude and not (-180.0 <= decimal <= 180.0):
        return None

    return decimal


# ─────────────────────────────────────────────
# STATE TRANSITION MODEL
# ─────────────────────────────────────────────

def fx(x, dt, u):
    pos = x[0:3].copy()
    vel = x[3:6].copy()
    quat = normalize_quaternion(x[6:10].copy())
    b_g = x[10:13].copy()
    b_a = x[13:16].copy()

    gyro = u[:3] - b_g
    acc = u[3:6] - b_a

    if ACCEL_INPUT_IS_G:
        acc = acc * GRAVITY_MPS2

    rot = R.from_quat(quat)
    acc_world = rot.apply(acc)

    if not IMU_ACCEL_IS_GRAVITY_FREE:
        acc_world[2] -= GRAVITY_MPS2

    new_pos = pos + vel * dt + 0.5 * acc_world * dt ** 2
    new_vel = vel + acc_world * dt
    new_quat = quaternion_propagate(quat, gyro, dt)

    x_new = np.zeros(16)
    x_new[0:3] = new_pos
    x_new[3:6] = new_vel
    x_new[6:10] = normalize_quaternion(new_quat)
    x_new[10:13] = b_g
    x_new[13:16] = b_a
    return x_new


def compute_F_jacobian(x, dt, u, eps=1e-7):
    n = len(x)
    f0 = fx(x, dt, u)
    F = np.zeros((n, n))
    for i in range(n):
        x_pert = x.copy()
        x_pert[i] += eps
        if 6 <= i <= 9:
            x_pert[6:10] = normalize_quaternion(x_pert[6:10])
        fi = fx(x_pert, dt, u)
        F[:, i] = (fi - f0) / eps
    return F


def compute_Q(dt):
    Q = np.zeros((16, 16))
    Q[0:3, 0:3] = np.eye(3) * (POSITION_PROCESS_NOISE ** 2) * dt
    Q[3:6, 3:6] = np.eye(3) * (ACCEL_NOISE_DENSITY ** 2) * dt
    Q[6:10, 6:10] = np.eye(4) * (GYRO_NOISE_DENSITY ** 2) * dt
    Q[10:13, 10:13] = np.eye(3) * (
        GYRO_BIAS_RANDOM_WALK ** 2) * dt
    Q[13:16, 13:16] = np.eye(3) * (
        ACCEL_BIAS_RANDOM_WALK ** 2) * dt
    return Q


# ─────────────────────────────────────────────
# MEASUREMENT MODELS
# ─────────────────────────────────────────────

def hx_position(x):
    return x[0:3].copy()


def H_jacobian_position(x):
    H = np.zeros((3, 16))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    return H


def hx_velocity(x):
    return x[3:6].copy()


def H_jacobian_velocity(x):
    H = np.zeros((3, 16))
    H[0, 3] = 1.0
    H[1, 4] = 1.0
    H[2, 5] = 1.0
    return H


def hx_pos_vel(x):
    return np.concatenate([x[0:3], x[3:6]])


def H_jacobian_pos_vel(x):
    H = np.zeros((6, 16))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    H[3, 3] = 1.0
    H[4, 4] = 1.0
    H[5, 5] = 1.0
    return H


def hx_heading(x):
    return np.array([quaternion_to_yaw(x[6:10])])


def H_jacobian_heading(x):
    H = np.zeros((1, 16))
    base_yaw = hx_heading(x)[0]
    eps = 1e-6
    for idx in range(6, 10):
        x_pert = x.copy()
        x_pert[idx] += eps
        x_pert[6:10] = normalize_quaternion(x_pert[6:10])
        perturbed_yaw = hx_heading(x_pert)[0]
        H[0, idx] = normalize_angle(
            perturbed_yaw - base_yaw) / eps
    return H


def H_jacobian_quaternion(x, eps=1e-7):
    H = np.zeros((3, 16))
    q_pred = normalize_quaternion(x[6:10])
    for i in range(6, 10):
        x_pert = x.copy()
        x_pert[i] += eps
        x_pert[6:10] = normalize_quaternion(x_pert[6:10])
        q_pert = normalize_quaternion(x_pert[6:10])
        rotvec = quaternion_residual(q_pred, q_pert)
        H[:, i] = rotvec / eps
    return H


# ─────────────────────────────────────────────
# POSITION LOGGER (EKF mode)
# ─────────────────────────────────────────────

class PositionLogger:
    HEADER = [
        "time", "source", "lat", "lon", "alt",
        "ekf_e", "ekf_n", "ekf_u",
        "ekf_ve", "ekf_vn", "ekf_vu",
        "ekf_yaw_deg", "gps_age_s",
    ]

    def __init__(self, filename="position.csv"):
        self.filename = filename
        self._file = None
        self._writer = None
        self._count = 0

    def _ensure_open(self):
        if self._file is None:
            write_header = (
                not os.path.exists(self.filename)
                or os.path.getsize(self.filename) == 0)
            if not write_header:
                try:
                    with open(self.filename, "r", newline="") as f:
                        first_row = next(csv.reader(f), [])
                    write_header = first_row != self.HEADER
                except Exception:
                    write_header = True
            self._file = open(self.filename, "a", newline="")
            self._writer = csv.writer(self._file)
            if write_header:
                self._writer.writerow(self.HEADER)

    def log_gps(self, gps_fix):
        self._ensure_open()
        self._writer.writerow([
            f"{time.time():.6f}", "gps",
            gps_fix.get("lat"), gps_fix.get("lon"),
            gps_fix.get("alt"),
            "", "", "",
            "", "", "",
            "", "",
        ])
        self._auto_flush()

    def log_ekf(self, ekf):
        gps_pos = ekf.get_gps_position()
        if gps_pos is None:
            return
        pos = ekf.get_position_local()
        vel = ekf.get_velocity()
        gps_age = ekf.gps_age_seconds()
        self._ensure_open()
        self._writer.writerow([
            f"{time.time():.6f}", "ekf",
            f"{gps_pos[0]:.10f}", f"{gps_pos[1]:.10f}",
            f"{gps_pos[2]:.4f}",
            f"{pos[0]:.6f}", f"{pos[1]:.6f}",
            f"{pos[2]:.6f}",
            f"{vel[0]:.6f}", f"{vel[1]:.6f}",
            f"{vel[2]:.6f}",
            f"{ekf.get_yaw_deg():.3f}",
            "" if gps_age is None else f"{gps_age:.6f}",
        ])
        self._auto_flush()

    def _auto_flush(self):
        self._count += 1
        if self._count % 2000 == 0:
            self._file.flush()

    def flush(self):
        if self._file:
            self._file.flush()

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


# ─────────────────────────────────────────────
# IMU-ONLY LOGGER
# ─────────────────────────────────────────────

class IMUOnlyLogger:
    HEADER = [
        "time_s", "imu_ts_us", "dt",
        "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
        "accel_x_g", "accel_y_g", "accel_z_g",
        "gravity_x_g", "gravity_y_g", "gravity_z_g",
        "linear_x_g", "linear_y_g", "linear_z_g",
        "mag_x", "mag_y", "mag_z",
        "quat_w", "quat_x", "quat_y", "quat_z",
        "roll_deg", "pitch_deg", "yaw_deg",
        "vel_e_mps", "vel_n_mps", "vel_u_mps",
        "speed_mps",
    ]

    def __init__(self, filename="IMU_only.csv"):
        self.filename = filename
        self._file = open(filename, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._count = 0

    def write_row(self, wall_time, imu_ts_us, dt, gyro,
                  accel_g, gravity_g, linear_g, mag,
                  quat_wxyz, euler_deg, vel_enu):
        speed_h = math.sqrt(vel_enu[0] ** 2 + vel_enu[1] ** 2)
        self._writer.writerow([
            f"{wall_time:.6f}", imu_ts_us, f"{dt:.6f}",
            f"{gyro[0]:.9f}", f"{gyro[1]:.9f}",
            f"{gyro[2]:.9f}",
            f"{accel_g[0]:.9f}", f"{accel_g[1]:.9f}",
            f"{accel_g[2]:.9f}",
            f"{gravity_g[0]:.9f}", f"{gravity_g[1]:.9f}",
            f"{gravity_g[2]:.9f}",
            f"{linear_g[0]:.9f}", f"{linear_g[1]:.9f}",
            f"{linear_g[2]:.9f}",
            f"{mag[0]:.9f}", f"{mag[1]:.9f}",
            f"{mag[2]:.9f}",
            f"{quat_wxyz[0]:.9f}", f"{quat_wxyz[1]:.9f}",
            f"{quat_wxyz[2]:.9f}", f"{quat_wxyz[3]:.9f}",
            f"{euler_deg[0]:.4f}", f"{euler_deg[1]:.4f}",
            f"{euler_deg[2]:.4f}",
            f"{vel_enu[0]:.6f}", f"{vel_enu[1]:.6f}",
            f"{vel_enu[2]:.6f}",
            f"{speed_h:.6f}",
        ])
        self._count += 1
        if self._count % 500 == 0:
            self._file.flush()

    def flush(self):
        if self._file:
            self._file.flush()

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
            print(f"[IMU-ONLY] Saved {self._count} rows "
                  f"to {self.filename}")

    def __del__(self):
        self.close()


# ─────────────────────────────────────────────
# IMU-ONLY PROCESSOR (no EKF)
# ─────────────────────────────────────────────

class IMUOnlyProcessor:
    def __init__(self):
        self.gravity_g = np.array([0.0, 0.0, -1.0])
        self.quat_scipy = np.array([0.0, 0.0, 0.0, 1.0])
        self.euler_deg = np.array([0.0, 0.0, 0.0])
        self.has_quat = False
        self.has_gravity = False

        self.vel_enu = np.array([0.0, 0.0, 0.0])
        self.prev_linear_world_mps2 = None

        self.last_imu_ts_us = None

        self.raw_count = 0
        self.gravity_count = 0
        self.quat_count = 0
        self.rpy_count = 0

        self.latest_raw = None
        self.latest_rpy = None

    def feed_gravity(self, gx, gy, gz):
        self.gravity_g = np.array([gx, gy, gz])
        self.has_gravity = True
        self.gravity_count += 1

    def feed_quat(self, q1, q2, q3, q4):
        self.quat_scipy = normalize_quaternion(
            np.array([q2, q3, q4, q1], dtype=float))
        r = R.from_quat(self.quat_scipy)
        self.euler_deg = r.as_euler('zyx', degrees=True)
        self.has_quat = True
        self.quat_count += 1

    def feed_rpy(self, roll_deg, pitch_deg, yaw_deg):
        self.latest_rpy = {
            "roll": roll_deg, "pitch": pitch_deg,
            "yaw": yaw_deg}
        self.rpy_count += 1

    def process_raw(self, imu_ts_us, gyro, accel_g, mag):
        self.raw_count += 1

        dt = 0.0
        if self.last_imu_ts_us is not None:
            raw_diff = imu_ts_us - self.last_imu_ts_us
            if raw_diff < 0:
                raw_diff += 2 ** 32
            dt = raw_diff / 1e6
            if dt <= 0 or dt > MAX_VALID_IMU_DT:
                dt = 0.0
        self.last_imu_ts_us = imu_ts_us

        linear_body_g = accel_g - self.gravity_g
        linear_body_mps2 = linear_body_g * GRAVITY_MPS2

        if self.has_quat and dt > 0:
            rot = R.from_quat(self.quat_scipy)
            linear_world_mps2 = rot.apply(linear_body_mps2)

            if self.prev_linear_world_mps2 is not None:
                avg_accel = 0.5 * (
                    self.prev_linear_world_mps2
                    + linear_world_mps2)
                self.vel_enu += avg_accel * dt
            else:
                self.vel_enu += linear_world_mps2 * dt

            self.prev_linear_world_mps2 = (
                linear_world_mps2.copy())

        result = {
            "imu_ts_us": imu_ts_us, "dt": dt,
            "gyro": gyro, "accel_g": accel_g,
            "gravity_g": self.gravity_g.copy(),
            "linear_g": linear_body_g, "mag": mag,
            "quat_wxyz": np.array([
                self.quat_scipy[3], self.quat_scipy[0],
                self.quat_scipy[1], self.quat_scipy[2]]),
            "euler_deg": self.euler_deg.copy(),
            "vel_enu": self.vel_enu.copy(),
        }
        self.latest_raw = result
        return result


# ─────────────────────────────────────────────
# IMU PACKET READER
# ─────────────────────────────────────────────

class IMUPacketReader:
    def __init__(self, ser, check_crc=True):
        self.ser = ser
        self.check_crc = check_crc
        self._buf = bytearray()
        self.crc_fail_count = 0
        self.bad_packet_count = 0

    def _refill(self):
        avail = self.ser.in_waiting
        if avail > 0:
            self._buf.extend(
                self.ser.read(min(avail, 4096)))

    def read_packet(self):
        self._refill()

        while True:
            sync_idx = -1
            for i in range(len(self._buf) - 1):
                if (self._buf[i] == IMU_SYNC_BYTE_1
                        and self._buf[i + 1] == IMU_SYNC_BYTE_2):
                    sync_idx = i
                    break

            if sync_idx < 0:
                if len(self._buf) > 1:
                    self._buf = self._buf[-1:]
                return None, None

            if sync_idx > 0:
                self._buf = self._buf[sync_idx:]

            if len(self._buf) < 3:
                return None, None

            length = self._buf[2]
            total_needed = 3 + length + IMU_CRC_BYTES

            if len(self._buf) < total_needed:
                return None, None

            length_byte = bytes([length])
            payload = bytes(self._buf[3:3 + length])
            crc_bytes = bytes(
                self._buf[3 + length:3 + length + 2])

            self._buf = self._buf[total_needed:]

            if self.check_crc:
                crc_received = struct.unpack(
                    "<H", crc_bytes)[0]
                crc_calculated = crc16_modbus(
                    length_byte + payload)

                if crc_received != crc_calculated:
                    self.crc_fail_count += 1
                    continue

            if len(payload) < 4:
                self.bad_packet_count += 1
                continue

            header = struct.unpack("<I", payload[:4])[0]
            cmd_id = header & 0x7F

            timestamp, data = self._parse_packet(
                cmd_id, payload[4:])
            if timestamp is not None and data is not None:
                return timestamp, data

    @staticmethod
    def _parse_packet(cmd_id, data):
        try:
            if cmd_id == IMU_CMD_RAW:
                if len(data) < 28:
                    return None, None
                num_floats = (len(data) - 4) // 4
                fmt = "<I" + ("f" * num_floats)
                required_bytes = 4 + num_floats * 4
                vals = struct.unpack(
                    fmt, data[:required_bytes])
                result = {
                    "packet_type": "RAW",
                    "gyro_x_rad_s": vals[1],
                    "gyro_y_rad_s": vals[2],
                    "gyro_z_rad_s": vals[3],
                    "accel_x_g": vals[4],
                    "accel_y_g": vals[5],
                    "accel_z_g": vals[6],
                }
                if num_floats >= 9:
                    result["mag_x"] = vals[7]
                    result["mag_y"] = vals[8]
                    result["mag_z"] = vals[9]
                else:
                    result["mag_x"] = math.nan
                    result["mag_y"] = math.nan
                    result["mag_z"] = math.nan
                return vals[0], result

            if cmd_id == IMU_CMD_QUAT:
                if len(data) < 20:
                    return None, None
                vals = struct.unpack("<Iffff", data[:20])
                return vals[0], {
                    "packet_type": "QUAT",
                    "q1": vals[1], "q2": vals[2],
                    "q3": vals[3], "q4": vals[4]}

            if cmd_id == IMU_CMD_RPY:
                if len(data) < 16:
                    return None, None
                vals = struct.unpack("<Ifff", data[:16])
                return vals[0], {
                    "packet_type": "RPY",
                    "roll_deg": vals[1],
                    "pitch_deg": vals[2],
                    "yaw_deg": vals[3]}

            if cmd_id == IMU_CMD_GRAVITY:
                if len(data) < 16:
                    return None, None
                vals = struct.unpack("<Ifff", data[:16])
                return vals[0], {
                    "packet_type": "GRAVITY",
                    "gravity_x_g": vals[1],
                    "gravity_y_g": vals[2],
                    "gravity_z_g": vals[3]}

        except struct.error:
            return None, None
        return None, None


# ─────────────────────────────────────────────
# GPS LINE PARSER (no serial read — just parse)
# ─────────────────────────────────────────────

def parse_gps_line(line, last_valid_raw_gps=None):
    """Parse a single NMEA line string. Returns dict or None."""
    if not line:
        return None

    if not line.startswith("$"):
        return None

    if "*" in line and not valid_nmea_checksum(line):
        return None

    try:
        msg = pynmea2.parse(line)

        if isinstance(msg, pynmea2.types.talker.GGA):
            lat = nmea_to_decimal_degrees(
                msg.lat, msg.lat_dir, is_latitude=True)
            lon = nmea_to_decimal_degrees(
                msg.lon, msg.lon_dir, is_latitude=False)

            alt = None
            if msg.altitude not in (None, ""):
                try:
                    alt = float(msg.altitude)
                except ValueError:
                    pass

            hdop = None
            if msg.horizontal_dil not in (None, ""):
                try:
                    hdop = float(msg.horizontal_dil)
                except ValueError:
                    pass

            fix_quality = None
            if msg.gps_qual not in (None, ""):
                try:
                    fix_quality = int(msg.gps_qual)
                except ValueError:
                    pass

            num_sats = None
            if msg.num_sats not in (None, ""):
                try:
                    num_sats = int(msg.num_sats)
                except ValueError:
                    pass

            if lat is None or lon is None:
                return None

            if (fix_quality is None
                    or fix_quality < MIN_FIX_QUALITY):
                return {
                    "type": "GGA", "valid": False,
                    "lat": lat, "lon": lon, "alt": alt,
                    "hdop": hdop,
                    "fix_quality": fix_quality,
                    "num_sats": num_sats}

            if last_valid_raw_gps is not None:
                try:
                    jump_m = geodesic(
                        (last_valid_raw_gps["lat"],
                         last_valid_raw_gps["lon"]),
                        (lat, lon)).meters
                    if jump_m > MAX_RAW_GPS_JUMP_M:
                        return {
                            "type": "GGA", "valid": False,
                            "lat": lat, "lon": lon,
                            "alt": alt, "hdop": hdop,
                            "fix_quality": fix_quality,
                            "num_sats": num_sats}
                except Exception:
                    pass

            return {
                "type": "GGA", "valid": True,
                "lat": lat, "lon": lon, "alt": alt,
                "hdop": hdop, "fix_quality": fix_quality,
                "num_sats": num_sats}

        if isinstance(msg, pynmea2.types.talker.RMC):
            speed_knots = None
            course_deg = None
            status = getattr(msg, 'status', None)

            if status == 'A':
                if msg.spd_over_grnd not in (None, ""):
                    try:
                        speed_knots = float(
                            msg.spd_over_grnd)
                    except ValueError:
                        pass
                if msg.true_course not in (None, ""):
                    try:
                        course_deg = float(
                            msg.true_course)
                    except ValueError:
                        pass

            speed_mps = None
            if speed_knots is not None:
                speed_mps = speed_knots * 0.514444

            return {
                "type": "RMC", "speed_mps": speed_mps,
                "course_deg": course_deg,
                "valid": status == 'A'}

        if isinstance(msg, pynmea2.types.talker.VTG):
            speed_kmh = None
            course_deg = None

            if hasattr(msg, 'spd_over_grnd_kmph'):
                if msg.spd_over_grnd_kmph not in (None, ""):
                    try:
                        speed_kmh = float(
                            msg.spd_over_grnd_kmph)
                    except ValueError:
                        pass
            if hasattr(msg, 'true_track'):
                if msg.true_track not in (None, ""):
                    try:
                        course_deg = float(msg.true_track)
                    except ValueError:
                        pass

            speed_mps = None
            if speed_kmh is not None:
                speed_mps = speed_kmh / 3.6

            return {
                "type": "VTG", "speed_mps": speed_mps,
                "course_deg": course_deg}

        if isinstance(msg, pynmea2.types.talker.GST):
            lat_err = (
                float(msg.data[5])
                if len(msg.data) > 5 and msg.data[5]
                else None)
            lon_err = (
                float(msg.data[6])
                if len(msg.data) > 6 and msg.data[6]
                else None)
            alt_err = (
                float(msg.data[7])
                if len(msg.data) > 7 and msg.data[7]
                else None)
            return {
                "type": "GST", "lat_err": lat_err,
                "lon_err": lon_err, "alt_err": alt_err}

        if isinstance(msg, pynmea2.types.talker.HDT):
            heading = (float(msg.heading)
                       if msg.heading else None)
            return {"type": "HDT", "heading_deg": heading}

        return None

    except pynmea2.ParseError:
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────
# GPS VELOCITY TRACKER
# ─────────────────────────────────────────────

class GPSVelocityTracker:
    def __init__(self):
        self._prev_gga_time = None
        self._prev_gga_lat = None
        self._prev_gga_lon = None
        self._prev_gga_alt = None
        self._rmc_vtg_speed_mps = None
        self._rmc_vtg_course_deg = None
        self._rmc_vtg_time = None

    def feed_rmc_vtg(self, speed_mps, course_deg):
        if speed_mps is not None:
            self._rmc_vtg_speed_mps = speed_mps
            self._rmc_vtg_course_deg = course_deg
            self._rmc_vtg_time = time.time()

    def feed_gga_and_get_velocity(self, lat, lon, alt,
                                  reference_gps):
        now = time.time()

        if (self._rmc_vtg_speed_mps is not None
                and self._rmc_vtg_time is not None
                and now - self._rmc_vtg_time < 0.5):

            speed = self._rmc_vtg_speed_mps
            course = self._rmc_vtg_course_deg

            if course is not None and speed > 0.3:
                ve, vn = course_speed_to_enu_velocity(
                    course, speed)
                vu = 0.0

                if (self._prev_gga_time is not None
                        and self._prev_gga_alt is not None
                        and alt is not None):
                    dt_gga = now - self._prev_gga_time
                    if (GPS_VEL_FDIFF_MIN_DT < dt_gga
                            < GPS_VEL_FDIFF_MAX_DT):
                        vu = ((alt - self._prev_gga_alt)
                              / dt_gga)

                self._prev_gga_time = now
                self._prev_gga_lat = lat
                self._prev_gga_lon = lon
                self._prev_gga_alt = alt
                self._rmc_vtg_speed_mps = None
                self._rmc_vtg_course_deg = None

                return (np.array([ve, vn, vu]),
                        GPS_VEL_RMC_SIGMA_MPS, "rmc")

            elif speed <= 0.3:
                self._prev_gga_time = now
                self._prev_gga_lat = lat
                self._prev_gga_lon = lon
                self._prev_gga_alt = alt
                self._rmc_vtg_speed_mps = None
                self._rmc_vtg_course_deg = None

                return (np.array([0.0, 0.0, 0.0]),
                        GPS_VEL_RMC_SIGMA_MPS * 2.0, "rmc")

        if (self._prev_gga_time is not None
                and self._prev_gga_lat is not None
                and reference_gps is not None):

            dt_gga = now - self._prev_gga_time

            if (GPS_VEL_FDIFF_MIN_DT < dt_gga
                    < GPS_VEL_FDIFF_MAX_DT):
                try:
                    de = geodesic(
                        (reference_gps[0],
                         self._prev_gga_lon),
                        (reference_gps[0], lon)).meters
                    if lon < self._prev_gga_lon:
                        de = -de

                    dn = geodesic(
                        (self._prev_gga_lat,
                         reference_gps[1]),
                        (lat, reference_gps[1])).meters
                    if lat < self._prev_gga_lat:
                        dn = -dn

                    ve = de / dt_gga
                    vn = dn / dt_gga
                    vu = 0.0
                    if (alt is not None
                            and self._prev_gga_alt is not None):
                        vu = ((alt - self._prev_gga_alt)
                              / dt_gga)

                    self._prev_gga_time = now
                    self._prev_gga_lat = lat
                    self._prev_gga_lon = lon
                    self._prev_gga_alt = alt

                    return (np.array([ve, vn, vu]),
                            GPS_VEL_FDIFF_SIGMA_MPS, "fdiff")
                except Exception:
                    pass

        self._prev_gga_time = now
        self._prev_gga_lat = lat
        self._prev_gga_lon = lon
        self._prev_gga_alt = alt
        return None, None, None


# ─────────────────────────────────────────────
# SENSOR FUSION EKF
# ─────────────────────────────────────────────

class SensorFusionEKF:
    def __init__(self):
        self.dim_x = 16
        self.x = np.zeros(self.dim_x)
        self.x[6:10] = [0.0, 0.0, 0.0, 1.0]
        self.P = np.eye(self.dim_x) * 100.0
        self.R_pos_default = np.eye(3) * (UERE_SIGMA_M ** 2)

        self.reference_gps = None
        self.last_imu_time = None
        self.dt = 1.0 / 400.0
        self.predict_counter = 0
        self.skipped_predicts = 0

        self._init_accel_buffer = []
        self._orientation_initialized = False

        self.last_gps_update_time = None
        self.has_gps = False
        self.has_imu = False

        self.gps_vel_tracker = GPSVelocityTracker()

        self.latest_gps_pos = None
        self.latest_pos_local = None
        self.latest_vel = None
        self.latest_yaw_deg = None
        self.latest_euler_deg = None
        self.latest_gps_update_info = None

    def set_reference_gps(self, lat, lon, alt):
        if alt is None:
            alt = 0.0
        self.reference_gps = (lat, lon, alt)
        self.last_gps_update_time = time.time()
        self.has_gps = True

    def try_initialize_orientation(self, accel_g):
        if self._orientation_initialized:
            return True

        self._init_accel_buffer.append(accel_g.copy())

        if len(self._init_accel_buffer) >= GRAVITY_INIT_SAMPLES:
            accel_avg_g = np.mean(
                self._init_accel_buffer, axis=0)
            accel_avg_mps2 = accel_avg_g * GRAVITY_MPS2

            mag = np.linalg.norm(accel_avg_mps2)
            if mag < 5.0 or mag > 15.0:
                print(f"[INIT] Accel magnitude "
                      f"{mag:.2f} m/s² out of range")
                self.x[6:10] = [0.0, 0.0, 0.0, 1.0]
            else:
                self.x[6:10] = (
                    initialize_orientation_from_accel(
                        accel_avg_mps2))
                r = R.from_quat(self.x[6:10])
                euler = r.as_euler('zyx', degrees=True)
                print(
                    f"[INIT] Orientation from gravity: "
                    f"yaw={euler[0]:.1f} "
                    f"pitch={euler[1]:.1f} "
                    f"roll={euler[2]:.1f}")

            self._orientation_initialized = True
            self.has_imu = True
            return True

        return False

    def predict(self, u, dt):
        F = compute_F_jacobian(self.x, dt, u)
        Q = compute_Q(dt)
        self.x = fx(self.x, dt, u)
        self.x[6:10] = normalize_quaternion(self.x[6:10])
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def _ekf_update(self, z, H, hx, R_mat,
                    residual_fn=None):
        if residual_fn is not None:
            y = residual_fn(z, hx)
        else:
            y = z - hx

        S = H @ self.P @ H.T + R_mat
        try:
            K = self.P @ H.T @ linalg.inv(S)
        except linalg.LinAlgError:
            return False

        self.x = self.x + (K @ y).flatten()
        self.x[6:10] = normalize_quaternion(self.x[6:10])

        I_KH = np.eye(self.dim_x) - K @ H
        self.P = (I_KH @ self.P @ I_KH.T
                  + K @ R_mat @ K.T)
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def update_imu(self, gyro_rad_s, accel_g,
                   timestamp_us):
        if not self._orientation_initialized:
            if not self.try_initialize_orientation(accel_g):
                self.last_imu_time = timestamp_us
                return False

        if self.last_imu_time is None:
            self.last_imu_time = timestamp_us
            return False

        raw_diff = timestamp_us - self.last_imu_time
        if raw_diff < 0:
            raw_diff += 2 ** 32
        dt = raw_diff / 1e6
        self.last_imu_time = timestamp_us

        if dt <= 0 or dt > MAX_VALID_IMU_DT:
            self.skipped_predicts += 1
            return False

        self.dt = dt
        u = np.concatenate([gyro_rad_s, accel_g])
        self.predict(u, dt)
        self.predict_counter += 1

        self.latest_pos_local = self.x[0:3].copy()
        self.latest_vel = self.x[3:6].copy()
        self.latest_yaw_deg = self.get_yaw_deg()
        self.latest_euler_deg = self.get_euler_deg()
        self.latest_gps_pos = self.get_gps_position()

        return True

    def update_gps(self, lat, lon, alt, hdop=None):
        if self.reference_gps is None:
            return False
        if alt is None:
            alt = self.reference_gps[2]

        try:
            e = geodesic(
                (self.reference_gps[0],
                 self.reference_gps[1]),
                (self.reference_gps[0], lon)).meters
            e *= (1.0 if lon >= self.reference_gps[1]
                  else -1.0)

            n = geodesic(
                (self.reference_gps[0],
                 self.reference_gps[1]),
                (lat, self.reference_gps[1])).meters
            n *= (1.0 if lat >= self.reference_gps[0]
                  else -1.0)

            if (abs(e) > MAX_EKF_POSITION_RADIUS_M
                    or abs(n) > MAX_EKF_POSITION_RADIUS_M):
                return False

            u_alt = alt - self.reference_gps[2]

            if hdop is not None and hdop > 0:
                h_sigma = hdop * UERE_SIGMA_M
            else:
                h_sigma = UERE_SIGMA_M

            vel_enu, vel_sigma, vel_source = (
                self.gps_vel_tracker
                .feed_gga_and_get_velocity(
                    lat, lon, alt, self.reference_gps))

            if vel_enu is not None:
                z = np.array([
                    e, n, u_alt,
                    vel_enu[0], vel_enu[1], vel_enu[2]])

                v_h = vel_sigma
                v_v = vel_sigma * GPS_VEL_VERTICAL_SIGMA_SCALE

                R_combined = np.diag([
                    h_sigma ** 2, h_sigma ** 2,
                    (h_sigma * 1.5) ** 2,
                    v_h ** 2, v_h ** 2, v_v ** 2])

                H = H_jacobian_pos_vel(self.x)
                hx = hx_pos_vel(self.x)
                success = self._ekf_update(
                    z, H, hx, R_combined)

                if success:
                    self.last_gps_update_time = time.time()
                    self.latest_gps_update_info = {
                        "vel_source": vel_source,
                        "vel_meas": vel_enu.copy(),
                        "vel_sigma": vel_sigma,
                        "vel_state": self.x[3:6].copy()}
                return success
            else:
                z = np.array([e, n, u_alt], dtype=float)
                R_pos = np.diag([
                    h_sigma ** 2, h_sigma ** 2,
                    (h_sigma * 1.5) ** 2])

                H = H_jacobian_position(self.x)
                hx = hx_position(self.x)
                success = self._ekf_update(
                    z, H, hx, R_pos)

                if success:
                    self.last_gps_update_time = time.time()
                    self.latest_gps_update_info = {
                        "vel_source": None}
                return success

        except Exception as exc:
            return False

    def update_heading(self, heading_deg,
                       heading_sigma_deg=
                       DEFAULT_HEADING_SIGMA_DEG):
        if heading_deg is None:
            return False
        yaw_meas = heading_deg_to_yaw_rad(heading_deg)
        R_heading = np.array([
            [np.deg2rad(heading_sigma_deg) ** 2]])
        H = H_jacobian_heading(self.x)
        hx = hx_heading(self.x)
        return self._ekf_update(
            np.array([yaw_meas]), H, hx, R_heading,
            residual_fn=angle_residual)

    def update_orientation_from_quat(self, q):
        q_meas = normalize_quaternion(q)

        if not self._orientation_initialized:
            self.x[6:10] = q_meas
            self.P[6:10, 6:10] = np.eye(4) * (
                QUAT_MEASUREMENT_SIGMA_RAD ** 2)
            self.P[6:10, :6] = 0.0
            self.P[:6, 6:10] = 0.0
            self.P[6:10, 10:] = 0.0
            self.P[10:, 6:10] = 0.0
            self._orientation_initialized = True
            self.has_imu = True
            r = R.from_quat(q_meas)
            euler = r.as_euler('zyx', degrees=True)
            print(
                f"[INIT] Orientation from QUAT: "
                f"yaw={euler[0]:.1f} "
                f"pitch={euler[1]:.1f} "
                f"roll={euler[2]:.1f}")
            return True

        q_pred = normalize_quaternion(self.x[6:10])
        y = quaternion_residual(q_meas, q_pred)

        H = H_jacobian_quaternion(self.x)
        R_quat = np.eye(3) * (
            QUAT_MEASUREMENT_SIGMA_RAD ** 2)

        S = H @ self.P @ H.T + R_quat
        try:
            K = self.P @ H.T @ linalg.inv(S)
        except linalg.LinAlgError:
            return False

        dx = (K @ y).flatten()
        self.x += dx
        self.x[6:10] = normalize_quaternion(self.x[6:10])

        I_KH = np.eye(self.dim_x) - K @ H
        self.P = (I_KH @ self.P @ I_KH.T
                  + K @ R_quat @ K.T)
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def get_position_local(self):
        return self.x[0:3].copy()

    def get_velocity(self):
        return self.x[3:6].copy()

    def get_yaw_deg(self):
        scipy_yaw_rad = quaternion_to_yaw(self.x[6:10])
        scipy_yaw_deg = np.rad2deg(scipy_yaw_rad)
        return (360.0 - scipy_yaw_deg) % 360.0

    def get_euler_deg(self):
        q = normalize_quaternion(self.x[6:10])
        r = R.from_quat(q)
        return r.as_euler('zyx', degrees=True)

    def get_gps_position(self):
        if self.reference_gps is None:
            return None
        e, n, u = self.x[0:3]
        if (np.linalg.norm([e, n])
                > MAX_EKF_POSITION_RADIUS_M
                or abs(u) > MAX_EKF_ALTITUDE_OFFSET_M):
            return None
        try:
            ref = self.reference_gps
            intermediate = geodesic(
                meters=abs(n)).destination(
                Point(ref[0], ref[1]),
                0.0 if n >= 0 else 180.0)
            dest = geodesic(
                meters=abs(e)).destination(
                Point(intermediate.latitude,
                      intermediate.longitude),
                90.0 if e >= 0 else 270.0)
            return (dest.latitude, dest.longitude,
                    ref[2] + u)
        except Exception:
            return None

    def gps_age_seconds(self):
        if self.last_gps_update_time is None:
            return None
        return time.time() - self.last_gps_update_time

    def get_mode_string(self):
        if self.has_gps and self.has_imu:
            return "GPS+IMU fusion"
        elif self.has_imu:
            return "IMU only (relative)"
        elif self.has_gps:
            return "GPS only (no prediction)"
        else:
            return "no sensors"


# ─────────────────────────────────────────────
# DEVICE VALIDATION
# ─────────────────────────────────────────────

def validate_imu_connection(imu_reader, timeout=2.0):
    print(f"Validating IMU ({timeout}s)...")
    start = time.time()
    packets = 0
    while time.time() - start < timeout:
        ts, data = imu_reader.read_packet()
        if data is not None:
            packets += 1
            if packets >= 5:
                print(f"  IMU OK: {packets} packets")
                return True
        time.sleep(0.0002)
    print(f"  IMU FAILED: {packets} packets")
    return False


def validate_gps_connection(gps_ser, timeout=3.0):
    print(f"Validating GPS ({timeout}s)...")
    start = time.time()
    lines = 0
    while time.time() - start < timeout:
        try:
            raw = gps_ser.readline()
            if raw:
                line = raw.decode(
                    "ascii", errors="ignore").strip()
                if line.startswith("$"):
                    lines += 1
                    if lines >= 3:
                        print(f"  GPS OK: {lines} NMEA lines")
                        return True
        except Exception:
            pass
    print(f"  GPS FAILED: {lines} NMEA lines")
    return False


# ─────────────────────────────────────────────
# SENSOR READER THREADS
# ─────────────────────────────────────────────

# Queue item format: (wall_clock_time, source, data)
# source: "imu" or "gps"
# For IMU: data = (imu_timestamp_us, imu_data_dict)
# For GPS: data = gps_fix_dict (parsed NMEA)

class IMUReaderThread(threading.Thread):
    """
    Dedicated thread for reading IMU packets.
    Reads continuously, puts parsed packets into
    the shared queue with wall-clock arrival time.
    """

    def __init__(self, imu_reader, sensor_queue,
                 stop_event):
        super().__init__(daemon=True)
        self.imu_reader = imu_reader
        self.queue = sensor_queue
        self.stop_event = stop_event
        self.packets_read = 0
        self.queue_full_drops = 0

    def run(self):
        while not self.stop_event.is_set():
            ts, data = self.imu_reader.read_packet()

            if data is None:
                time.sleep(0.0001)
                continue

            self.packets_read += 1
            wall_time = time.time()

            try:
                self.queue.put_nowait(
                    (wall_time, "imu", (ts, data)))
            except queue.Full:
                self.queue_full_drops += 1


class GPSReaderThread(threading.Thread):
    """
    Dedicated thread for reading GPS NMEA lines.
    Reads continuously, parses each line, puts
    parsed results into the shared queue.
    """

    def __init__(self, gps_ser, sensor_queue,
                 stop_event):
        super().__init__(daemon=True)
        self.gps_ser = gps_ser
        self.queue = sensor_queue
        self.stop_event = stop_event
        self.lines_read = 0
        self.queue_full_drops = 0
        self.last_valid_raw_gps = None

    def run(self):
        while not self.stop_event.is_set():
            try:
                raw_bytes = self.gps_ser.readline()
                if not raw_bytes:
                    continue

                line = raw_bytes.decode(
                    "ascii", errors="strict").strip()
            except UnicodeDecodeError:
                continue
            except serial.SerialException:
                if self.stop_event.is_set():
                    break
                continue
            except Exception:
                continue

            if not line:
                continue

            gps_fix = parse_gps_line(
                line,
                last_valid_raw_gps=self.last_valid_raw_gps)

            if gps_fix is None:
                continue

            # Track last valid GGA for jump detection
            if (gps_fix.get("type") == "GGA"
                    and gps_fix.get("valid")):
                self.last_valid_raw_gps = gps_fix.copy()

            self.lines_read += 1
            wall_time = time.time()

            try:
                self.queue.put_nowait(
                    (wall_time, "gps", gps_fix))
            except queue.Full:
                self.queue_full_drops += 1


# ─────────────────────────────────────────────
# IMU-ONLY MAIN LOOP (single thread, no EKF)
# ─────────────────────────────────────────────

def run_imu_only(imu_reader):
    print("=" * 60)
    print("IMU-ONLY MODE — No EKF")
    print("Logging raw sensor data to IMU_only.csv")
    print("=" * 60)

    processor = IMUOnlyProcessor()
    logger = IMUOnlyLogger("IMU_only.csv")

    last_print_time = time.time()
    rate_start_time = time.time()
    last_rate_time = time.time()

    try:
        while True:
            got_data = False

            while True:
                imu_ts, imu_data = imu_reader.read_packet()
                if imu_data is None:
                    break

                got_data = True
                pt = imu_data.get("packet_type")

                if pt == "GRAVITY":
                    processor.feed_gravity(
                        imu_data["gravity_x_g"],
                        imu_data["gravity_y_g"],
                        imu_data["gravity_z_g"])

                elif pt == "QUAT":
                    processor.feed_quat(
                        imu_data["q1"], imu_data["q2"],
                        imu_data["q3"], imu_data["q4"])

                elif pt == "RPY":
                    processor.feed_rpy(
                        imu_data["roll_deg"],
                        imu_data["pitch_deg"],
                        imu_data["yaw_deg"])

                elif pt == "RAW":
                    gyro = np.array([
                        imu_data["gyro_x_rad_s"],
                        imu_data["gyro_y_rad_s"],
                        imu_data["gyro_z_rad_s"]],
                        dtype=float)
                    accel = np.array([
                        imu_data["accel_x_g"],
                        imu_data["accel_y_g"],
                        imu_data["accel_z_g"]],
                        dtype=float)
                    mag = np.array([
                        imu_data.get("mag_x", math.nan),
                        imu_data.get("mag_y", math.nan),
                        imu_data.get("mag_z", math.nan)],
                        dtype=float)

                    result = processor.process_raw(
                        imu_ts, gyro, accel, mag)

                    if result is not None:
                        logger.write_row(
                            wall_time=time.time(),
                            imu_ts_us=result["imu_ts_us"],
                            dt=result["dt"],
                            gyro=result["gyro"],
                            accel_g=result["accel_g"],
                            gravity_g=result["gravity_g"],
                            linear_g=result["linear_g"],
                            mag=result["mag"],
                            quat_wxyz=result["quat_wxyz"],
                            euler_deg=result["euler_deg"],
                            vel_enu=result["vel_enu"])

            now = time.time()
            if now - last_print_time >= PRINT_EVERY:
                last_print_time = now
                r = processor
                if r.latest_raw is not None:
                    raw = r.latest_raw
                    vel = r.vel_enu
                    speed_h = math.sqrt(
                        vel[0] ** 2 + vel[1] ** 2)
                    accel_norm = np.linalg.norm(
                        raw["accel_g"])
                    gravity_norm = np.linalg.norm(
                        r.gravity_g)
                    linear_norm = np.linalg.norm(
                        raw["linear_g"])

                    print("=" * 72)
                    print(f"RAW ts: {raw['imu_ts_us']}")
                    print(f"  Gyro  rad/s: "
                          f"{format_vec(raw['gyro'], 6)}")
                    print(f"  Accel g:     "
                          f"{format_vec(raw['accel_g'], 6)}"
                          f"  norm={accel_norm:.6f}")
                    print(f"  Gravity g:   "
                          f"{format_vec(r.gravity_g, 6)}"
                          f"  norm={gravity_norm:.6f}")
                    print(f"  Linear g:    "
                          f"{format_vec(raw['linear_g'], 6)}"
                          f"  norm={linear_norm:.6f}")
                    print(f"  Mag:         "
                          f"{format_vec(raw['mag'], 4)}")
                    print(f"  Euler: "
                          f"yaw={r.euler_deg[0]:.2f}° "
                          f"pitch={r.euler_deg[1]:.2f}° "
                          f"roll={r.euler_deg[2]:.2f}°")
                    print(f"  Vel ENU m/s: "
                          f"E={vel[0]:.4f} "
                          f"N={vel[1]:.4f} "
                          f"U={vel[2]:.4f} | "
                          f"speed={speed_h:.4f}")

                    if r.latest_rpy is not None:
                        rpy = r.latest_rpy
                        print(f"  RPY official: "
                              f"roll={rpy['roll']:.2f}° "
                              f"pitch={rpy['pitch']:.2f}° "
                              f"yaw={rpy['yaw']:.2f}°")

                if now - last_rate_time >= 2.0:
                    elapsed = now - rate_start_time
                    if elapsed > 0:
                        print(
                            f"  RATE: "
                            f"RAW="
                            f"{r.raw_count / elapsed:.0f} Hz"
                            f" | GRAV="
                            f"{r.gravity_count / elapsed:.0f}"
                            f" Hz | QUAT="
                            f"{r.quat_count / elapsed:.0f}"
                            f" Hz | RPY="
                            f"{r.rpy_count / elapsed:.0f}"
                            f" Hz | CRC_fail="
                            f"{imu_reader.crc_fail_count}"
                            f" | Logged={logger._count}")
                    last_rate_time = now

            if not got_data:
                time.sleep(0.0002)

    except KeyboardInterrupt:
        print("\nStopping IMU-only mode...")
    finally:
        logger.close()


# ─────────────────────────────────────────────
# EKF FUSION MAIN LOOP (threaded readers)
# ─────────────────────────────────────────────

def run_ekf_fusion(imu_reader, imu_ser, gps_ser):
    ekf = SensorFusionEKF()
    pos_logger = PositionLogger("position.csv")

    last_gps_error = None
    last_heading_deg = None

    # ── Shared queue and stop event ──
    sensor_queue = queue.Queue(
        maxsize=SENSOR_QUEUE_MAXSIZE)
    stop_event = threading.Event()

    imu_thread = None
    gps_thread = None

    # ── Phase 1: Wait for GPS fix (single-threaded) ──
    # Before starting threads, do the GPS wait phase
    # so we have a reference position for ENU conversion.
    if gps_ser is not None:
        print(f"Waiting for GPS fix ({GPS_FIX_TIMEOUT_S}s)...")
        start_time = time.time()
        last_valid_raw_gps = None

        try:
            while (ekf.reference_gps is None
                   and time.time() - start_time
                   < GPS_FIX_TIMEOUT_S):

                # Read GPS
                try:
                    raw_bytes = gps_ser.readline()
                    if raw_bytes:
                        line = raw_bytes.decode(
                            "ascii",
                            errors="strict").strip()
                        gps_fix = parse_gps_line(
                            line,
                            last_valid_raw_gps=(
                                last_valid_raw_gps))

                        if gps_fix is not None:
                            mt = gps_fix.get("type")

                            if mt == "RMC":
                                if gps_fix.get("valid"):
                                    ekf.gps_vel_tracker.feed_rmc_vtg(gps_fix.get("speed_mps"), gps_fix.get("course_deg"))
                                      

                            elif mt == "VTG":
                                ekf.gps_vel_tracker.feed_rmc_vtg(
                                    gps_fix.get(
                                        "speed_mps"),
                                    gps_fix.get(
                                        "course_deg"))

                            elif (mt == "GGA"
                                  and gps_fix.get("valid")):
                                last_valid_raw_gps = (
                                    gps_fix.copy())
                                lat = gps_fix["lat"]
                                lon = gps_fix["lon"]
                                alt = gps_fix.get("alt")

                                if (lat and lon
                                        and lat != 0.0
                                        and lon != 0.0):
                                    ekf.set_reference_gps(
                                        lat, lon, alt)
                                    print(
                                        f"Reference GPS set:"
                                        f" {ekf.reference_gps}"
                                    )
                except Exception:
                    pass

                # Drain IMU during wait
                if imu_reader is not None:
                    for _ in range(50):
                        ts, data = (
                            imu_reader.read_packet())
                        if data is None:
                            break
                        if data.get(
                                "packet_type") == "QUAT":
                            q = np.array([
                                data["q2"], data["q3"],
                                data["q4"], data["q1"]],
                                dtype=float)
                            ekf.update_orientation_from_quat(
                                q)

                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\nStopped during GPS wait.")
            pos_logger.close()
            return

        if ekf.reference_gps is None:
            print("[WARNING] No GPS fix obtained.")
    else:
        print("GPS disabled.")

    if imu_reader is None:
        print("IMU disabled. GPS-only mode.")

    print(f"Mode: {ekf.get_mode_string()}")
    print("Starting threaded fusion loop...")

    # ── Start reader threads ──
    if imu_reader is not None:
        imu_thread = IMUReaderThread(
            imu_reader, sensor_queue, stop_event)
        imu_thread.start()
        print("[THREAD] IMU reader started")

    if gps_ser is not None:
        gps_thread = GPSReaderThread(
            gps_ser, sensor_queue, stop_event)
        gps_thread.start()
        print("[THREAD] GPS reader started")

    # ── Phase 2: Main processing loop ──
    # Dequeue items in arrival order and process.
    # Items arrive naturally interleaved by wall clock:
    #   imu, imu,..., imu, gps, imu, imu,..., imu, gps
    # because IMU thread produces ~40x faster than GPS.

    imu_predictions_total = 0
    gps_updates_total = 0
    gps_vel_total = 0

    rate_start_time = time.time()
    last_print_time = time.time()
    last_rate_time = time.time()
    last_gps_print_info = None

    try:
        while True:
            # ── Get next sensor item ──
            try:
                wall_time, source, data = (
                    sensor_queue.get(timeout=0.01))
            except queue.Empty:
                # Nothing available — check print timer
                now = time.time()
                if now - last_print_time >= PRINT_EVERY:
                    last_print_time = now
                    _print_ekf_status(
                        ekf, last_gps_print_info,
                        imu_thread, gps_thread,
                        imu_predictions_total,
                        gps_updates_total,
                        gps_vel_total,
                        rate_start_time, last_rate_time,
                        pos_logger)
                    last_gps_print_info = None
                    if now - last_rate_time >= 2.0:
                        last_rate_time = now

                # GPS timeout inflation
                gps_age = ekf.gps_age_seconds()
                if (gps_age is not None
                        and gps_age > 5.0):
                    inflate = min(gps_age * 0.5, 50.0)
                    ekf.P[0:3, 0:3] += (
                        np.eye(3) * inflate * ekf.dt)
                continue

            # ── Process IMU ──
            if source == "imu":
                imu_ts, imu_data = data
                pt = imu_data.get("packet_type")

                if pt == "RAW":
                    gyro = np.array([
                        imu_data["gyro_x_rad_s"],
                        imu_data["gyro_y_rad_s"],
                        imu_data["gyro_z_rad_s"]],
                        dtype=float)
                    accel = np.array([
                        imu_data["accel_x_g"],
                        imu_data["accel_y_g"],
                        imu_data["accel_z_g"]],
                        dtype=float)

                    predicted = ekf.update_imu(
                        gyro, accel, imu_ts)

                    if predicted:
                        imu_predictions_total += 1
                        pos_logger.log_ekf(ekf)

                elif pt == "QUAT":
                    q = np.array([
                        imu_data["q2"], imu_data["q3"],
                        imu_data["q4"], imu_data["q1"]],
                        dtype=float)
                    ekf.update_orientation_from_quat(q)

            # ── Process GPS ──
            elif source == "gps":
                gps_fix = data
                msg_type = gps_fix.get("type")

                if msg_type == "GGA":
                    if gps_fix.get("valid"):
                        lat = gps_fix["lat"]
                        lon = gps_fix["lon"]
                        alt = gps_fix.get("alt")

                        pos_logger.log_gps(gps_fix)

                        if ekf.reference_gps is None:
                            if (lat and lon
                                    and lat != 0.0
                                    and lon != 0.0):
                                ekf.set_reference_gps(
                                    lat, lon, alt)
                                print(
                                    f"[LATE GPS INIT] "
                                    f"Ref: "
                                    f"{ekf.reference_gps}")

                        if ekf.reference_gps is not None:
                            gps_updated = ekf.update_gps(
                                lat, lon, alt,
                                hdop=gps_fix.get("hdop"))

                            if gps_updated:
                                gps_updates_total += 1
                                last_gps_print_info = {
                                    "gps_pos":
                                        ekf.get_gps_position(),
                                    "vel":
                                        ekf.get_velocity(),
                                    "yaw":
                                        ekf.get_yaw_deg(),
                                    "update_info":
                                        ekf.latest_gps_update_info,
                                }

                elif msg_type == "RMC":
                    if gps_fix.get("valid"):
                        ekf.gps_vel_tracker.feed_rmc_vtg(
                            gps_fix.get("speed_mps"),
                            gps_fix.get("course_deg"))
                        gps_vel_total += 1

                elif msg_type == "VTG":
                    ekf.gps_vel_tracker.feed_rmc_vtg(
                        gps_fix.get("speed_mps"),
                        gps_fix.get("course_deg"))
                    gps_vel_total += 1

                elif msg_type == "GST":
                    last_gps_error = (
                        gps_fix.get("lat_err"),
                        gps_fix.get("lon_err"),
                        gps_fix.get("alt_err"))

                elif msg_type == "HDT":
                    last_heading_deg = gps_fix.get(
                        "heading_deg")
                    if last_heading_deg is not None:
                        ekf.update_heading(
                            last_heading_deg,
                            heading_sigma_deg=(
                                DEFAULT_HEADING_SIGMA_DEG))

            # ── GPS timeout inflation ──
            gps_age = ekf.gps_age_seconds()
            if gps_age is not None and gps_age > 5.0:
                inflate = min(gps_age * 0.5, 50.0)
                ekf.P[0:3, 0:3] += (
                    np.eye(3) * inflate * ekf.dt)

            # ── Periodic display ──
            now = time.time()
            if now - last_print_time >= PRINT_EVERY:
                last_print_time = now
                _print_ekf_status(
                    ekf, last_gps_print_info,
                    imu_thread, gps_thread,
                    imu_predictions_total,
                    gps_updates_total,
                    gps_vel_total,
                    rate_start_time, last_rate_time,
                    pos_logger)
                last_gps_print_info = None
                if now - last_rate_time >= 2.0:
                    last_rate_time = now

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()

        if imu_thread is not None:
            imu_thread.join(timeout=1.0)
            print(f"[THREAD] IMU: "
                  f"{imu_thread.packets_read} packets, "
                  f"{imu_thread.queue_full_drops} drops")

        if gps_thread is not None:
            gps_thread.join(timeout=1.0)
            print(f"[THREAD] GPS: "
                  f"{gps_thread.lines_read} lines, "
                  f"{gps_thread.queue_full_drops} drops")

        pos_logger.close()
        print(f"Closed. Mode: {ekf.get_mode_string()}")
        if imu_reader is not None:
            print(f"CRC failures: "
                  f"{imu_reader.crc_fail_count}")
            print(f"Bad packets: "
                  f"{imu_reader.bad_packet_count}")


def _print_ekf_status(ekf, last_gps_print_info,
                      imu_thread, gps_thread,
                      imu_predictions_total,
                      gps_updates_total,
                      gps_vel_total,
                      rate_start_time, last_rate_time,
                      pos_logger):
    """Print EKF status — called on timer from main loop."""
    now = time.time()
    print("=" * 72)

    # GPS update info
    if last_gps_print_info is not None:
        info = last_gps_print_info
        gps_pos = info["gps_pos"]
        ui = info.get("update_info")

        if gps_pos is not None:
            line = (
                f"[GPS] lat={gps_pos[0]:.8f} "
                f"lon={gps_pos[1]:.8f} "
                f"alt={gps_pos[2]:.2f} "
                f"yaw={info['yaw']:.1f}°")
            if ui and ui.get("vel_source"):
                vs = ui["vel_source"]
                vm = ui["vel_meas"]
                line += (
                    f" | vel_{vs}=("
                    f"{vm[0]:.2f},{vm[1]:.2f},"
                    f"{vm[2]:.2f})")
            print(line)

    # EKF state
    if ekf.latest_pos_local is not None:
        pos = ekf.latest_pos_local
        vel = ekf.latest_vel
        yaw = ekf.latest_yaw_deg
        gps_pos = ekf.latest_gps_pos

        if gps_pos is not None:
            print(
                f"[EKF] lat={gps_pos[0]:.8f} "
                f"lon={gps_pos[1]:.8f} "
                f"alt={gps_pos[2]:.2f} "
                f"yaw={yaw:.1f}° | "
                f"local=({pos[0]:.2f},{pos[1]:.2f},"
                f"{pos[2]:.2f}) "
                f"vel=({vel[0]:.2f},{vel[1]:.2f},"
                f"{vel[2]:.2f})")
        else:
            euler = ekf.latest_euler_deg
            print(
                f"[EKF] pos=({pos[0]:.3f},"
                f"{pos[1]:.3f},{pos[2]:.3f}) m | "
                f"vel=({vel[0]:.3f},{vel[1]:.3f},"
                f"{vel[2]:.3f}) m/s | "
                f"yaw={yaw:.1f}° "
                f"pitch={euler[1]:.1f}° "
                f"roll={euler[2]:.1f}°")

    # Rate
    if now - last_rate_time >= 2.0:
        elapsed = now - rate_start_time
        mode = ekf.get_mode_string()
        parts = [f"RATE mode={mode}"]

        if imu_thread is not None:
            parts.append(
                f"imu_read="
                f"{imu_thread.packets_read / elapsed:.0f}"
                f"/s")
            parts.append(
                f"pred="
                f"{imu_predictions_total / elapsed:.0f}"
                f"/s")
            parts.append(
                f"skip={ekf.skipped_predicts}")
            parts.append(f"dt={ekf.dt:.6f}")
            parts.append(
                f"crc={imu_thread.imu_reader.crc_fail_count}")
            if imu_thread.queue_full_drops > 0:
                parts.append(
                    f"imu_drops="
                    f"{imu_thread.queue_full_drops}")

        if gps_thread is not None:
            parts.append(
                f"gps_read="
                f"{gps_thread.lines_read / elapsed:.0f}"
                f"/s")
            parts.append(
                f"gps_upd="
                f"{gps_updates_total / elapsed:.0f}"
                f"/s")
            parts.append(
                f"vel_feed="
                f"{gps_vel_total / elapsed:.0f}/s")
            if gps_thread.queue_full_drops > 0:
                parts.append(
                    f"gps_drops="
                    f"{gps_thread.queue_full_drops}")

        qsize = 0
        try:
            qsize = imu_thread.queue.qsize() if imu_thread else 0
        except Exception:
            pass
        parts.append(f"q={qsize}")
        parts.append(f"logged={pos_logger._count}")

        print("  " + " | ".join(parts))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    imu_ser = None
    gps_ser = None
    imu_reader = None

    if IMU_ENABLED:
        try:
            imu_ser = serial.Serial(
                IMU_PORT, IMU_BAUD, timeout=0)
            imu_reader = IMUPacketReader(
                imu_ser,
                check_crc=not _args.skip_crc)
            print(f"Opened IMU: {IMU_PORT} @ {IMU_BAUD}")
            if not validate_imu_connection(imu_reader):
                print("[WARNING] IMU not responding.")
                imu_ser.close()
                imu_ser = None
                imu_reader = None
        except serial.SerialException as exc:
            print(f"[WARNING] Could not open IMU: {exc}")

    if GPS_ENABLED:
        try:
            gps_ser = serial.Serial(
                GPS_PORT, GPS_BAUD, timeout=0.1)
            print(f"Opened GPS: {GPS_PORT} @ {GPS_BAUD}")
            if not validate_gps_connection(gps_ser):
                print("[WARNING] GPS not responding.")
                gps_ser.close()
                gps_ser = None
        except serial.SerialException as exc:
            print(f"[WARNING] Could not open GPS: {exc}")

    if imu_ser is None and gps_ser is None:
        print("[ERROR] No sensors available. Exiting.")
        return

    # ── Route to correct mode ──
    if imu_reader is not None and gps_ser is None:
        print("=" * 60)
        print("MODE: IMU-ONLY (no GPS)")
        print("  No EKF. Raw data → IMU_only.csv")
        print("=" * 60)
        try:
            run_imu_only(imu_reader)
        finally:
            imu_ser.close()
            print(f"CRC failures: "
                  f"{imu_reader.crc_fail_count}")
            print(f"Bad packets: "
                  f"{imu_reader.bad_packet_count}")

    else:
        print("=" * 60)
        if imu_reader is not None:
            print("MODE: GPS+IMU EKF FUSION (threaded)")
        else:
            print("MODE: GPS-ONLY (no IMU)")
        print("  EKF active → position.csv")
        print("=" * 60)
        try:
            run_ekf_fusion(imu_reader, imu_ser, gps_ser)
        finally:
            if imu_ser is not None:
                imu_ser.close()
            if gps_ser is not None:
                gps_ser.close()
            print("Serial ports closed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nForce stopped.")
