# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:22:08 2026

@author: Desktop
"""

"""
GPS/IMU Sensor Fusion using Extended Kalman Filter
===================================================
Hardware:
  - SYD Dynamics TransducerM TM151 AHRS 9-Axis IMU at 400 Hz
    Binary EasyProfile protocol
    Accel: gravity-free linear acceleration (near zero when stationary)
    Gyro: rad/s, Timestamp: microseconds
  - GPS at ~5-10 Hz (NMEA: GGA, GST, HDT)

Logging:
  - position.csv: every GPS fix and every EKF prediction
    columns: time, source (gps/ekf), lat, lon, alt

Usage:
  python fusion.py                          # auto-detect ports
  python fusion.py --list-ports             # show available ports
  python fusion.py --imu-port COM5          # override IMU port
  python fusion.py --gps-port /dev/ttyUSB0  # override GPS port
  python fusion.py --no-gps                 # IMU-only mode
  python fusion.py --no-imu                 # GPS-only mode
"""

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
        print(f"[WARNING] Unknown platform '{sys.platform}', using Linux defaults")
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
                    f"{p.description}"
                )
            else:
                print(
                    f"  {p.device:20s} | "
                    f"VID:PID=None         | "
                    f"{p.description}"
                )
    except Exception as exc:
        print(f"  Error listing ports: {exc}")


def find_septentrio_nmea_port(timeout=3.0):
    """
    Septentrio receivers expose 2 COM ports with same VID:PID.
    Try both and return the one that sends NMEA data.
    """
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

    print(f"Found {len(septentrio_ports)} Septentrio ports: {septentrio_ports}")

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
                        line = raw.decode("ascii", errors="ignore").strip()
                        if line.startswith("$") and (
                                "GGA" in line or "GST" in line
                                or "HDT" in line or "RMC" in line
                                or "GSV" in line):
                            nmea_count += 1
                            if nmea_count >= 3:
                                ser.close()
                                print(f"  -> {port} is NMEA port ({nmea_count} sentences)")
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
    IMU_USB_IDS = [
        (0x0483, 0x5740),
    ]

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
                f"{p.description}"
            )
        else:
            print(
                f"  {p.device:20s} | "
                f"VID:PID=None         | "
                f"{p.description}"
            )
            continue

        for known_vid, known_pid in IMU_USB_IDS:
            if vid == known_vid and pid == known_pid:
                if imu_port is None:
                    imu_port = p.device
                    print(f"  -> Detected IMU: {p.device}")

    # Septentrio special handling (2 ports, find NMEA one)
    gps_port = find_septentrio_nmea_port(timeout=3.0)

    # If not Septentrio, try other GPS receivers
    if gps_port is None:
        OTHER_GPS_IDS = [
            (0x1546, 0x01A8),
            (0x10C4, 0xEA60),
            (0x067B, 0x2303),
            (0x0403, 0x6001),
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
        description="GPS/IMU Sensor Fusion EKF"
    )
    parser.add_argument("--imu-port", type=str, default=None)
    parser.add_argument("--gps-port", type=str, default=None)
    parser.add_argument("--imu-baud", type=int, default=None)
    parser.add_argument("--gps-baud", type=int, default=None)
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--no-gps", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
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

IMU_PORT, GPS_PORT = resolve_ports(OS_NAME, _args)

# ─── IMU settings ───
ACCEL_INPUT_IS_G = False
IMU_ACCEL_IS_GRAVITY_FREE = True
GRAVITY_MPS2 = 9.81
IMU_BAUD = _args.imu_baud if _args.imu_baud else 115200

# ─── GPS settings ───
GPS_BAUD = _args.gps_baud if _args.gps_baud else 115200
UERE_SIGMA_M = 4.0
MIN_FIX_QUALITY = 1
MAX_RAW_GPS_JUMP_M = 15.0

# ─── EKF safety ───
MAX_EKF_POSITION_RADIUS_M = 1000.0
MAX_EKF_ALTITUDE_OFFSET_M = 200.0
MAX_VALID_IMU_DT = 0.005

# ─── Debug ───
PRINT_PREDICT_DEBUG = True
PRINT_PREDICT_EVERY_N = 400
PRINT_RAW_GPS_DEBUG = True
PRINT_ALL_NMEA_LINES = False

# ─── Process noise ───
ACCEL_NOISE_DENSITY = 0.5
GYRO_NOISE_DENSITY = 0.01
ACCEL_BIAS_RANDOM_WALK = 0.001
GYRO_BIAS_RANDOM_WALK = 0.0001
POSITION_PROCESS_NOISE = 0.001

# ─── IMU protocol ───
IMU_SYNC_BYTE_1 = 0xAA
IMU_SYNC_BYTE_2 = 0x55
IMU_CMD_RAW = 41
IMU_CMD_QUAT = 32
IMU_CMD_RPY = 35
IMU_CRC_BYTES = 2

# ─── Heading ───
DEFAULT_HEADING_SIGMA_DEG = 5.0

# ─── Gravity alignment ───
GRAVITY_INIT_SAMPLES = 200

# ─── GPS timeout ───
GPS_FIX_TIMEOUT_S = 30

print(f"OS: {OS_NAME}")
print(f"IMU: {'ENABLED' if IMU_ENABLED else 'DISABLED'} | {IMU_PORT} @ {IMU_BAUD}")
print(f"GPS: {'ENABLED' if GPS_ENABLED else 'DISABLED'} | {GPS_PORT} @ {GPS_BAUD}")


# ─────────────────────────────────────────────
# QUATERNION & ANGLE UTILITIES
# ─────────────────────────────────────────────

def normalize_quaternion(q):
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
    return np.array([normalize_angle(float(z[0]) - float(z_pred[0]))])


def initialize_orientation_from_accel(accel_avg_mps2):
    ax, ay, az = accel_avg_mps2
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))
    r = R.from_euler('zyx', [0.0, pitch, roll])
    return r.as_quat()


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
    Q[10:13, 10:13] = np.eye(3) * (GYRO_BIAS_RANDOM_WALK ** 2) * dt
    Q[13:16, 13:16] = np.eye(3) * (ACCEL_BIAS_RANDOM_WALK ** 2) * dt
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
        H[0, idx] = normalize_angle(perturbed_yaw - base_yaw) / eps
    return H


# ─────────────────────────────────────────────
# POSITION LOGGER
# ─────────────────────────────────────────────

class PositionLogger:
    """
    Logs only position: time, source (gps/ekf), lat, lon, alt.
    GPS rows at ~10 Hz, EKF rows at ~400 Hz.
    """

    def __init__(self, filename="position.csv"):
        self.filename = filename
        self._file = None
        self._writer = None
        self._count = 0

    def _ensure_open(self):
        if self._file is None:
            write_header = (not os.path.exists(self.filename)
                           or os.path.getsize(self.filename) == 0)
            self._file = open(self.filename, "a", newline="")
            self._writer = csv.writer(self._file)
            if write_header:
                self._writer.writerow([
                    "time", "source", "lat", "lon", "alt",
                ])

    def log_gps(self, gps_fix):
        self._ensure_open()
        self._writer.writerow([
            f"{time.time():.6f}",
            "gps",
            gps_fix.get("lat"),
            gps_fix.get("lon"),
            gps_fix.get("alt"),
        ])
        self._auto_flush()

    def log_ekf(self, ekf):
        gps_pos = ekf.get_gps_position()
        if gps_pos is None:
            return
        self._ensure_open()
        self._writer.writerow([
            f"{time.time():.6f}",
            "ekf",
            f"{gps_pos[0]:.10f}",
            f"{gps_pos[1]:.10f}",
            f"{gps_pos[2]:.4f}",
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


# Global logger
pos_logger = PositionLogger("position.csv")


# ─────────────────────────────────────────────
# IMU PACKET READER
# ─────────────────────────────────────────────

class IMUPacketReader:
    def __init__(self, ser):
        self.ser = ser
        self._buf = bytearray()

    def _refill(self):
        avail = self.ser.in_waiting
        if avail > 0:
            self._buf.extend(self.ser.read(min(avail, 4096)))

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

            payload = bytes(self._buf[3: 3 + length])
            self._buf = self._buf[total_needed:]

            if len(payload) < 4:
                continue

            header = struct.unpack("<I", payload[:4])[0]
            cmd_id = header & 0x7F

            timestamp, data = self._parse_packet(cmd_id, payload[4:])
            if timestamp is not None and data is not None:
                return timestamp, data

    @staticmethod
    def _parse_packet(cmd_id, data):
        try:
            if cmd_id == IMU_CMD_RAW:
                data_len = len(data)
                if data_len < 28:
                    return None, None
                num_floats = (data_len - 4) // 4
                fmt = "<I" + ("f" * num_floats)
                required_bytes = 4 + (num_floats * 4)
                vals = struct.unpack(fmt, data[:required_bytes])
                result = {
                    "packet_type": "RAW",
                    "accel_x": vals[1], "accel_y": vals[2],
                    "accel_z": vals[3],
                    "gyro_x": vals[4], "gyro_y": vals[5],
                    "gyro_z": vals[6],
                }
                if num_floats >= 9:
                    result["mag_x"] = vals[7]
                    result["mag_y"] = vals[8]
                    result["mag_z"] = vals[9]
                return vals[0], result

            if cmd_id == IMU_CMD_QUAT:
                if len(data) < 20:
                    return None, None
                vals = struct.unpack("<Iffff", data[:20])
                return vals[0], {
                    "packet_type": "QUAT",
                    "q1": vals[1], "q2": vals[2],
                    "q3": vals[3], "q4": vals[4],
                }

            if cmd_id == IMU_CMD_RPY:
                if len(data) < 16:
                    return None, None
                vals = struct.unpack("<Ifff", data[:16])
                return vals[0], {
                    "packet_type": "RPY",
                    "roll": vals[1], "pitch": vals[2],
                    "yaw": vals[3],
                }

        except struct.error:
            pass
        return None, None


# ─────────────────────────────────────────────
# GPS LINE READER
# ─────────────────────────────────────────────

def read_gps_line(ser, last_valid_raw_gps=None):
    try:
        raw_bytes = ser.readline()
        if not raw_bytes:
            return None
        line = raw_bytes.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError:
        return None
    except (KeyboardInterrupt, SystemExit):
        raise

    if not line:
        return None

    if PRINT_ALL_NMEA_LINES:
        print(f"[NMEA] {line}")

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

            if PRINT_RAW_GPS_DEBUG:
                print(
                    f"[GGA] lat={lat} lon={lon} alt={alt} "
                    f"fix={fix_quality} sats={num_sats} hdop={hdop}"
                )

            if lat is None or lon is None:
                return None

            if fix_quality is None or fix_quality < MIN_FIX_QUALITY:
                return {
                    "type": "GGA", "valid": False,
                    "lat": lat, "lon": lon, "alt": alt,
                    "hdop": hdop, "fix_quality": fix_quality,
                    "num_sats": num_sats,
                }

            if last_valid_raw_gps is not None:
                try:
                    jump_m = geodesic(
                        (last_valid_raw_gps["lat"],
                         last_valid_raw_gps["lon"]),
                        (lat, lon)
                    ).meters
                    if jump_m > MAX_RAW_GPS_JUMP_M:
                        print(f"[GGA REJECTED] Jump {jump_m:.1f} m")
                        return {
                            "type": "GGA", "valid": False,
                            "lat": lat, "lon": lon, "alt": alt,
                            "hdop": hdop, "fix_quality": fix_quality,
                            "num_sats": num_sats,
                        }
                except Exception:
                    pass

            return {
                "type": "GGA", "valid": True,
                "lat": lat, "lon": lon, "alt": alt,
                "hdop": hdop, "fix_quality": fix_quality,
                "num_sats": num_sats,
            }

        if isinstance(msg, pynmea2.types.talker.GST):
            lat_err = (float(msg.data[5])
                       if len(msg.data) > 5 and msg.data[5] else None)
            lon_err = (float(msg.data[6])
                       if len(msg.data) > 6 and msg.data[6] else None)
            alt_err = (float(msg.data[7])
                       if len(msg.data) > 7 and msg.data[7] else None)
            return {
                "type": "GST",
                "lat_err": lat_err, "lon_err": lon_err,
                "alt_err": alt_err,
            }

        if isinstance(msg, pynmea2.types.talker.HDT):
            heading = float(msg.heading) if msg.heading else None
            return {"type": "HDT", "heading_deg": heading}

        return None

    except pynmea2.ParseError:
        return None
    except Exception:
        return None


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

    def set_reference_gps(self, lat, lon, alt):
        if alt is None:
            alt = 0.0
        self.reference_gps = (lat, lon, alt)
        self.last_gps_update_time = time.time()
        self.has_gps = True

    def try_initialize_orientation(self, accel):
        if self._orientation_initialized:
            return True

        if IMU_ACCEL_IS_GRAVITY_FREE:
            self.x[6:10] = [0.0, 0.0, 0.0, 1.0]
            self._orientation_initialized = True
            self.has_imu = True
            print("[INIT] Gravity-free accel. Identity orientation.")
            return True

        self._init_accel_buffer.append(accel.copy())
        buf_len = len(self._init_accel_buffer)

        if buf_len >= GRAVITY_INIT_SAMPLES:
            accel_avg = np.mean(self._init_accel_buffer, axis=0)
            if ACCEL_INPUT_IS_G:
                accel_avg_mps2 = accel_avg * GRAVITY_MPS2
            else:
                accel_avg_mps2 = accel_avg

            mag = np.linalg.norm(accel_avg_mps2)
            if mag < 5.0 or mag > 15.0:
                self.x[6:10] = [0.0, 0.0, 0.0, 1.0]
            else:
                self.x[6:10] = initialize_orientation_from_accel(accel_avg_mps2)

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

    def _ekf_update(self, z, H, hx, R_mat, residual_fn=None):
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
        self.P = I_KH @ self.P @ I_KH.T + K @ R_mat @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def update_imu(self, gyro, accel, timestamp_us):
        if not self._orientation_initialized:
            self.try_initialize_orientation(accel)
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
        u = np.concatenate([gyro, accel])
        self.predict(u, dt)
        self.predict_counter += 1

        pos = self.x[0:3]
        if (np.linalg.norm(pos[:2]) > MAX_EKF_POSITION_RADIUS_M
                or abs(pos[2]) > MAX_EKF_ALTITUDE_OFFSET_M):
            print(
                f"[STATE WARNING] EKF diverging: "
                f"pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            )

        if (PRINT_PREDICT_DEBUG
                and self.predict_counter % PRINT_PREDICT_EVERY_N == 0):
            vel = self.x[3:6]
            print(
                f"[PREDICT] dt={self.dt:.6f} "
                f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                f"vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})"
            )

        return True

    def update_gps(self, lat, lon, alt, hdop=None):
        if self.reference_gps is None:
            return False
        if alt is None:
            alt = self.reference_gps[2]

        try:
            e = geodesic(
                (self.reference_gps[0], self.reference_gps[1]),
                (self.reference_gps[0], lon)
            ).meters
            e *= 1.0 if lon >= self.reference_gps[1] else -1.0

            n = geodesic(
                (self.reference_gps[0], self.reference_gps[1]),
                (lat, self.reference_gps[1])
            ).meters
            n *= 1.0 if lat >= self.reference_gps[0] else -1.0

            if (abs(e) > MAX_EKF_POSITION_RADIUS_M
                    or abs(n) > MAX_EKF_POSITION_RADIUS_M):
                return False

            u_alt = alt - self.reference_gps[2]
            z = np.array([e, n, u_alt], dtype=float)

            if hdop is not None and hdop > 0:
                h_sigma = hdop * UERE_SIGMA_M
                R_pos = np.diag([
                    h_sigma ** 2,
                    h_sigma ** 2,
                    (h_sigma * 1.5) ** 2
                ])
            else:
                R_pos = self.R_pos_default

            H = H_jacobian_position(self.x)
            hx = hx_position(self.x)
            success = self._ekf_update(z, H, hx, R_pos)

            if success:
                self.last_gps_update_time = time.time()
            return success

        except Exception as exc:
            print(f"[GPS UPDATE ERROR] {exc}")
            return False

    def update_heading(self, heading_deg,
                       heading_sigma_deg=DEFAULT_HEADING_SIGMA_DEG):
        if heading_deg is None:
            return False
        yaw_meas = heading_deg_to_yaw_rad(heading_deg)
        R_heading = np.array([[np.deg2rad(heading_sigma_deg) ** 2]])
        H = H_jacobian_heading(self.x)
        hx = hx_heading(self.x)
        return self._ekf_update(
            np.array([yaw_meas]), H, hx, R_heading,
            residual_fn=angle_residual
        )

    def update_orientation_from_quat(self, q):
        q = normalize_quaternion(q)
        self.x[6:10] = q
        if not self._orientation_initialized:
            self._orientation_initialized = True
            self.has_imu = True

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
        if (np.linalg.norm([e, n]) > MAX_EKF_POSITION_RADIUS_M
                or abs(u) > MAX_EKF_ALTITUDE_OFFSET_M):
            return None
        try:
            ref = self.reference_gps
            intermediate = geodesic(meters=abs(n)).destination(
                Point(ref[0], ref[1]),
                0.0 if n >= 0 else 180.0
            )
            dest = geodesic(meters=abs(e)).destination(
                Point(intermediate.latitude, intermediate.longitude),
                90.0 if e >= 0 else 270.0
            )
            return dest.latitude, dest.longitude, ref[2] + u
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
                line = raw.decode("ascii", errors="ignore").strip()
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
# MAIN
# ─────────────────────────────────────────────

def main():
    ekf = SensorFusionEKF()

    imu_ser = None
    gps_ser = None
    imu_reader = None

    if IMU_ENABLED:
        try:
            imu_ser = serial.Serial(IMU_PORT, IMU_BAUD, timeout=0)
            imu_reader = IMUPacketReader(imu_ser)
            print(f"Opened IMU: {IMU_PORT} @ {IMU_BAUD}")
            if not validate_imu_connection(imu_reader):
                print("[WARNING] IMU not responding. Disabling.")
                imu_ser.close()
                imu_ser = None
                imu_reader = None
        except serial.SerialException as exc:
            print(f"[WARNING] Could not open IMU: {exc}")

    if GPS_ENABLED:
        try:
            gps_ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=0.1)
            print(f"Opened GPS: {GPS_PORT} @ {GPS_BAUD}")
            if not validate_gps_connection(gps_ser):
                print("[WARNING] GPS not responding. Disabling.")
                gps_ser.close()
                gps_ser = None
        except serial.SerialException as exc:
            print(f"[WARNING] Could not open GPS: {exc}")

    if imu_ser is None and gps_ser is None:
        print("[ERROR] No sensors. Exiting.")
        return

    # ── Phase 1: Wait for GPS fix ──
    last_gps_error = None
    last_heading_deg = None
    last_valid_raw_gps = None

    if gps_ser is not None:
        print(f"Waiting for GPS fix ({GPS_FIX_TIMEOUT_S}s)...")
        start_time = time.time()

        try:
            while (ekf.reference_gps is None
                   and time.time() - start_time < GPS_FIX_TIMEOUT_S):

                gps_fix = read_gps_line(
                    gps_ser, last_valid_raw_gps=last_valid_raw_gps)

                if not gps_fix:
                    if imu_reader is not None:
                        for _ in range(50):
                            ts, data = imu_reader.read_packet()
                            if data is None:
                                break
                            if data.get("packet_type") == "RAW":
                                accel = np.array([
                                    data["accel_x"],
                                    data["accel_y"],
                                    data["accel_z"],
                                ], dtype=float)
                                ekf.try_initialize_orientation(accel)
                            elif data.get("packet_type") == "QUAT":
                                q = np.array([
                                    data["q2"], data["q3"],
                                    data["q4"], data["q1"],
                                ], dtype=float)
                                ekf.update_orientation_from_quat(q)
                    time.sleep(0.001)
                    continue

                if gps_fix.get("type") == "GST":
                    last_gps_error = (
                        gps_fix.get("lat_err"),
                        gps_fix.get("lon_err"),
                        gps_fix.get("alt_err"),
                    )
                elif gps_fix.get("type") == "HDT":
                    last_heading_deg = gps_fix.get("heading_deg")
                elif (gps_fix.get("type") == "GGA"
                      and gps_fix.get("valid")):
                    lat = gps_fix["lat"]
                    lon = gps_fix["lon"]
                    alt = gps_fix.get("alt")
                    last_valid_raw_gps = gps_fix.copy()

                    if lat and lon and lat != 0.0 and lon != 0.0:
                        ekf.set_reference_gps(lat, lon, alt)
                        print(f"Reference GPS set: {ekf.reference_gps}")

        except KeyboardInterrupt:
            print("\nStopped during GPS wait.")
            pos_logger.close()
            if imu_ser:
                imu_ser.close()
            if gps_ser:
                gps_ser.close()
            return

        if ekf.reference_gps is None:
            print("[WARNING] No GPS fix. IMU-only mode.")
    else:
        print("GPS disabled. IMU-only mode.")

    if imu_ser is None:
        print("IMU disabled. GPS-only mode.")

    print(f"Mode: {ekf.get_mode_string()}")
    print("Starting fusion loop...")

    # ── Phase 2: Main fusion loop ──
    imu_packets_read = 0
    imu_predictions_done = 0
    gps_lines_read = 0
    loop_counter = 0
    rate_window_start = time.time()
    last_predict_print_time = 0.0

    try:
        while True:
            loop_counter += 1

            # ── Drain IMU packets (max 100 per batch) ──
            imu_batch_count = 0
            if imu_reader is not None:
                while True:
                    imu_timestamp, imu_data = imu_reader.read_packet()
                    if imu_data is None:
                        break

                    imu_packets_read += 1
                    imu_batch_count += 1

                    if imu_data.get("packet_type") == "RAW":
                        gyro = np.array([
                            imu_data["gyro_x"],
                            imu_data["gyro_y"],
                            imu_data["gyro_z"],
                        ], dtype=float)
                        accel = np.array([
                            imu_data["accel_x"],
                            imu_data["accel_y"],
                            imu_data["accel_z"],
                        ], dtype=float)

                        predicted = ekf.update_imu(
                            gyro, accel, imu_timestamp)

                        if predicted:
                            imu_predictions_done += 1
                            pos_logger.log_ekf(ekf)

                            now = time.time()
                            if now - last_predict_print_time > 0.25:
                                pos_local = ekf.get_position_local()
                                vel = ekf.get_velocity()
                                yaw_display = ekf.get_yaw_deg()

                                gps_pos = ekf.get_gps_position()
                                if gps_pos is not None:
                                    print(
                                        f"[IMU] "
                                        f"lat={gps_pos[0]:.8f} "
                                        f"lon={gps_pos[1]:.8f} "
                                        f"alt={gps_pos[2]:.2f} "
                                        f"yaw={yaw_display:.1f} | "
                                        f"local=("
                                        f"{pos_local[0]:.2f},"
                                        f"{pos_local[1]:.2f},"
                                        f"{pos_local[2]:.2f}) "
                                        f"vel=("
                                        f"{vel[0]:.2f},"
                                        f"{vel[1]:.2f},"
                                        f"{vel[2]:.2f})"
                                    )
                                else:
                                    euler = ekf.get_euler_deg()
                                    print(
                                        f"[IMU-ONLY] "
                                        f"pos=("
                                        f"{pos_local[0]:.3f},"
                                        f"{pos_local[1]:.3f},"
                                        f"{pos_local[2]:.3f}) m | "
                                        f"vel=("
                                        f"{vel[0]:.3f},"
                                        f"{vel[1]:.3f},"
                                        f"{vel[2]:.3f}) m/s | "
                                        f"yaw={yaw_display:.1f} "
                                        f"pitch={euler[1]:.1f} "
                                        f"roll={euler[2]:.1f}"
                                    )
                                last_predict_print_time = now

                    elif imu_data.get("packet_type") == "QUAT":
                        q = np.array([
                            imu_data["q2"], imu_data["q3"],
                            imu_data["q4"], imu_data["q1"],
                        ], dtype=float)
                        ekf.update_orientation_from_quat(q)

                    if imu_batch_count > 100:
                        break

            # ── Read ALL available GPS lines ──
            if gps_ser is not None:
                gps_batch = 0
                while gps_batch < 50:
                    gps_fix = read_gps_line(
                        gps_ser,
                        last_valid_raw_gps=last_valid_raw_gps,
                    )

                    if gps_fix is None:
                        break

                    gps_batch += 1
                    gps_lines_read += 1
                    msg_type = gps_fix.get("type")

                    if msg_type == "GGA":
                        if gps_fix.get("valid"):
                            last_valid_raw_gps = gps_fix.copy()
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
                                        f"Ref: {ekf.reference_gps}"
                                    )

                            if ekf.reference_gps is not None:
                                gps_updated = ekf.update_gps(
                                    lat, lon, alt,
                                    hdop=gps_fix.get("hdop"),
                                )
                                if gps_updated:
                                    gps_pos = ekf.get_gps_position()
                                    if gps_pos is not None:
                                        print(
                                            f"[GPS UPDATE] "
                                            f"lat={gps_pos[0]:.8f} "
                                            f"lon={gps_pos[1]:.8f} "
                                            f"alt={gps_pos[2]:.2f} "
                                            f"yaw="
                                            f"{ekf.get_yaw_deg():.1f}"
                                        )

                    elif msg_type == "GST":
                        last_gps_error = (
                            gps_fix.get("lat_err"),
                            gps_fix.get("lon_err"),
                            gps_fix.get("alt_err"),
                        )

                    elif msg_type == "HDT":
                        last_heading_deg = gps_fix.get("heading_deg")
                        if last_heading_deg is not None:
                            ekf.update_heading(
                                last_heading_deg,
                                heading_sigma_deg=(
                                    DEFAULT_HEADING_SIGMA_DEG),
                            )

            # ── GPS timeout ──
            gps_age = ekf.gps_age_seconds()
            if gps_age is not None and gps_age > 5.0:
                inflate = min(gps_age * 0.5, 50.0)
                ekf.P[0:3, 0:3] += np.eye(3) * inflate * ekf.dt

            # ── Rate reporting ──
            now = time.time()
            if now - rate_window_start >= 2.0:
                elapsed = now - rate_window_start
                mode = ekf.get_mode_string()
                parts = [f"[RATE] mode={mode}"]

                if imu_reader is not None:
                    parts.append(
                        f"imu={imu_packets_read / elapsed:.0f}/s")
                    parts.append(
                        f"pred={imu_predictions_done / elapsed:.0f}/s")
                    parts.append(
                        f"skip={ekf.skipped_predicts}")
                    parts.append(f"dt={ekf.dt:.6f}")

                if gps_ser is not None:
                    parts.append(
                        f"gps={gps_lines_read / elapsed:.0f}/s")

                parts.append(
                    f"loops={loop_counter / elapsed:.0f}/s")

                print(" | ".join(parts))

                loop_counter = 0
                imu_packets_read = 0
                imu_predictions_done = 0
                gps_lines_read = 0
                rate_window_start = now

            if imu_batch_count == 0:
                if imu_reader is not None:
                    time.sleep(0.0002)
                else:
                    time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        pos_logger.close()
        if imu_ser is not None:
            imu_ser.close()
        if gps_ser is not None:
            gps_ser.close()
        print(f"Closed. Mode: {ekf.get_mode_string()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nForce stopped.")
    finally:
        try:
            pos_logger.close()
        except Exception:
            pass
