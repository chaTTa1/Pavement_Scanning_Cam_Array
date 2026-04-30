# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:56:01 2026

@author: Desktop
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:58:43 2026

@author: Desktop
"""

"""
GPS/IMU Raw Data Recorder (with Time Synchronization Support)
===============================================================
Records ALL raw GPS NMEA and IMU binary data to files with
precise timestamps for offline time alignment and EKF post-processing.

Time synchronization strategy:
  Three clocks exist:
    1. IMU hardware clock (imu_timestamp_us): microsecond counter, monotonic,
       precise relative timing between IMU samples, but unknown epoch.
    2. GPS UTC clock (gps_utc): extracted from GGA/RMC NMEA time field,
       absolute UTC time of GPS measurement, ~10ns accuracy.
    3. PC wall clock (wall_time): time.perf_counter anchored to time.time(),
       used to correlate IMU and GPS clocks.

  Post-processing alignment:
    - GPS UTC is the absolute time reference
    - wall_time on GPS events gives PC-clock at GPS arrival
    - wall_time on IMU events gives PC-clock at IMU arrival
    - imu_timestamp_us gives precise inter-sample timing
    - By comparing wall_time of GPS and nearby IMU events,
      you can establish: imu_timestamp_us = f(GPS_UTC)
    - Then all IMU samples can be mapped to GPS UTC time

  The key equation (computed offline):
    gps_utc_of_imu_sample = gps_utc_nearest
                          + (imu_ts - imu_ts_at_nearest_gps) / 1e6

Output files:
  - imu_raw.csv:      RAW packets (gyro + accel + mag) at 400Hz
  - imu_quat.csv:     QUAT packets (onboard AHRS quaternion)
  - imu_rpy.csv:      RPY packets (onboard AHRS euler angles)
  - imu_gravity.csv:  GRAVITY packets (gravity vector in body frame)
  - gps_raw.csv:      ALL parsed NMEA sentences with raw string
  - recording_info.txt: metadata about the recording session
"""

import serial
import serial.tools.list_ports
import pynmea2
import struct
import time
import csv
import os
import sys
import argparse
import math
import threading
import queue
from datetime import datetime, timezone


# ─────────────────────────────────────────────
# HIGH-RESOLUTION TIMER
# ─────────────────────────────────────────────

def _calibrate_perf_offset(n_samples=20):
    best_offset = None
    best_gap = float("inf")
    for _ in range(n_samples):
        t1 = time.perf_counter()
        wall = time.time()
        t2 = time.perf_counter()
        gap = t2 - t1
        if gap < best_gap:
            best_gap = gap
            best_offset = wall - (t1 + t2) / 2.0
    return best_offset


_PERF_EPOCH_OFFSET = _calibrate_perf_offset()


def wall_clock():
    return _PERF_EPOCH_OFFSET + time.perf_counter()


# ─────────────────────────────────────────────
# NON-BLOCKING CONSOLE OUTPUT
# ─────────────────────────────────────────────

_print_q = queue.Queue(maxsize=200)
_print_stop = threading.Event()


def _print_worker():
    while not _print_stop.is_set():
        try:
            msg = _print_q.get(timeout=0.1)
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()
        except queue.Empty:
            continue
        except Exception:
            pass


_print_thread = threading.Thread(
    target=_print_worker, daemon=True, name="PrintWorker")
_print_thread.start()


def nb_print(msg):
    try:
        _print_q.put_nowait(str(msg))
    except queue.Full:
        pass


def stop_print_worker():
    _print_stop.set()
    while not _print_q.empty():
        try:
            sys.stdout.write(_print_q.get_nowait() + "\n")
        except queue.Empty:
            break
    sys.stdout.flush()


# ─────────────────────────────────────────────
# NMEA UTC TIME EXTRACTION
# ─────────────────────────────────────────────

def nmea_time_to_utc_seconds(nmea_time_str, nmea_date_str=None):
    """
    Convert NMEA time string (HHMMSS.SS) and optional date (DDMMYY)
    to UTC seconds since midnight (if no date) or epoch seconds (if date).

    GGA has time but no date → returns seconds since midnight UTC.
    RMC has both time and date → returns full epoch seconds.

    Returns (utc_seconds, has_date) tuple.
    """
    if nmea_time_str is None or nmea_time_str == "":
        return None, False

    try:
        nmea_time_str = nmea_time_str.strip()
        if len(nmea_time_str) < 6:
            return None, False

        hours = int(nmea_time_str[0:2])
        minutes = int(nmea_time_str[2:4])
        seconds = float(nmea_time_str[4:])

        seconds_since_midnight = hours * 3600.0 + minutes * 60.0 + seconds

        if nmea_date_str is not None and nmea_date_str.strip() != "":
            nmea_date_str = nmea_date_str.strip()
            if len(nmea_date_str) >= 6:
                day = int(nmea_date_str[0:2])
                month = int(nmea_date_str[2:4])
                year = int(nmea_date_str[4:6])
                if year < 80:
                    year += 2000
                else:
                    year += 1900

                dt = datetime(year, month, day,
                              hours, minutes, int(seconds),
                              int((seconds % 1) * 1e6),
                              tzinfo=timezone.utc)
                return dt.timestamp(), True

        return seconds_since_midnight, False

    except (ValueError, IndexError):
        return None, False


class GpsWallClockCalibrator:
    def __init__(self, window_size=60):
        self._pairs = []
        self._window = window_size
        self.offset = None
        self.drift_ppm = None
        self._scale = None
        self._intercept = None
        self._n_updates = 0

    def add_sample(self, wt, gps_utc):
        if wt is None or gps_utc is None:
            return
        self._pairs.append((float(wt), float(gps_utc)))
        if len(self._pairs) > self._window:
            self._pairs = self._pairs[-self._window:]
        self._update()

    def _update(self):
        if len(self._pairs) == 1:
            self.offset = self._pairs[0][1] - self._pairs[0][0]
            self.drift_ppm = 0.0
            self._scale = 1.0
            self._intercept = self.offset
            return
        if len(self._pairs) < 2:
            return

        n = float(len(self._pairs))
        sx = sum(p[0] for p in self._pairs)
        sy = sum(p[1] for p in self._pairs)
        sxx = sum(p[0] * p[0] for p in self._pairs)
        sxy = sum(p[0] * p[1] for p in self._pairs)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            self.offset = sum(p[1] - p[0] for p in self._pairs) / len(self._pairs)
            self.drift_ppm = 0.0
            self._scale = 1.0
            self._intercept = self.offset
            return

        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n
        mean_wt = sx / n
        self.offset = b + (a - 1.0) * mean_wt
        self.drift_ppm = (a - 1.0) * 1e6
        self._scale = a
        self._intercept = b
        self._n_updates += 1

    def wall_to_gps_utc(self, wt):
        if self.offset is None:
            return None
        if self._scale is None or self._intercept is None:
            return wt + self.offset
        return self._scale * wt + self._intercept

    def summary(self):
        return {
            "n_samples": len(self._pairs),
            "n_updates": self._n_updates,
            "offset_s": self.offset,
            "drift_ppm": self.drift_ppm,
        }


class GpsArrivalJitterTracker:
    def __init__(self):
        self._last_gga_wt = None
        self._last_rmc_wt = None
        self.gga_intervals = []
        self.rmc_intervals = []

    def on_gga(self, wt):
        if self._last_gga_wt is not None:
            self.gga_intervals.append(float(wt) - self._last_gga_wt)
        self._last_gga_wt = float(wt)

    def on_rmc(self, wt):
        if self._last_rmc_wt is not None:
            self.rmc_intervals.append(float(wt) - self._last_rmc_wt)
        self._last_rmc_wt = float(wt)

    @staticmethod
    def _summarize(intervals):
        if len(intervals) <= 10:
            return None
        arr = sorted(float(x) for x in intervals)
        nominal = arr[len(arr) // 2]
        jitter = [x - nominal for x in arr]
        mean_j = sum(jitter) / len(jitter)
        var_j = sum((x - mean_j) ** 2 for x in jitter) / len(jitter)
        return {
            "nominal_interval_ms": nominal * 1000.0,
            "jitter_std_ms": math.sqrt(var_j) * 1000.0,
            "jitter_max_ms": max(abs(x) for x in jitter) * 1000.0,
            "n_samples": len(arr),
        }

    def summary(self):
        result = {}
        gga = self._summarize(self.gga_intervals)
        rmc = self._summarize(self.rmc_intervals)
        if gga is not None:
            result["GGA"] = gga
        if rmc is not None:
            result["RMC"] = rmc
        return result


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

    gps_port = find_septentrio_nmea_port(timeout=3.0)

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
        description="GPS/IMU Raw Data Recorder")
    parser.add_argument("--imu-port", type=str, default=None)
    parser.add_argument("--gps-port", type=str, default=None)
    parser.add_argument("--imu-baud", type=int, default=None)
    parser.add_argument("--gps-baud", type=int, default=None)
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--no-gps", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--duration", type=float, default=None,
                        help="Recording duration in seconds (None=until Ctrl+C)")
    parser.add_argument("--warmup-s", type=float, default=2.0,
                        help=("Seconds to read and discard sensor data before "
                              "recording starts"))
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for CSV files")
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
RECORD_DURATION = _args.duration
WARMUP_S = max(0.0, _args.warmup_s)
OUTPUT_DIR = _args.output_dir

IMU_PORT, GPS_PORT = resolve_ports(OS_NAME, _args)

IMU_BAUD = _args.imu_baud if _args.imu_baud else 115200
GPS_BAUD = _args.gps_baud if _args.gps_baud else 115200

SERIAL_BUFFER_SIZE = 65536
EVENT_QUEUE_SIZE = 100000

IMU_SYNC_BYTE_1 = 0xAA
IMU_SYNC_BYTE_2 = 0x55
IMU_CMD_RAW = 41
IMU_CMD_QUAT = 32
IMU_CMD_RPY = 35
IMU_CMD_GRAVITY = 36
IMU_CRC_BYTES = 2

MIN_FIX_QUALITY = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"OS: {OS_NAME}")
print(f"IMU: {'ENABLED' if IMU_ENABLED else 'DISABLED'} | {IMU_PORT} @ {IMU_BAUD}")
print(f"GPS: {'ENABLED' if GPS_ENABLED else 'DISABLED'} | {GPS_PORT} @ {GPS_BAUD}")
print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
print(f"Event queue: {EVENT_QUEUE_SIZE} slots (blocking put, no drops)")
if RECORD_DURATION:
    print(f"Duration: {RECORD_DURATION:.0f} seconds")
else:
    print(f"Duration: until Ctrl+C")
print(f"Warmup discard: {WARMUP_S:.1f} seconds")


# ─────────────────────────────────────────────
# CRC-16 MODBUS
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
# ASYNC CSV LOGGER
# ─────────────────────────────────────────────

class AsyncCSVLogger:
    def __init__(self, filename, headers, q_size=10000):
        self.filename = filename
        self.headers = headers
        self._q = queue.Queue(maxsize=q_size)
        self._stop = threading.Event()
        self._rows_written = 0
        self._rows_dropped = 0
        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name=f"Logger-{os.path.basename(filename)}")
        self._thread.start()

    def _writer_loop(self):
        write_header = (not os.path.exists(self.filename)
                       or os.path.getsize(self.filename) == 0)
        f = open(self.filename, "a", newline="")
        writer = csv.writer(f)
        if write_header:
            writer.writerow(self.headers)

        batch = []
        while not self._stop.is_set():
            try:
                row = self._q.get(timeout=0.05)
                batch.append(row)
            except queue.Empty:
                if batch:
                    for b in batch:
                        writer.writerow(b)
                    self._rows_written += len(batch)
                    batch.clear()
                    f.flush()
                continue

            while len(batch) < 1000:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break

            for b in batch:
                writer.writerow(b)
            self._rows_written += len(batch)
            batch.clear()

            if self._rows_written % 10000 < 1000:
                f.flush()

        while not self._q.empty():
            try:
                writer.writerow(self._q.get_nowait())
                self._rows_written += 1
            except queue.Empty:
                break
        f.flush()
        f.close()

    def write_row(self, row):
        try:
            self._q.put_nowait(row)
        except queue.Full:
            self._rows_dropped += 1

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    @property
    def stats(self):
        return {
            "written": self._rows_written,
            "dropped": self._rows_dropped,
            "pending": self._q.qsize(),
        }

    def __del__(self):
        if not self._stop.is_set():
            self.close()


# ─── Create loggers ───
# Each IMU packet type gets its own file for clean post-processing.
# Time columns:
#   wall_time:        PC clock at serial read (for cross-sensor correlation)
#   imu_timestamp_us: IMU hardware clock (for precise inter-sample timing)
#   gps_utc:          GPS satellite time (absolute reference, GPS files only)

imu_raw_logger = AsyncCSVLogger(
    os.path.join(OUTPUT_DIR, "imu_raw.csv"),
    [
        "wall_time",            # PC clock epoch seconds (for correlation with GPS)
        "imu_timestamp_us",     # IMU hardware clock microseconds (precise relative timing)
        "gyro_x_rad_s",         # angular rate X body frame (rad/s)
        "gyro_y_rad_s",         # angular rate Y body frame (rad/s)
        "gyro_z_rad_s",         # angular rate Z body frame (rad/s)
        "accel_x_g",            # acceleration X body frame (g, INCLUDES gravity)
        "accel_y_g",            # acceleration Y body frame (g, INCLUDES gravity)
        "accel_z_g",            # acceleration Z body frame (g, INCLUDES gravity)
        "mag_x",                # magnetometer X (device units, may be empty)
        "mag_y",                # magnetometer Y
        "mag_z",                # magnetometer Z
    ],
    q_size=10000,
)

imu_quat_logger = AsyncCSVLogger(
    os.path.join(OUTPUT_DIR, "imu_quat.csv"),
    [
        "wall_time",
        "imu_timestamp_us",
        "q1",                   # quaternion scalar (w)
        "q2",                   # quaternion x
        "q3",                   # quaternion y
        "q4",                   # quaternion z
    ],
    q_size=5000,
)

imu_rpy_logger = AsyncCSVLogger(
    os.path.join(OUTPUT_DIR, "imu_rpy.csv"),
    [
        "wall_time",
        "imu_timestamp_us",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
    ],
    q_size=5000,
)

imu_gravity_logger = AsyncCSVLogger(
    os.path.join(OUTPUT_DIR, "imu_gravity.csv"),
    [
        "wall_time",
        "imu_timestamp_us",
        "gravity_x_g",          # gravity vector X body frame (g)
        "gravity_y_g",          # gravity vector Y body frame (g)
        "gravity_z_g",          # gravity vector Z body frame (g)
    ],
    q_size=5000,
)

gps_logger = AsyncCSVLogger(
    os.path.join(OUTPUT_DIR, "gps_raw.csv"),
    [
        "wall_time",            # PC clock epoch seconds (for correlation with IMU)
        "gps_utc",              # GPS UTC time: epoch seconds (RMC) or seconds-since-midnight (GGA)
        "gps_utc_type",         # "epoch" if full date+time, "tod" if time-of-day only
        "msg_type",             # GGA, GST, HDT, RMC, VTG, or sentence type
        "lat",                  # decimal degrees
        "lon",                  # decimal degrees
        "alt",                  # meters
        "fix_quality",          # 0-5
        "num_sats",             # satellite count
        "hdop",                 # horizontal dilution of precision
        "lat_err_m",            # 1-sigma meters (GST)
        "lon_err_m",            # 1-sigma meters (GST)
        "alt_err_m",            # 1-sigma meters (GST)
        "heading_deg",          # true heading (HDT)
        "speed_knots",          # speed over ground (RMC/VTG)
        "speed_kmh",            # speed over ground km/h (VTG)
        "course_true_deg",      # course over ground (RMC/VTG)
        "rmc_status",           # A=active V=void (RMC)
        "raw_nmea",             # complete raw sentence
        "nearest_imu_wall_time", # most recent IMU event wall_time at GPS arrival
        "nearest_imu_timestamp_us",
        "nearest_imu_dt_ms",     # GPS wall_time - nearest_imu_wall_time
    ],
    q_size=5000,
)


# ─────────────────────────────────────────────
# IMU PACKET READER (with CRC)
# ─────────────────────────────────────────────

class IMUPacketReader:
    def __init__(self, ser):
        self.ser = ser
        self._buf = bytearray()
        self.crc_pass_count = 0
        self.crc_fail_count = 0
        self._last_read_time = None

    def _refill(self):
        avail = self.ser.in_waiting
        if avail > 0:
            read_time = wall_clock()
            chunk = self.ser.read(min(avail, 4096))
            if chunk:
                self._last_read_time = read_time
                self._buf.extend(chunk)

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
                return None, None, None

            if sync_idx > 0:
                self._buf = self._buf[sync_idx:]

            if len(self._buf) < 3:
                return None, None, None

            length = self._buf[2]
            total_needed = 3 + length + IMU_CRC_BYTES
            if len(self._buf) < total_needed:
                return None, None, None

            length_byte = bytes([length])
            payload = bytes(self._buf[3: 3 + length])
            crc_bytes = bytes(self._buf[3 + length: 3 + length + IMU_CRC_BYTES])

            self._buf = self._buf[total_needed:]

            crc_received = struct.unpack("<H", crc_bytes)[0]
            crc_calculated = crc16_modbus(length_byte + payload)

            if crc_received != crc_calculated:
                self.crc_fail_count += 1
                continue

            self.crc_pass_count += 1

            if len(payload) < 4:
                continue

            header = struct.unpack("<I", payload[:4])[0]
            cmd_id = header & 0x7F

            timestamp, data = self._parse_packet(cmd_id, payload[4:])
            if timestamp is not None and data is not None:
                return timestamp, data, self._last_read_time

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
                    "gyro_x": vals[1],
                    "gyro_y": vals[2],
                    "gyro_z": vals[3],
                    "accel_x": vals[4],
                    "accel_y": vals[5],
                    "accel_z": vals[6],
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

            if cmd_id == IMU_CMD_GRAVITY:
                if len(data) < 16:
                    return None, None
                vals = struct.unpack("<Ifff", data[:16])
                return vals[0], {
                    "packet_type": "GRAVITY",
                    "gravity_x": vals[1],
                    "gravity_y": vals[2],
                    "gravity_z": vals[3],
                }

        except struct.error:
            pass
        return None, None


# ─────────────────────────────────────────────
# GPS LINE PARSER
# ─────────────────────────────────────────────

def _gps_imu_crossref_fields(wt, nearest_imu_wt=None, nearest_imu_ts=None):
    if nearest_imu_wt is None:
        return ["", "", ""]
    dt_ms = (float(wt) - float(nearest_imu_wt)) * 1000.0
    return [
        f"{float(nearest_imu_wt):.6f}",
        nearest_imu_ts if nearest_imu_ts is not None else "",
        f"{dt_ms:.3f}",
    ]


def parse_and_log_gps_line(line, wt, nearest_imu_wt=None, nearest_imu_ts=None):
    crossref = _gps_imu_crossref_fields(wt, nearest_imu_wt, nearest_imu_ts)
    if not line:
        return None

    if not line.startswith("$"):
        return None

    if "*" in line and not valid_nmea_checksum(line):
        gps_logger.write_row([
            f"{wt:.6f}",
            "", "",
            "CHECKSUM_ERR",
            "", "", "",
            "", "", "",
            "", "", "",
            "",
            "", "", "", "",
            line,
        ] + crossref)
        return None

    try:
        msg = pynmea2.parse(line)

        # ── GGA ──
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

            gps_utc, has_date = nmea_time_to_utc_seconds(
                msg.timestamp.isoformat() if msg.timestamp else None)
            if gps_utc is None and msg.data and len(msg.data) > 0:
                gps_utc, has_date = nmea_time_to_utc_seconds(msg.data[0])

            gps_logger.write_row([
                f"{wt:.6f}",
                f"{gps_utc:.6f}" if gps_utc is not None else "",
                "epoch" if has_date else "tod",
                "GGA",
                f"{lat:.10f}" if lat is not None else "",
                f"{lon:.10f}" if lon is not None else "",
                f"{alt:.4f}" if alt is not None else "",
                fix_quality if fix_quality is not None else "",
                num_sats if num_sats is not None else "",
                f"{hdop:.2f}" if hdop is not None else "",
                "", "", "",
                "",
                "", "", "", "",
                line,
            ] + crossref)

            return {
                "type": "GGA",
                "valid": fix_quality is not None and fix_quality >= 1,
                "lat": lat, "lon": lon, "alt": alt,
                "fix_quality": fix_quality, "num_sats": num_sats,
                "hdop": hdop,
                "gps_utc": gps_utc,
                "gps_utc_type": "epoch" if has_date else "tod",
            }

        # ── GST ──
        if isinstance(msg, pynmea2.types.talker.GST):
            lat_err = None
            lon_err = None
            alt_err = None
            if len(msg.data) > 5 and msg.data[5]:
                try:
                    lat_err = float(msg.data[5])
                except ValueError:
                    pass
            if len(msg.data) > 6 and msg.data[6]:
                try:
                    lon_err = float(msg.data[6])
                except ValueError:
                    pass
            if len(msg.data) > 7 and msg.data[7]:
                try:
                    alt_err = float(msg.data[7])
                except ValueError:
                    pass

            gps_utc, has_date = nmea_time_to_utc_seconds(
                msg.data[0] if len(msg.data) > 0 else None)

            gps_logger.write_row([
                f"{wt:.6f}",
                f"{gps_utc:.6f}" if gps_utc is not None else "",
                "epoch" if has_date else "tod",
                "GST",
                "", "", "",
                "", "", "",
                f"{lat_err:.4f}" if lat_err is not None else "",
                f"{lon_err:.4f}" if lon_err is not None else "",
                f"{alt_err:.4f}" if alt_err is not None else "",
                "",
                "", "", "", "",
                line,
            ] + crossref)

            return {"type": "GST", "lat_err": lat_err,
                    "lon_err": lon_err, "alt_err": alt_err}

        # ── HDT ──
        if isinstance(msg, pynmea2.types.talker.HDT):
            heading = None
            if msg.heading:
                try:
                    heading = float(msg.heading)
                except ValueError:
                    pass

            gps_logger.write_row([
                f"{wt:.6f}",
                "", "",
                "HDT",
                "", "", "",
                "", "", "",
                "", "", "",
                f"{heading:.4f}" if heading is not None else "",
                "", "", "", "",
                line,
            ] + crossref)

            return {"type": "HDT", "heading_deg": heading}

        # ── RMC ──
        if isinstance(msg, pynmea2.types.talker.RMC):
            lat = nmea_to_decimal_degrees(
                msg.lat, msg.lat_dir, is_latitude=True)
            lon = nmea_to_decimal_degrees(
                msg.lon, msg.lon_dir, is_latitude=False)

            speed_knots = None
            if msg.spd_over_grnd not in (None, ""):
                try:
                    speed_knots = float(msg.spd_over_grnd)
                except ValueError:
                    pass

            course_true = None
            if msg.true_course not in (None, ""):
                try:
                    course_true = float(msg.true_course)
                except ValueError:
                    pass

            rmc_status = None
            if msg.status not in (None, ""):
                rmc_status = msg.status

            speed_kmh = None
            if speed_knots is not None:
                speed_kmh = speed_knots * 1.852

            time_str = msg.data[0] if len(msg.data) > 0 else None
            date_str = msg.data[8] if len(msg.data) > 8 else None
            gps_utc, has_date = nmea_time_to_utc_seconds(time_str, date_str)

            gps_logger.write_row([
                f"{wt:.6f}",
                f"{gps_utc:.6f}" if gps_utc is not None else "",
                "epoch" if has_date else "tod",
                "RMC",
                f"{lat:.10f}" if lat is not None else "",
                f"{lon:.10f}" if lon is not None else "",
                "",
                "", "", "",
                "", "", "",
                "",
                f"{speed_knots:.4f}" if speed_knots is not None else "",
                f"{speed_kmh:.4f}" if speed_kmh is not None else "",
                f"{course_true:.4f}" if course_true is not None else "",
                rmc_status if rmc_status is not None else "",
                line,
            ] + crossref)

            return {
                "type": "RMC",
                "lat": lat, "lon": lon,
                "speed_knots": speed_knots,
                "course_true": course_true,
                "status": rmc_status,
                "gps_utc": gps_utc,
                "gps_utc_type": "epoch" if has_date else "tod",
            }

        # ── VTG ──
        if isinstance(msg, pynmea2.types.talker.VTG):
            course_true = None
            speed_knots = None
            speed_kmh = None

            if hasattr(msg, 'true_track') and msg.true_track not in (None, ""):
                try:
                    course_true = float(msg.true_track)
                except (ValueError, AttributeError):
                    pass

            if hasattr(msg, 'spd_over_grnd_kts') and msg.spd_over_grnd_kts not in (None, ""):
                try:
                    speed_knots = float(msg.spd_over_grnd_kts)
                except (ValueError, AttributeError):
                    pass

            if hasattr(msg, 'spd_over_grnd_kmph') and msg.spd_over_grnd_kmph not in (None, ""):
                try:
                    speed_kmh = float(msg.spd_over_grnd_kmph)
                except (ValueError, AttributeError):
                    pass

            if course_true is None and len(msg.data) > 0 and msg.data[0]:
                try:
                    course_true = float(msg.data[0])
                except ValueError:
                    pass
            if speed_knots is None and len(msg.data) > 4 and msg.data[4]:
                try:
                    speed_knots = float(msg.data[4])
                except ValueError:
                    pass
            if speed_kmh is None and len(msg.data) > 6 and msg.data[6]:
                try:
                    speed_kmh = float(msg.data[6])
                except ValueError:
                    pass

            gps_logger.write_row([
                f"{wt:.6f}",
                "", "",
                "VTG",
                "", "", "",
                "", "", "",
                "", "", "",
                "",
                f"{speed_knots:.4f}" if speed_knots is not None else "",
                f"{speed_kmh:.4f}" if speed_kmh is not None else "",
                f"{course_true:.4f}" if course_true is not None else "",
                "",
                line,
            ] + crossref)

            return {
                "type": "VTG",
                "speed_knots": speed_knots,
                "speed_kmh": speed_kmh,
                "course_true": course_true,
            }

        # ── All other sentence types ──
        sentence_type = "OTHER"
        try:
            sentence_type = msg.sentence_type
        except AttributeError:
            if len(line) > 6:
                sentence_type = line[3:6].rstrip(",")

        gps_logger.write_row([
            f"{wt:.6f}",
            "", "",
            sentence_type,
            "", "", "",
            "", "", "",
            "", "", "",
            "",
            "", "", "", "",
            line,
        ] + crossref)

        return None

    except pynmea2.ParseError:
        gps_logger.write_row([
            f"{wt:.6f}",
            "", "",
            "PARSE_ERR",
            "", "", "",
            "", "", "",
            "", "", "",
            "",
            "", "", "", "",
            line,
        ] + crossref)
        return None

    except Exception:
        gps_logger.write_row([
            f"{wt:.6f}",
            "", "",
            "EXCEPTION",
            "", "", "",
            "", "", "",
            "", "", "",
            "",
            "", "", "", "",
            line,
        ] + crossref)
        return None


# ─────────────────────────────────────────────
# EVENT TYPES
# ─────────────────────────────────────────────

EVENT_IMU = "imu"
EVENT_GPS = "gps"


# ─────────────────────────────────────────────
# THREADED SERIAL READERS (blocking put)
# ─────────────────────────────────────────────

class IMUReaderThread(threading.Thread):
    def __init__(self, ser, event_q, stop_event):
        super().__init__(daemon=True, name="IMUReader")
        self.reader = IMUPacketReader(ser)
        self.event_q = event_q
        self.stop_event = stop_event
        self.packets_read = 0
        self.packets_queued = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                ts, data, read_wt = self.reader.read_packet()
                if data is not None:
                    self.packets_read += 1
                    wt = read_wt if read_wt is not None else wall_clock()
                    try:
                        self.event_q.put(
                            (wt, EVENT_IMU, (ts, data)),
                            timeout=1.0)
                        self.packets_queued += 1
                    except queue.Full:
                        nb_print("[IMU THREAD] WARNING: queue full after 1s!")
                else:
                    time.sleep(0.0001)
            except serial.SerialException:
                if not self.stop_event.is_set():
                    nb_print("[IMU THREAD] Serial error")
                break
            except Exception as exc:
                if not self.stop_event.is_set():
                    nb_print(f"[IMU THREAD] Error: {exc}")
                time.sleep(0.001)


class GPSReaderThread(threading.Thread):
    def __init__(self, ser, event_q, stop_event):
        super().__init__(daemon=True, name="GPSReader")
        self.ser = ser
        self.event_q = event_q
        self.stop_event = stop_event
        self.lines_read = 0
        self.lines_queued = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                raw_bytes = self.ser.readline()
                if not raw_bytes:
                    continue

                try:
                    line = raw_bytes.decode("ascii", errors="strict").strip()
                except UnicodeDecodeError:
                    continue

                if not line:
                    continue

                if not line.startswith("$"):
                    continue

                self.lines_read += 1
                wt = wall_clock()
                try:
                    self.event_q.put(
                        (wt, EVENT_GPS, line),
                        timeout=1.0)
                    self.lines_queued += 1
                except queue.Full:
                    nb_print("[GPS THREAD] WARNING: queue full after 1s!")

            except serial.SerialException:
                if not self.stop_event.is_set():
                    nb_print("[GPS THREAD] Serial error")
                break
            except Exception as exc:
                if not self.stop_event.is_set():
                    nb_print(f"[GPS THREAD] Error: {exc}")
                time.sleep(0.01)


# ─────────────────────────────────────────────
# SERIAL BUFFER
# ─────────────────────────────────────────────

def enlarge_serial_buffer(ser, size=SERIAL_BUFFER_SIZE):
    try:
        ser.set_buffer_size(rx_size=size)
        print(f"  Serial RX buffer set to {size} bytes")
    except (AttributeError, serial.SerialException, OSError):
        print(f"  Serial RX buffer resize not supported (OS manages it)")


def clear_serial_input_buffer(ser, name):
    if ser is None:
        return
    try:
        waiting = ser.in_waiting
    except (AttributeError, serial.SerialException, OSError):
        waiting = None

    try:
        ser.reset_input_buffer()
        if waiting is None:
            print(f"  {name} RX buffer cleared")
        else:
            print(f"  {name} RX buffer cleared ({waiting} bytes discarded)")
    except (AttributeError, serial.SerialException, OSError) as exc:
        print(f"  {name} RX buffer clear not supported: {exc}")


def drain_event_queue(event_q):
    count = 0
    while True:
        try:
            event_q.get_nowait()
            count += 1
        except queue.Empty:
            break
    return count


def warmup_sensor_threads(event_q, warmup_s):
    if warmup_s <= 0.0:
        return 0, 0

    print()
    print(f"[WARMUP] Reading sensors for {warmup_s:.1f}s before logging...")
    print("[WARMUP] Data during this period is discarded.")

    start = wall_clock()
    imu_events = 0
    gps_events = 0
    while wall_clock() - start < warmup_s:
        try:
            wt, evt_type, payload = event_q.get(timeout=0.05)
        except queue.Empty:
            continue

        if evt_type == EVENT_IMU:
            imu_events += 1
        elif evt_type == EVENT_GPS:
            gps_events += 1

    discarded = drain_event_queue(event_q)
    print(f"[WARMUP] Discarded {imu_events} IMU events, "
          f"{gps_events} GPS events, plus {discarded} queued events.")
    return imu_events, gps_events


# ─────────────────────────────────────────────
# DEVICE VALIDATION
# ─────────────────────────────────────────────

def validate_imu_connection(imu_reader, timeout=2.0):
    print(f"Validating IMU ({timeout}s)...")
    start = time.time()
    packets = 0
    while time.time() - start < timeout:
        ts, data, _ = imu_reader.read_packet()
        if data is not None:
            packets += 1
            if packets >= 5:
                print(f"  IMU OK: {packets} packets, "
                      f"CRC pass={imu_reader.crc_pass_count} "
                      f"fail={imu_reader.crc_fail_count}")
                return True
    print(f"  IMU FAILED: only {packets} packets, "
          f"CRC pass={imu_reader.crc_pass_count} "
          f"fail={imu_reader.crc_fail_count}")
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
                        print(f"  GPS OK: {lines} NMEA lines received")
                        return True
        except Exception:
            pass
    print(f"  GPS FAILED: only {lines} NMEA lines")
    return False


# ─────────────────────────────────────────────
# RECORDING INFO
# ─────────────────────────────────────────────

def write_recording_info(output_dir, imu_port, gps_port,
                         imu_baud, gps_baud, start_time):
    filepath = os.path.join(output_dir, "recording_info.txt")
    with open(filepath, "w") as f:
        f.write("Recording Session Info\n")
        f.write("======================\n")
        f.write(f"Start time:     {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"Start epoch:    {start_time:.6f}\n")
        f.write(f"OS:             {OS_NAME}\n")
        f.write(f"IMU port:       {imu_port}\n")
        f.write(f"IMU baud:       {imu_baud}\n")
        f.write(f"GPS port:       {gps_port}\n")
        f.write(f"GPS baud:       {gps_baud}\n")
        f.write(f"IMU enabled:    {IMU_ENABLED}\n")
        f.write(f"GPS enabled:    {GPS_ENABLED}\n")
        f.write(f"Duration:       {RECORD_DURATION if RECORD_DURATION else 'unlimited'}\n")
        f.write(f"Warmup discard: {WARMUP_S:.1f} seconds\n")
        f.write(f"Output dir:     {os.path.abspath(output_dir)}\n")
        f.write(f"Serial buffer:  {SERIAL_BUFFER_SIZE} bytes\n")
        f.write(f"Event queue:    {EVENT_QUEUE_SIZE} slots (blocking put)\n")
        f.write(f"\n")
        f.write("Time synchronization:\n")
        f.write("  wall_time:        PC perf_counter anchored to time.time()\n")
        f.write("  imu_timestamp_us: IMU hardware clock (microseconds, wraps at 2^32)\n")
        f.write("  gps_utc:          UTC from NMEA sentences (epoch or time-of-day)\n")
        f.write("  gps_utc_type:     'epoch' = full datetime, 'tod' = seconds since midnight\n")
        f.write("  Alignment method: use wall_time to correlate IMU and GPS clocks offline\n")
        f.write("  gps_raw crossref: nearest IMU wall_time/timestamp recorded per GPS sentence\n")
        f.write("  RMC clock model:  online GPS UTC = scale * wall_time + intercept when RMC exists\n")
        f.write(f"\n")
        f.write("IMU: SYD Dynamics TransducerM TM151\n")
        f.write("  Protocol: EasyProfile binary with CRC-16/MODBUS\n")
        f.write("  Rate: 400 Hz\n")
        f.write("  RAW packet (cmd 41):\n")
        f.write("    Field order: timestamp, gyroX, gyroY, gyroZ, accX, accY, accZ, [magX, magY, magZ]\n")
        f.write("    Gyro unit: rad/s\n")
        f.write("    Accel unit: g (INCLUDES GRAVITY)\n")
        f.write("  QUAT packet (cmd 32): [q1(w), q2(x), q3(y), q4(z)]\n")
        f.write("  RPY packet (cmd 35): [roll, pitch, yaw] degrees\n")
        f.write("  GRAVITY packet (cmd 36): [gx, gy, gz] in g (body frame)\n")
        f.write(f"\n")
        f.write("GPS: Septentrio (NMEA output)\n")
        f.write("  Sentences: GGA, GST, HDT, RMC, VTG\n")
        f.write("  Rate: ~10 Hz\n")
        f.write("  RTK corrections: enabled\n")
        f.write(f"\n")
        f.write("Output files:\n")
        f.write("  imu_raw.csv:     RAW packets (gyro + accel + mag)\n")
        f.write("  imu_quat.csv:    QUAT packets (AHRS quaternion)\n")
        f.write("  imu_rpy.csv:     RPY packets (AHRS euler angles)\n")
        f.write("  imu_gravity.csv: GRAVITY packets (gravity vector body frame)\n")
        f.write("  gps_raw.csv:     All NMEA sentences with parsed fields + raw string\n")
    print(f"Recording info written to: {filepath}")


def update_recording_info(output_dir, end_time, duration,
                          imu_raw_stats, imu_quat_stats,
                          imu_rpy_stats, imu_gravity_stats,
                          gps_stats,
                          imu_thread_stats, gps_thread_stats,
                          crc_pass, crc_fail,
                          gps_clock_summary=None,
                          gps_jitter_summary=None):
    filepath = os.path.join(output_dir, "recording_info.txt")
    with open(filepath, "a") as f:
        f.write("\n")
        f.write("Recording Complete\n")
        f.write("==================\n")
        f.write(f"End time:       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"Duration:       {duration:.1f} seconds\n")
        f.write(f"\n")
        f.write("IMU CRC:\n")
        f.write(f"  CRC pass:     {crc_pass}\n")
        f.write(f"  CRC fail:     {crc_fail}\n")
        if crc_pass + crc_fail > 0:
            f.write(f"  CRC rate:     {crc_pass / (crc_pass + crc_fail) * 100:.2f}%\n")
        f.write(f"\n")
        f.write("IMU RAW Logger:\n")
        f.write(f"  Rows written: {imu_raw_stats['written']}\n")
        f.write(f"  Rows dropped: {imu_raw_stats['dropped']}\n")
        f.write(f"  Rate:         {imu_raw_stats['written'] / max(duration, 0.1):.1f} rows/s\n")
        f.write(f"\n")
        f.write("IMU QUAT Logger:\n")
        f.write(f"  Rows written: {imu_quat_stats['written']}\n")
        f.write(f"  Rows dropped: {imu_quat_stats['dropped']}\n")
        f.write(f"\n")
        f.write("IMU RPY Logger:\n")
        f.write(f"  Rows written: {imu_rpy_stats['written']}\n")
        f.write(f"  Rows dropped: {imu_rpy_stats['dropped']}\n")
        f.write(f"\n")
        f.write("IMU GRAVITY Logger:\n")
        f.write(f"  Rows written: {imu_gravity_stats['written']}\n")
        f.write(f"  Rows dropped: {imu_gravity_stats['dropped']}\n")
        f.write(f"\n")
        f.write("GPS Logger:\n")
        f.write(f"  Rows written: {gps_stats['written']}\n")
        f.write(f"  Rows dropped: {gps_stats['dropped']}\n")
        f.write(f"  Rate:         {gps_stats['written'] / max(duration, 0.1):.1f} rows/s\n")
        f.write(f"\n")
        if imu_thread_stats:
            f.write("IMU Reader Thread:\n")
            f.write(f"  Packets read:   {imu_thread_stats['read']}\n")
            f.write(f"  Packets queued: {imu_thread_stats['queued']}\n")
            lost = imu_thread_stats['read'] - imu_thread_stats['queued']
            f.write(f"  Packets lost:   {lost}\n")
            f.write(f"\n")
        if gps_thread_stats:
            f.write("GPS Reader Thread:\n")
            f.write(f"  Lines read:     {gps_thread_stats['read']}\n")
            f.write(f"  Lines queued:   {gps_thread_stats['queued']}\n")
            lost = gps_thread_stats['read'] - gps_thread_stats['queued']
            f.write(f"  Lines lost:     {lost}\n")
            f.write(f"\n")

        if gps_clock_summary:
            f.write("GPS UTC <-> Wall Clock Calibration:\n")
            f.write(f"  RMC samples:    {gps_clock_summary.get('n_samples', 0)}\n")
            f.write(f"  Updates:        {gps_clock_summary.get('n_updates', 0)}\n")
            offset_s = gps_clock_summary.get("offset_s")
            drift_ppm = gps_clock_summary.get("drift_ppm")
            f.write("  Offset s:       "
                    f"{offset_s:.9f}\n" if offset_s is not None
                    else "  Offset s:       unavailable\n")
            f.write("  Drift ppm:      "
                    f"{drift_ppm:.6f}\n" if drift_ppm is not None
                    else "  Drift ppm:      unavailable\n")
            f.write(f"\n")

        if gps_jitter_summary:
            f.write("GPS Arrival Jitter:\n")
            for name in ("GGA", "RMC"):
                item = gps_jitter_summary.get(name)
                if not item:
                    continue
                f.write(f"  {name}:\n")
                f.write(f"    samples:             {item['n_samples']}\n")
                f.write(f"    nominal interval ms: {item['nominal_interval_ms']:.3f}\n")
                f.write(f"    jitter std ms:       {item['jitter_std_ms']:.3f}\n")
                f.write(f"    jitter max ms:       {item['jitter_max_ms']:.3f}\n")
            f.write(f"\n")

        total_drops = (imu_raw_stats['dropped'] + imu_quat_stats['dropped']
                      + imu_rpy_stats['dropped'] + imu_gravity_stats['dropped']
                      + gps_stats['dropped'])

        if total_drops == 0 and crc_fail == 0:
            f.write("★ RECORDING COMPLETE — ZERO DROPS, ZERO CRC ERRORS ★\n")
        elif total_drops == 0:
            f.write(f"★ ZERO DROPS — {crc_fail} bad CRC packets discarded ★\n")
        else:
            f.write(f"⚠ WARNING: {total_drops} CSV drops + {crc_fail} CRC failures ⚠\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    stop_event = threading.Event()

    imu_ser = None
    gps_ser = None
    imu_thread = None
    gps_thread = None

    event_q = queue.Queue(maxsize=EVENT_QUEUE_SIZE)

    # ── Open IMU ──
    if IMU_ENABLED:
        try:
            imu_ser = serial.Serial(IMU_PORT, IMU_BAUD, timeout=0.01)
            enlarge_serial_buffer(imu_ser)
            print(f"Opened IMU: {IMU_PORT} @ {IMU_BAUD}")
            clear_serial_input_buffer(imu_ser, "IMU")
            temp_reader = IMUPacketReader(imu_ser)
            if not validate_imu_connection(temp_reader):
                print("[WARNING] IMU not responding. Disabling.")
                imu_ser.close()
                imu_ser = None
            else:
                clear_serial_input_buffer(imu_ser, "IMU")
        except serial.SerialException as exc:
            print(f"[WARNING] Could not open IMU: {exc}")

    # ── Open GPS ──
    if GPS_ENABLED:
        try:
            gps_ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=0.1)
            enlarge_serial_buffer(gps_ser)
            print(f"Opened GPS: {GPS_PORT} @ {GPS_BAUD}")
            clear_serial_input_buffer(gps_ser, "GPS")
            if not validate_gps_connection(gps_ser):
                print("[WARNING] GPS not responding. Disabling.")
                gps_ser.close()
                gps_ser = None
            else:
                clear_serial_input_buffer(gps_ser, "GPS")
        except serial.SerialException as exc:
            print(f"[WARNING] Could not open GPS: {exc}")

    if imu_ser is None and gps_ser is None:
        print("[ERROR] No sensors available. Exiting.")
        return

    # ── Start reader threads ──
    if imu_ser is not None:
        imu_thread = IMUReaderThread(imu_ser, event_q, stop_event)
        imu_thread.start()
        print("[THREAD] IMU reader started → event_q (blocking put)")

    if gps_ser is not None:
        gps_thread = GPSReaderThread(gps_ser, event_q, stop_event)
        gps_thread.start()
        print("[THREAD] GPS reader started → event_q (blocking put)")

    warmup_sensor_threads(event_q, WARMUP_S)

    record_start_time = wall_clock()
    write_recording_info(
        OUTPUT_DIR, IMU_PORT, GPS_PORT,
        IMU_BAUD, GPS_BAUD, record_start_time)

    print()
    print("=" * 60)
    print("  RECORDING STARTED")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # ── Main recording loop ──
    imu_count = 0
    gps_count = 0
    imu_raw_count = 0
    imu_quat_count = 0
    imu_rpy_count = 0
    imu_gravity_count = 0
    gps_gga_count = 0
    gps_gga_valid_count = 0
    gps_gst_count = 0
    gps_hdt_count = 0
    gps_rmc_count = 0
    gps_vtg_count = 0
    gps_other_count = 0
    loop_counter = 0
    rate_window_start = wall_clock()
    last_status_time = wall_clock()
    last_imu_wt = None
    last_imu_ts = None
    gps_clock_calibrator = GpsWallClockCalibrator()
    gps_jitter_tracker = GpsArrivalJitterTracker()

    try:
        while True:
            loop_counter += 1

            if RECORD_DURATION is not None:
                elapsed = wall_clock() - record_start_time
                if elapsed >= RECORD_DURATION:
                    nb_print(f"\n[DONE] Recording duration "
                             f"{RECORD_DURATION:.0f}s reached.")
                    break

            # ── Drain ALL events in arrival order ──
            did_work = False
            while True:
                try:
                    wt, evt_type, payload = event_q.get_nowait()
                except queue.Empty:
                    break

                did_work = True

                # ─── IMU EVENT ───
                if evt_type == EVENT_IMU:
                    imu_timestamp, imu_data = payload
                    imu_count += 1
                    last_imu_wt = wt
                    last_imu_ts = imu_timestamp
                    ptype = imu_data.get("packet_type", "")

                    if ptype == "RAW":
                        imu_raw_count += 1
                        mag_x = imu_data.get("mag_x")
                        mag_y = imu_data.get("mag_y")
                        mag_z = imu_data.get("mag_z")
                        imu_raw_logger.write_row([
                            f"{wt:.6f}",
                            imu_timestamp,
                            f"{imu_data['gyro_x']:.8f}",
                            f"{imu_data['gyro_y']:.8f}",
                            f"{imu_data['gyro_z']:.8f}",
                            f"{imu_data['accel_x']:.8f}",
                            f"{imu_data['accel_y']:.8f}",
                            f"{imu_data['accel_z']:.8f}",
                            f"{mag_x:.8f}" if mag_x is not None else "",
                            f"{mag_y:.8f}" if mag_y is not None else "",
                            f"{mag_z:.8f}" if mag_z is not None else "",
                        ])

                    elif ptype == "QUAT":
                        imu_quat_count += 1
                        imu_quat_logger.write_row([
                            f"{wt:.6f}",
                            imu_timestamp,
                            f"{imu_data['q1']:.8f}",
                            f"{imu_data['q2']:.8f}",
                            f"{imu_data['q3']:.8f}",
                            f"{imu_data['q4']:.8f}",
                        ])

                    elif ptype == "RPY":
                        imu_rpy_count += 1
                        imu_rpy_logger.write_row([
                            f"{wt:.6f}",
                            imu_timestamp,
                            f"{imu_data['roll']:.8f}",
                            f"{imu_data['pitch']:.8f}",
                            f"{imu_data['yaw']:.8f}",
                        ])

                    elif ptype == "GRAVITY":
                        imu_gravity_count += 1
                        imu_gravity_logger.write_row([
                            f"{wt:.6f}",
                            imu_timestamp,
                            f"{imu_data['gravity_x']:.8f}",
                            f"{imu_data['gravity_y']:.8f}",
                            f"{imu_data['gravity_z']:.8f}",
                        ])

                # ─── GPS EVENT ───
                elif evt_type == EVENT_GPS:
                    raw_line = payload
                    gps_count += 1

                    result = parse_and_log_gps_line(
                        raw_line, wt, last_imu_wt, last_imu_ts)

                    if result is not None:
                        rtype = result.get("type", "")
                        if rtype == "GGA":
                            gps_gga_count += 1
                            gps_jitter_tracker.on_gga(wt)
                            if result.get("valid"):
                                gps_gga_valid_count += 1
                        elif rtype == "GST":
                            gps_gst_count += 1
                        elif rtype == "HDT":
                            gps_hdt_count += 1
                        elif rtype == "RMC":
                            gps_rmc_count += 1
                            gps_jitter_tracker.on_rmc(wt)
                            if result.get("gps_utc_type") == "epoch":
                                gps_clock_calibrator.add_sample(
                                    wt, result.get("gps_utc"))
                        elif rtype == "VTG":
                            gps_vtg_count += 1
                        else:
                            gps_other_count += 1

            # ── Status report every 2 seconds ──
            now = wall_clock()
            if now - last_status_time >= 2.0:
                elapsed_total = now - record_start_time
                elapsed_window = now - rate_window_start

                imu_rate = imu_count / elapsed_window if elapsed_window > 0 else 0
                gps_rate = gps_count / elapsed_window if elapsed_window > 0 else 0

                crc_pass = imu_thread.reader.crc_pass_count if imu_thread else 0
                crc_fail = imu_thread.reader.crc_fail_count if imu_thread else 0

                imu_csv_drops = (imu_raw_logger.stats["dropped"]
                                + imu_quat_logger.stats["dropped"]
                                + imu_rpy_logger.stats["dropped"]
                                + imu_gravity_logger.stats["dropped"])
                gps_csv_drops = gps_logger.stats["dropped"]
                total_drops = imu_csv_drops + gps_csv_drops

                drop_str = ("✓ ZERO DROPS" if total_drops == 0
                           else f"⚠ {total_drops} DROPS")
                crc_str = (f"CRC ok" if crc_fail == 0
                          else f"CRC fail={crc_fail}")

                parts = [
                    f"[REC {elapsed_total:.0f}s]",
                    f"imu={imu_rate:.0f}/s "
                    f"(R:{imu_raw_count} Q:{imu_quat_count} "
                    f"P:{imu_rpy_count} G:{imu_gravity_count})",
                    f"gps={gps_rate:.0f}/s "
                    f"(GGA:{gps_gga_count} v:{gps_gga_valid_count} "
                    f"GST:{gps_gst_count} HDT:{gps_hdt_count} "
                    f"RMC:{gps_rmc_count} VTG:{gps_vtg_count})",
                    f"eq={event_q.qsize()}",
                    crc_str,
                    drop_str,
                ]
                nb_print(" | ".join(parts))

                imu_count = 0
                gps_count = 0
                loop_counter = 0
                rate_window_start = now
                last_status_time = now

            if not did_work:
                time.sleep(0.0002)

    except KeyboardInterrupt:
        nb_print("\n[STOPPING] Ctrl+C received...")

    # ── Shutdown ──
    record_end_time = wall_clock()
    record_duration = record_end_time - record_start_time

    nb_print(f"\nRecording stopped after {record_duration:.1f} seconds")

    stop_event.set()

    imu_thread_stats = None
    gps_thread_stats = None
    crc_pass = 0
    crc_fail = 0

    if imu_thread is not None:
        imu_thread.join(timeout=2.0)
        crc_pass = imu_thread.reader.crc_pass_count
        crc_fail = imu_thread.reader.crc_fail_count
        imu_thread_stats = {
            "read": imu_thread.packets_read,
            "queued": imu_thread.packets_queued,
        }
        nb_print(f"[IMU THREAD] read={imu_thread.packets_read} "
                 f"queued={imu_thread.packets_queued} "
                 f"CRC pass={crc_pass} fail={crc_fail}")

    if gps_thread is not None:
        gps_thread.join(timeout=2.0)
        gps_thread_stats = {
            "read": gps_thread.lines_read,
            "queued": gps_thread.lines_queued,
        }
        nb_print(f"[GPS THREAD] read={gps_thread.lines_read} "
                 f"queued={gps_thread.lines_queued}")

    imu_raw_logger.close()
    imu_quat_logger.close()
    imu_rpy_logger.close()
    imu_gravity_logger.close()
    gps_logger.close()

    imu_raw_stats = imu_raw_logger.stats
    imu_quat_stats = imu_quat_logger.stats
    imu_rpy_stats = imu_rpy_logger.stats
    imu_gravity_stats = imu_gravity_logger.stats
    gps_stats = gps_logger.stats

    nb_print(f"\n[LOG] imu_raw.csv:     {imu_raw_stats['written']} rows, "
             f"{imu_raw_stats['dropped']} dropped")
    nb_print(f"[LOG] imu_quat.csv:    {imu_quat_stats['written']} rows, "
             f"{imu_quat_stats['dropped']} dropped")
    nb_print(f"[LOG] imu_rpy.csv:     {imu_rpy_stats['written']} rows, "
             f"{imu_rpy_stats['dropped']} dropped")
    nb_print(f"[LOG] imu_gravity.csv: {imu_gravity_stats['written']} rows, "
             f"{imu_gravity_stats['dropped']} dropped")
    nb_print(f"[LOG] gps_raw.csv:     {gps_stats['written']} rows, "
             f"{gps_stats['dropped']} dropped")

    update_recording_info(
        OUTPUT_DIR, record_end_time, record_duration,
        imu_raw_stats, imu_quat_stats,
        imu_rpy_stats, imu_gravity_stats,
        gps_stats,
        imu_thread_stats, gps_thread_stats,
        crc_pass, crc_fail,
        gps_clock_calibrator.summary(),
        gps_jitter_tracker.summary())

    total_drops = (imu_raw_stats["dropped"] + imu_quat_stats["dropped"]
                  + imu_rpy_stats["dropped"] + imu_gravity_stats["dropped"]
                  + gps_stats["dropped"])

    nb_print("")
    nb_print("=" * 60)
    if total_drops == 0 and crc_fail == 0:
        nb_print("  ★ RECORDING COMPLETE — ZERO DROPS, ZERO CRC ERRORS ★")
    elif total_drops == 0:
        nb_print(f"  ★ ZERO DROPS — {crc_fail} bad CRC packets discarded ★")
    else:
        nb_print(f"  ⚠ {total_drops} CSV DROPS + {crc_fail} CRC FAILURES ⚠")
    nb_print(f"  Duration:      {record_duration:.1f} seconds")
    nb_print(f"  IMU RAW:       {imu_raw_stats['written']} rows")
    nb_print(f"  IMU QUAT:      {imu_quat_stats['written']} rows")
    nb_print(f"  IMU RPY:       {imu_rpy_stats['written']} rows")
    nb_print(f"  IMU GRAVITY:   {imu_gravity_stats['written']} rows")
    nb_print(f"  GPS:           {gps_stats['written']} rows")
    nb_print(f"  CRC:           {crc_pass} pass / {crc_fail} fail")
    nb_print(f"  Files:         {os.path.abspath(OUTPUT_DIR)}/")
    nb_print("=" * 60)

    if imu_ser is not None:
        imu_ser.close()
    if gps_ser is not None:
        gps_ser.close()

    time.sleep(0.2)
    stop_print_worker()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nForce stopped.")
    finally:
        try:
            imu_raw_logger.close()
            imu_quat_logger.close()
            imu_rpy_logger.close()
            imu_gravity_logger.close()
            gps_logger.close()
            stop_print_worker()
        except Exception:
            pass
