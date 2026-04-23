# -*- coding: utf-8 -*-
"""
GPS/IMU Sensor Fusion — GPS-Anchored Dead Reckoning
=====================================================
Algorithm:
  1. GPS point A arrives → set as current position (ground truth)
  2. IMU integrates orientation + velocity + position from A
  3. GPS point B arrives → HARD RESET position to B (ground truth)
  4. Repeat from B

Hardware:
  - SYD Dynamics TransducerM TM151 AHRS 9-Axis IMU at 400 Hz
    Binary EasyProfile protocol
    Accel: gravity-free linear acceleration (near zero when stationary)
    Gyro: rad/s, Timestamp: microseconds
  - GPS at ~10 Hz (NMEA: GGA, GST, HDT)

Threading model:
  - IMU reader thread:  serial → shared event_queue
  - GPS reader thread:  serial → shared event_queue
  - Log writer thread:  log_queue → disk (non-blocking for main thread)
  - Main thread:        event_queue → dead reckoning → log_queue

Logging:
  - position.csv:  GPS fixes + decimated DR predictions
  - gps_raw.csv:   every valid GPS fix with quality info
  - imu_raw.csv:   every IMU RAW packet (optional: --log-imu-raw)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
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


# ─────────────────────────────────────────────
# HIGH-RESOLUTION TIMER
# ─────────────────────────────────────────────
# time.perf_counter() has sub-microsecond resolution on all platforms.
# We combine it with time.time() for absolute timestamps.

_PERF_EPOCH_OFFSET = time.time() - time.perf_counter()


def wall_clock():
    """
    High-resolution wall clock.
    Uses perf_counter for resolution, anchored to time.time() epoch.
    Sub-microsecond resolution on all platforms (vs 15ms for time.time on Windows).
    """
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
    """Non-blocking print. Drops message if queue full."""
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
# FLAT-EARTH COORDINATE CONVERSION
# ─────────────────────────────────────────────

_M_PER_DEG_LAT = 111_132.92
_M_PER_DEG_LON_EQ = 111_319.49


def _m_per_deg_lon(lat_deg):
    return _M_PER_DEG_LON_EQ * math.cos(math.radians(lat_deg))


def gps_to_enu(lat, lon, alt, anchor_lat, anchor_lon, anchor_alt):
    east = (lon - anchor_lon) * _m_per_deg_lon(anchor_lat)
    north = (lat - anchor_lat) * _M_PER_DEG_LAT
    up = alt - anchor_alt
    return east, north, up


def enu_to_gps(east, north, up, anchor_lat, anchor_lon, anchor_alt):
    lat = anchor_lat + north / _M_PER_DEG_LAT
    lon = anchor_lon + east / _m_per_deg_lon(anchor_lat)
    alt = anchor_alt + up
    return lat, lon, alt


def flat_earth_distance(lat1, lon1, lat2, lon2):
    mid_lat = (lat1 + lat2) * 0.5
    dn = (lat2 - lat1) * _M_PER_DEG_LAT
    de = (lon2 - lon1) * _m_per_deg_lon(mid_lat)
    return math.sqrt(dn * dn + de * de)


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
        description="GPS/IMU Dead Reckoning with GPS Reset"
    )
    parser.add_argument("--imu-port", type=str, default=None)
    parser.add_argument("--gps-port", type=str, default=None)
    parser.add_argument("--imu-baud", type=int, default=None)
    parser.add_argument("--gps-baud", type=int, default=None)
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--no-gps", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--log-imu-raw", action="store_true",
                        help="Log every IMU packet to imu_raw.csv (~50MB/hr)")
    parser.add_argument("--dr-log-decimation", type=int, default=10,
                        help="Log every Nth DR prediction to position.csv "
                             "(default=10 → 40Hz from 400Hz IMU)")
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
LOG_IMU_RAW = _args.log_imu_raw
DR_LOG_DECIMATION = max(1, _args.dr_log_decimation)

IMU_PORT, GPS_PORT = resolve_ports(OS_NAME, _args)

# ─── IMU settings ───
IMU_BAUD = _args.imu_baud if _args.imu_baud else 115200
MAX_VALID_IMU_DT = 0.005

# ─── GPS settings ───
GPS_BAUD = _args.gps_baud if _args.gps_baud else 115200
MIN_FIX_QUALITY = 1
MAX_RAW_GPS_JUMP_M = 15.0

# ─── Serial buffer ───
SERIAL_BUFFER_SIZE = 65536

# ─── Queue sizes ───
EVENT_QUEUE_SIZE = 4200
LOG_QUEUE_SIZE = 2000

# ─── Debug ───
PRINT_RAW_GPS_DEBUG = True
PRINT_ALL_NMEA_LINES = False

# ─── GPS dropout ───
GPS_DROPOUT_WARN_S = 2.0    # warn after 2s without GPS
GPS_DROPOUT_CLAMP_S = 10.0  # clamp velocity to zero after 10s

# ─── IMU protocol ───
IMU_SYNC_BYTE_1 = 0xAA
IMU_SYNC_BYTE_2 = 0x55
IMU_CMD_RAW = 41
IMU_CMD_QUAT = 32
IMU_CMD_RPY = 35
IMU_CRC_BYTES = 2

# ─── GPS timeout ───
GPS_FIX_TIMEOUT_S = 30

print(f"OS: {OS_NAME}")
print(f"IMU: {'ENABLED' if IMU_ENABLED else 'DISABLED'} | {IMU_PORT} @ {IMU_BAUD}")
print(f"GPS: {'ENABLED' if GPS_ENABLED else 'DISABLED'} | {GPS_PORT} @ {GPS_BAUD}")
print(f"Log IMU raw: {LOG_IMU_RAW}")
print(f"DR log decimation: 1/{DR_LOG_DECIMATION} "
      f"(~{400 / DR_LOG_DECIMATION:.0f} Hz to CSV)")


# ─────────────────────────────────────────────
# QUATERNION UTILITIES
# ─────────────────────────────────────────────

def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


def quaternion_to_yaw_deg(q):
    q = normalize_quaternion(q)
    r = R.from_quat(q)
    yaw_rad = r.as_euler("zyx", degrees=False)[0]
    heading_deg = (90.0 - np.degrees(yaw_rad)) % 360.0
    return heading_deg


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
# ASYNC CSV LOGGER (writes on dedicated thread)
# ─────────────────────────────────────────────

class AsyncCSVLogger:
    """
    Thread-safe CSV logger. write_row() is non-blocking:
    it pushes to an internal queue. A background thread
    drains the queue to disk.
    """

    def __init__(self, filename, headers, q_size=5000):
        self.filename = filename
        self.headers = headers
        self._q = queue.Queue(maxsize=q_size)
        self._stop = threading.Event()
        self._rows_written = 0
        self._rows_dropped = 0

        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name=f"Logger-{filename}",
        )
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
                # Block up to 50ms for first item
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

            # Drain remaining without blocking
            while len(batch) < 500:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break

            for b in batch:
                writer.writerow(b)
            self._rows_written += len(batch)
            batch.clear()

            if self._rows_written % 5000 < 500:
                f.flush()

        # Final drain
        while not self._q.empty():
            try:
                writer.writerow(self._q.get_nowait())
                self._rows_written += 1
            except queue.Empty:
                break
        f.flush()
        f.close()

    def write_row(self, row):
        """Non-blocking. Drops row if queue is full."""
        try:
            self._q.put_nowait(row)
        except queue.Full:
            self._rows_dropped += 1

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

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

pos_logger = AsyncCSVLogger("position.csv", [
    "time", "source", "lat", "lon", "alt",
    "ve", "vn", "vu",
    "yaw", "pitch", "roll",
    "drift_m", "imu_dt", "gps_age",
])

gps_logger = AsyncCSVLogger("gps_raw.csv", [
    "time", "lat", "lon", "alt",
    "fix_quality", "num_sats", "hdop",
    "drift_at_reset_m",
    "vel_e", "vel_n", "vel_u",
], q_size=500)

imu_raw_logger = None
if LOG_IMU_RAW:
    imu_raw_logger = AsyncCSVLogger("imu_raw.csv", [
        "time", "imu_ts", "dt",
        "ax", "ay", "az",
        "gx", "gy", "gz",
    ], q_size=5000)


# ─────────────────────────────────────────────
# IMU PACKET READER (used by thread)
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
# GPS LINE PARSER (used by thread)
# ─────────────────────────────────────────────

def parse_gps_line(line):
    if not line:
        return None

    if PRINT_ALL_NMEA_LINES:
        nb_print(f"[NMEA] {line}")

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
                nb_print(
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
# EVENT TYPES
# ─────────────────────────────────────────────

EVENT_IMU = "imu"
EVENT_GPS = "gps"


# ─────────────────────────────────────────────
# THREADED SERIAL READERS → SINGLE QUEUE
# ─────────────────────────────────────────────

class IMUReaderThread(threading.Thread):
    def __init__(self, ser, event_q, stop_event):
        super().__init__(daemon=True, name="IMUReader")
        self.reader = IMUPacketReader(ser)
        self.event_q = event_q
        self.stop_event = stop_event
        self.packets_read = 0
        self.packets_dropped = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                ts, data = self.reader.read_packet()
                if data is not None:
                    self.packets_read += 1
                    wall = wall_clock()
                    try:
                        self.event_q.put_nowait(
                            (wall, EVENT_IMU, (ts, data)))
                    except queue.Full:
                        try:
                            self.event_q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.event_q.put_nowait(
                                (wall, EVENT_IMU, (ts, data)))
                        except queue.Full:
                            pass
                        self.packets_dropped += 1
                else:
                    time.sleep(0.0001)
            except serial.SerialException:
                if not self.stop_event.is_set():
                    print("[IMU THREAD] Serial error")
                break
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[IMU THREAD] Error: {exc}")
                time.sleep(0.001)


class GPSReaderThread(threading.Thread):
    def __init__(self, ser, event_q, stop_event):
        super().__init__(daemon=True, name="GPSReader")
        self.ser = ser
        self.event_q = event_q
        self.stop_event = stop_event
        self.lines_read = 0
        self.lines_dropped = 0

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

                parsed = parse_gps_line(line)
                if parsed is not None:
                    self.lines_read += 1
                    wall = wall_clock()
                    try:
                        self.event_q.put_nowait(
                            (wall, EVENT_GPS, parsed))
                    except queue.Full:
                        try:
                            self.event_q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.event_q.put_nowait(
                                (wall, EVENT_GPS, parsed))
                        except queue.Full:
                            pass
                        self.lines_dropped += 1

            except serial.SerialException:
                if not self.stop_event.is_set():
                    print("[GPS THREAD] Serial error")
                break
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[GPS THREAD] Error: {exc}")
                time.sleep(0.01)


# ─────────────────────────────────────────────
# DEAD RECKONING ENGINE (GPS-ANCHORED)
# ─────────────────────────────────────────────

class DeadReckoningEngine:
    def __init__(self):
        self.anchor_lat = None
        self.anchor_lon = None
        self.anchor_alt = None
        self.pos_local = np.zeros(3)
        self.vel_local = np.zeros(3)

        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self._orientation_initialized = False

        self.last_imu_timestamp_us = None
        self.last_imu_dt = 0.0

        self.last_gps_time = None

        self.predict_counter = 0
        self.skipped_predicts = 0
        self.gps_update_count = 0

        self.has_gps = False
        self.has_imu = False

        # GPS dropout tracking
        self._gps_dropout_warned = False
        self._gps_dropout_clamped = False

    def update_gps(self, lat, lon, alt, wall_time):
        if alt is None:
            alt = 0.0 if self.anchor_alt is None else self.anchor_alt

        if self.anchor_lat is not None and self.last_gps_time is not None:
            dt_gps = wall_time - self.last_gps_time
            if 0.01 < dt_gps < 2.0:
                d_east, d_north, d_up = gps_to_enu(
                    lat, lon, alt,
                    self.anchor_lat, self.anchor_lon, self.anchor_alt
                )
                self.vel_local = np.array([
                    d_east / dt_gps,
                    d_north / dt_gps,
                    d_up / dt_gps,
                ])
        else:
            self.vel_local = np.zeros(3)

        self.anchor_lat = lat
        self.anchor_lon = lon
        self.anchor_alt = alt
        self.pos_local = np.zeros(3)
        self.last_gps_time = wall_time

        self.has_gps = True
        self.gps_update_count += 1

        # Clear dropout state
        self._gps_dropout_warned = False
        self._gps_dropout_clamped = False

    def update_imu(self, gyro, accel, timestamp_us):
        if not self._orientation_initialized:
            self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
            self._orientation_initialized = True
            self.has_imu = True
            self.last_imu_timestamp_us = timestamp_us
            nb_print("[INIT] Gravity-free accel. Identity orientation.")
            return False

        if self.last_imu_timestamp_us is None:
            self.last_imu_timestamp_us = timestamp_us
            return False

        raw_diff = timestamp_us - self.last_imu_timestamp_us
        if raw_diff < 0:
            raw_diff += 2 ** 32
        dt = raw_diff / 1e6
        self.last_imu_timestamp_us = timestamp_us

        if dt <= 0 or dt > MAX_VALID_IMU_DT:
            self.skipped_predicts += 1
            return False

        self.last_imu_dt = dt

        # ── GPS dropout handling ──
        if self.last_gps_time is not None:
            gps_age = wall_clock() - self.last_gps_time

            if (gps_age > GPS_DROPOUT_WARN_S
                    and not self._gps_dropout_warned):
                self._gps_dropout_warned = True
                nb_print(f"[GPS DROPOUT] No GPS for {gps_age:.1f}s, "
                      f"drift={self.get_drift_m():.2f}m")

            if (gps_age > GPS_DROPOUT_CLAMP_S
                    and not self._gps_dropout_clamped):
                self._gps_dropout_clamped = True
                self.vel_local = np.zeros(3)
                nb_print(f"[GPS DROPOUT] No GPS for {gps_age:.1f}s, "
                      f"velocity clamped to zero. "
                      f"drift={self.get_drift_m():.2f}m")

        # ── 1. Update orientation ──
        angle = np.linalg.norm(gyro) * dt
        if angle > 1e-12:
            r_current = R.from_quat(self.orientation)
            delta_r = R.from_rotvec(gyro * dt)
            r_new = r_current * delta_r
            self.orientation = r_new.as_quat()
            self.orientation = normalize_quaternion(self.orientation)

        # ── 2. Rotate accel to ENU ──
        rot = R.from_quat(self.orientation)
        accel_enu = rot.apply(accel)

        # ── 3. Integrate ──
        if not self._gps_dropout_clamped:
            self.pos_local += (self.vel_local * dt
                              + 0.5 * accel_enu * dt ** 2)
            self.vel_local += accel_enu * dt
        # If clamped: still update orientation but freeze position

        self.predict_counter += 1
        return True

    def update_orientation_from_quat(self, q):
        self.orientation = normalize_quaternion(q)
        if not self._orientation_initialized:
            self._orientation_initialized = True
            self.has_imu = True
            nb_print("[INIT] Orientation set from QUAT packet.")

    def update_heading(self, heading_deg):
        if heading_deg is None:
            return
        r = R.from_quat(self.orientation)
        euler = r.as_euler('zyx', degrees=True)
        new_yaw_deg = (90.0 - heading_deg) % 360.0
        if new_yaw_deg > 180.0:
            new_yaw_deg -= 360.0
        euler[0] = new_yaw_deg
        r_new = R.from_euler('zyx', euler, degrees=True)
        self.orientation = normalize_quaternion(r_new.as_quat())

    def get_gps_position(self):
        if self.anchor_lat is None:
            return None
        e, n, u = self.pos_local
        if np.linalg.norm([e, n]) > 1000.0 or abs(u) > 200.0:
            return None
        return enu_to_gps(
            e, n, u,
            self.anchor_lat, self.anchor_lon, self.anchor_alt
        )

    def get_drift_m(self):
        return float(np.linalg.norm(self.pos_local[:2]))

    def get_position_local(self):
        return self.pos_local.copy()

    def get_velocity(self):
        return self.vel_local.copy()

    def get_yaw_deg(self):
        return quaternion_to_yaw_deg(self.orientation)

    def get_euler_deg(self):
        r = R.from_quat(normalize_quaternion(self.orientation))
        return r.as_euler('zyx', degrees=True)

    def gps_age_seconds(self):
        if self.last_gps_time is None:
            return None
        return wall_clock() - self.last_gps_time

    def get_mode_string(self):
        if self._gps_dropout_clamped:
            return "GPS DROPOUT (frozen)"
        if self._gps_dropout_warned:
            return "GPS-anchored DR (GPS stale)"
        if self.has_gps and self.has_imu:
            return "GPS-anchored dead reckoning"
        elif self.has_imu:
            return "IMU only (relative)"
        elif self.has_gps:
            return "GPS only (no prediction)"
        else:
            return "no sensors"


# ─────────────────────────────────────────────
# SERIAL BUFFER ENLARGEMENT
# ─────────────────────────────────────────────

def enlarge_serial_buffer(ser, size=SERIAL_BUFFER_SIZE):
    try:
        ser.set_buffer_size(rx_size=size)
        print(f"  Serial RX buffer set to {size} bytes")
    except (AttributeError, serial.SerialException, OSError):
        print(f"  Serial RX buffer resize not supported (OS manages it)")


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
# LOGGING HELPERS
# ─────────────────────────────────────────────

def log_position_row(dr, source, wt):
    gps_pos = dr.get_gps_position()
    if gps_pos is None:
        return

    vel = dr.get_velocity()
    euler = dr.get_euler_deg()
    yaw = dr.get_yaw_deg()
    drift = dr.get_drift_m()
    gps_age = dr.gps_age_seconds()

    pos_logger.write_row([
        f"{wt:.6f}",
        source,
        f"{gps_pos[0]:.10f}",
        f"{gps_pos[1]:.10f}",
        f"{gps_pos[2]:.4f}",
        f"{vel[0]:.4f}",
        f"{vel[1]:.4f}",
        f"{vel[2]:.4f}",
        f"{yaw:.2f}",
        f"{euler[1]:.2f}",
        f"{euler[2]:.2f}",
        f"{drift:.4f}",
        f"{dr.last_imu_dt:.6f}",
        f"{gps_age:.3f}" if gps_age is not None else "",
    ])


def log_gps_reset(dr, gps_fix, drift_before, wt):
    vel = dr.get_velocity()
    gps_logger.write_row([
        f"{wt:.6f}",
        gps_fix["lat"],
        gps_fix["lon"],
        gps_fix.get("alt", ""),
        gps_fix.get("fix_quality", ""),
        gps_fix.get("num_sats", ""),
        gps_fix.get("hdop", ""),
        f"{drift_before:.4f}",
        f"{vel[0]:.4f}",
        f"{vel[1]:.4f}",
        f"{vel[2]:.4f}",
    ])


def log_imu_raw_row(wt, imu_ts, dt, accel, gyro):
    if imu_raw_logger is None:
        return
    imu_raw_logger.write_row([
        f"{wt:.6f}",
        imu_ts,
        f"{dt:.6f}",
        f"{accel[0]:.6f}",
        f"{accel[1]:.6f}",
        f"{accel[2]:.6f}",
        f"{gyro[0]:.6f}",
        f"{gyro[1]:.6f}",
        f"{gyro[2]:.6f}",
    ])


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    dr = DeadReckoningEngine()
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
            temp_reader = IMUPacketReader(imu_ser)
            if not validate_imu_connection(temp_reader):
                print("[WARNING] IMU not responding. Disabling.")
                imu_ser.close()
                imu_ser = None
        except serial.SerialException as exc:
            print(f"[WARNING] Could not open IMU: {exc}")

    # ── Open GPS ──
    if GPS_ENABLED:
        try:
            gps_ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=0.1)
            enlarge_serial_buffer(gps_ser)
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

    # ── Start reader threads ──
    if imu_ser is not None:
        imu_thread = IMUReaderThread(imu_ser, event_q, stop_event)
        imu_thread.start()
        print("[THREAD] IMU reader → event_q")

    if gps_ser is not None:
        gps_thread = GPSReaderThread(gps_ser, event_q, stop_event)
        gps_thread.start()
        print("[THREAD] GPS reader → event_q")

    # ─────────────────────────────────────
    # Phase 1: Wait for first GPS fix
    # ─────────────────────────────────────
    last_valid_raw_gps = None
    last_heading_deg = None

    if gps_ser is not None:
        print(f"Waiting for first GPS fix ({GPS_FIX_TIMEOUT_S}s)...")
        start_time = time.time()

        try:
            while (dr.anchor_lat is None
                   and time.time() - start_time < GPS_FIX_TIMEOUT_S):

                try:
                    wt, evt_type, payload = event_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                if evt_type == EVENT_IMU:
                    ts, data = payload
                    if data.get("packet_type") == "QUAT":
                        q = np.array([
                            data["q2"], data["q3"],
                            data["q4"], data["q1"],
                        ], dtype=float)
                        dr.update_orientation_from_quat(q)

                elif evt_type == EVENT_GPS:
                    gps_fix = payload
                    if gps_fix.get("type") == "HDT":
                        last_heading_deg = gps_fix.get("heading_deg")
                    elif (gps_fix.get("type") == "GGA"
                          and gps_fix.get("valid")):
                        lat = gps_fix["lat"]
                        lon = gps_fix["lon"]
                        alt = gps_fix.get("alt")
                        last_valid_raw_gps = gps_fix.copy()

                        if lat and lon and lat != 0.0 and lon != 0.0:
                            dr.update_gps(lat, lon, alt, wt)
                            log_position_row(dr, "gps", wt)
                            log_gps_reset(dr, gps_fix, 0.0, wt)
                            print(f"First GPS fix (anchor): "
                                  f"lat={lat:.8f} lon={lon:.8f} "
                                  f"alt={alt}")

        except KeyboardInterrupt:
            print("\nStopped during GPS wait.")
            stop_event.set()
            _close_all()
            return

        if dr.anchor_lat is None:
            print("[WARNING] No GPS fix obtained. IMU-only mode.")
    else:
        print("GPS disabled. IMU-only mode.")

    if imu_ser is None:
        print("IMU disabled. GPS-only mode.")

    print(f"Mode: {dr.get_mode_string()}")
    print("Starting fusion loop...")

    # ─────────────────────────────────────
    # Phase 2: Main loop
    # ─────────────────────────────────────
    imu_processed = 0
    imu_predictions_done = 0
    gps_processed = 0
    dr_log_counter = 0
    loop_counter = 0
    rate_window_start = wall_clock()
    last_predict_print_time = 0.0

    try:
        while True:
            loop_counter += 1

            # ─── Drain ALL events in arrival order ───
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
                    imu_processed += 1

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

                        predicted = dr.update_imu(
                            gyro, accel, imu_timestamp)

                        if predicted:
                            imu_predictions_done += 1
                            dr_log_counter += 1

                            # Decimated DR logging
                            if dr_log_counter >= DR_LOG_DECIMATION:
                                dr_log_counter = 0
                                log_position_row(dr, "dr", wt)

                            # IMU raw logging (optional)
                            log_imu_raw_row(
                                wt, imu_timestamp,
                                dr.last_imu_dt, accel, gyro)

                            # Console output (throttled)
                            now = wall_clock()
                            if now - last_predict_print_time > 0.25:
                                pos_local = dr.get_position_local()
                                vel = dr.get_velocity()
                                yaw = dr.get_yaw_deg()
                                gps_pos = dr.get_gps_position()

                                if gps_pos is not None:
                                    gps_age = dr.gps_age_seconds()
                                    age_str = (f"{gps_age:.2f}s"
                                               if gps_age else "?")
                                    nb_print(
                                        f"[DR] "
                                        f"lat={gps_pos[0]:.8f} "
                                        f"lon={gps_pos[1]:.8f} "
                                        f"alt={gps_pos[2]:.2f} "
                                        f"yaw={yaw:.1f} | "
                                        f"drift="
                                        f"{dr.get_drift_m():.3f}m "
                                        f"vel=("
                                        f"{vel[0]:.2f},"
                                        f"{vel[1]:.2f},"
                                        f"{vel[2]:.2f}) "
                                        f"gps_age={age_str}"
                                    )
                                else:
                                    euler = dr.get_euler_deg()
                                    nb_print(
                                        f"[IMU-ONLY] "
                                        f"pos=("
                                        f"{pos_local[0]:.3f},"
                                        f"{pos_local[1]:.3f},"
                                        f"{pos_local[2]:.3f}) m "
                                        f"vel=("
                                        f"{vel[0]:.3f},"
                                        f"{vel[1]:.3f},"
                                        f"{vel[2]:.3f}) m/s "
                                        f"yaw={yaw:.1f} "
                                        f"pitch={euler[1]:.1f} "
                                        f"roll={euler[2]:.1f}"
                                    )
                                last_predict_print_time = now

                    elif imu_data.get("packet_type") == "QUAT":
                        q = np.array([
                            imu_data["q2"], imu_data["q3"],
                            imu_data["q4"], imu_data["q1"],
                        ], dtype=float)
                        dr.update_orientation_from_quat(q)

                # ─── GPS EVENT ───
                elif evt_type == EVENT_GPS:
                    gps_fix = payload
                    gps_processed += 1
                    msg_type = gps_fix.get("type")

                    if msg_type == "GGA":
                        if gps_fix.get("valid"):
                            lat = gps_fix["lat"]
                            lon = gps_fix["lon"]
                            alt = gps_fix.get("alt")

                            if last_valid_raw_gps is not None:
                                try:
                                    jump_m = flat_earth_distance(
                                        last_valid_raw_gps["lat"],
                                        last_valid_raw_gps["lon"],
                                        lat, lon
                                    )
                                    if jump_m > MAX_RAW_GPS_JUMP_M:
                                        nb_print(
                                            f"[GGA REJECTED] "
                                            f"Jump {jump_m:.1f} m")
                                        continue
                                except Exception:
                                    pass

                            last_valid_raw_gps = gps_fix.copy()

                            drift_before = dr.get_drift_m()
                            dr.update_gps(lat, lon, alt, wt)

                            # Always log GPS resets
                            log_position_row(dr, "gps", wt)
                            log_gps_reset(
                                dr, gps_fix, drift_before, wt)

                            # Reset DR log counter so next
                            # DR sample after GPS is logged
                            dr_log_counter = DR_LOG_DECIMATION - 1

                            gps_pos = dr.get_gps_position()
                            if gps_pos is not None:
                                print(
                                    f"[GPS RESET] "
                                    f"lat={gps_pos[0]:.8f} "
                                    f"lon={gps_pos[1]:.8f} "
                                    f"alt={gps_pos[2]:.2f} "
                                    f"yaw={dr.get_yaw_deg():.1f} "
                                    f"drift_was="
                                    f"{drift_before:.3f}m "
                                    f"vel=("
                                    f"{dr.vel_local[0]:.2f},"
                                    f"{dr.vel_local[1]:.2f},"
                                    f"{dr.vel_local[2]:.2f})"
                                )

                    elif msg_type == "HDT":
                        last_heading_deg = gps_fix.get("heading_deg")
                        if last_heading_deg is not None:
                            dr.update_heading(last_heading_deg)

            # ─── Rate reporting ───
            now = wall_clock()
            if now - rate_window_start >= 2.0:
                elapsed = now - rate_window_start
                mode = dr.get_mode_string()
                parts = [f"[RATE] {mode}"]

                if imu_thread is not None:
                    parts.append(
                        f"imu={imu_processed / elapsed:.0f}/s")
                    parts.append(
                        f"pred={imu_predictions_done / elapsed:.0f}/s")
                    parts.append(
                        f"skip={dr.skipped_predicts}")
                    parts.append(
                        f"imu_drop={imu_thread.packets_dropped}")

                if gps_thread is not None:
                    parts.append(
                        f"gps={gps_processed / elapsed:.0f}/s")
                    parts.append(
                        f"gps_drop={gps_thread.lines_dropped}")
                    parts.append(
                        f"resets={dr.gps_update_count}")

                parts.append(f"eq={event_q.qsize()}")

                # Log writer health
                ps = pos_logger.stats
                parts.append(
                    f"csv_drop={ps['dropped']}")
                parts.append(
                    f"csv_pend={ps['pending']}")

                parts.append(
                    f"loops={loop_counter / elapsed:.0f}/s")

                nb_print(" | ".join(parts))

                loop_counter = 0
                imu_processed = 0
                imu_predictions_done = 0
                gps_processed = 0
                rate_window_start = now

            if not did_work:
                time.sleep(0.0002)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        _close_all()


def _close_all():
    if 'imu_thread' in dir() or True:
        try:
            # Give threads time to finish
            time.sleep(0.1)
        except Exception:
            pass

    pos_logger.close()
    gps_logger.close()
    if imu_raw_logger is not None:
        imu_raw_logger.close()

    # Print final stats
    ps = pos_logger.stats
    gs = gps_logger.stats
    print(f"[LOG] position.csv: {ps['written']} rows, "
          f"{ps['dropped']} dropped")
    print(f"[LOG] gps_raw.csv: {gs['written']} rows, "
          f"{gs['dropped']} dropped")
    if imu_raw_logger is not None:
        irs = imu_raw_logger.stats
        print(f"[LOG] imu_raw.csv: {irs['written']} rows, "
              f"{irs['dropped']} dropped")
    stop_print_worker()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nForce stopped.")
    finally:
        try:
            pos_logger.close()
            gps_logger.close()
            if imu_raw_logger is not None:
                imu_raw_logger.close()
        except Exception:
            pass