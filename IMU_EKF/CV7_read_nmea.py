#!/usr/bin/env python3
"""Read an already-configured CV7-INS and mosaic-H into one CSV file.

The CV7 side follows MicroStrain's official MSCL Python receive pattern:
``Connection.Serial`` -> ``InertialNode`` -> ``getDataPackets``.  This
recorder is deliberately read-only.  It never calls setToIdle(), resume(),
setActiveChannelFields(), enableDataStream(), saveSettings(), or any other
device/GNSS configuration method.

The mosaic-H raw monitor stream is read independently from its USB NMEA port.
The GPS-to-CV7 aiding link continues to operate outside this program.

Each CSV row is one received event.  ``source`` distinguishes CV7_IMU,
CV7_EKF, CV7_GNSS, CV7_OTHER, and GPS_NMEA rows.  This preserves the native
rates instead of duplicating a 10 Hz GPS sample over 500 Hz IMU/EKF rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import platform
import queue
import signal
import sys
import time
from datetime import date, datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

mscl = None
MSCL_IMPORT_DETAILS: List[str] = []


def _is_microstrain_mscl(candidate: Any) -> bool:
    """Reject unrelated PyPI modules which also happen to be named mscl."""
    return all(hasattr(candidate, name) for name in ("Connection", "InertialNode"))


try:
    import mscl as _top_level_mscl

    if _is_microstrain_mscl(_top_level_mscl):
        mscl = _top_level_mscl
    else:
        MSCL_IMPORT_DETAILS.append(
            "top-level 'mscl' is not MicroStrain MSCL: "
            f"{getattr(_top_level_mscl, '__file__', '<unknown path>')}"
        )
except ImportError as exc:
    MSCL_IMPORT_DETAILS.append(f"top-level 'mscl' unavailable: {exc}")

if mscl is None:
    try:
        # PyPI package ``python-mscl`` wraps the same MSCL binary under this
        # package namespace instead of installing a top-level ``mscl`` module.
        from python_mscl import mscl as _packaged_mscl

        if _is_microstrain_mscl(_packaged_mscl):
            mscl = _packaged_mscl
        else:
            MSCL_IMPORT_DETAILS.append(
                "'python_mscl.mscl' does not expose Connection/InertialNode: "
                f"{getattr(_packaged_mscl, '__file__', '<unknown path>')}"
            )
    except ImportError as exc:
        MSCL_IMPORT_DETAILS.append(f"'python_mscl' unavailable: {exc}")

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

try:
    from .cv7_gui_telemetry import (
        TelemetryState,
        UdpTelemetryPublisher,
        find_viewer_executable,
        launch_viewer,
    )
except ImportError:
    # Direct execution (``python IMU_EKF/CV7_read_nmea.py``) puts this file's
    # directory on sys.path, so use the sibling module without package syntax.
    from cv7_gui_telemetry import (
        TelemetryState,
        UdpTelemetryPublisher,
        find_viewer_executable,
        launch_viewer,
    )


DEFAULT_CV7_PORT = "auto"
DEFAULT_BAUD = 115200
SEPTENTRIO_USB_ID = (0x152A, 0x85C0)
GPS_NMEA_TYPES = {"GGA", "GST", "HDT", "RMC", "VTG", "ZDA", "GSV", "HRP"}
GPS_LIVE_TOLERANCE_S = 5.0
GPS_LIVE_FALLBACK_S = 3.0
EARTH_RADIUS_M = 6378137.0
EKF_PLOT_COLOR = "#1f77b4"
GPS_PLOT_COLOR = "#ff7f0e"

COMMON_COLUMNS = [
    "record_index",
    "host_unix_ns",
    "host_unix_s",
    "host_utc_iso",
    "elapsed_s",
    "source",
    "port",
    "baud",
    "descriptor_set",
    "packet_collected_ns",
    "mip_fields_json",
]

GPS_COLUMNS = [
    "gps_sentence_index",
    "gps_talker",
    "gps_message_type",
    "gps_checksum_present",
    "gps_checksum_ok",
    "gps_parsed_ok",
    "gps_parse_error",
    "gps_utc_time",
    "gps_utc_seconds_of_day",
    "gps_date",
    "gps_datetime_utc",
    "gps_latitude_deg",
    "gps_longitude_deg",
    "gps_altitude_msl_m",
    "gps_geoid_separation_m",
    "gps_ellipsoid_height_m",
    "gps_fix_quality",
    "gps_num_satellites",
    "gps_hdop",
    "gps_heading_true_deg",
    "gps_pitch_deg",
    "gps_roll_deg",
    "gps_hrp_mode",
    "gps_speed_knots",
    "gps_speed_kmh",
    "gps_course_true_deg",
    "gps_course_magnetic_deg",
    "gps_rmc_status",
    "gps_mode",
    "gps_gst_rms_m",
    "gps_gst_sigma_major_m",
    "gps_gst_sigma_minor_m",
    "gps_gst_orientation_deg",
    "gps_lat_sigma_m",
    "gps_lon_sigma_m",
    "gps_alt_sigma_m",
    "gps_raw_fields_json",
    "gps_raw_nmea",
]


def utc_iso(unix_ns: int) -> str:
    value = datetime.fromtimestamp(unix_ns / 1e9, tz=timezone.utc)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)
    return int(number) if number is not None else None


def nmea_degrees(value: str, hemisphere: str) -> Optional[float]:
    raw = safe_float(value)
    if raw is None:
        return None
    degrees = int(raw // 100)
    result = degrees + (raw - degrees * 100) / 60.0
    if hemisphere.upper() in ("S", "W"):
        result = -result
    return result


def parse_hhmmss(value: str) -> Tuple[Optional[str], Optional[float], Optional[dt_time]]:
    raw = safe_float(value)
    if raw is None:
        return None, None, None
    hours = int(raw // 10000)
    minutes = int((raw - hours * 10000) // 100)
    seconds = raw - hours * 10000 - minutes * 100
    if not (0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds < 61):
        return None, None, None
    whole_seconds = min(int(seconds), 59)
    microseconds = int(round((seconds - int(seconds)) * 1_000_000))
    if microseconds >= 1_000_000:
        whole_seconds = min(whole_seconds + 1, 59)
        microseconds = 0
    parsed = dt_time(hours, minutes, whole_seconds, microseconds, tzinfo=timezone.utc)
    text = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    return text, hours * 3600.0 + minutes * 60.0 + seconds, parsed


def parse_ddmmyy(value: str) -> Optional[date]:
    if len(value) != 6 or not value.isdigit():
        return None
    day, month, yy = int(value[:2]), int(value[2:4]), int(value[4:6])
    year = 2000 + yy if yy < 80 else 1900 + yy
    try:
        return date(year, month, day)
    except ValueError:
        return None


def checksum_status(line: str) -> Tuple[int, Optional[int]]:
    if not line.startswith("$") or "*" not in line:
        return 0, None
    body, supplied = line[1:].rsplit("*", 1)
    if len(supplied) < 2:
        return 1, 0
    calculated = 0
    for character in body:
        calculated ^= ord(character)
    try:
        expected = int(supplied[:2], 16)
    except ValueError:
        return 1, 0
    return 1, int(calculated == expected)


class NmeaParser:
    """Small lossless parser for the mosaic-H sentences used by this project."""

    def __init__(self) -> None:
        self.last_date: Optional[date] = None

    def parse(self, line: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {column: "" for column in GPS_COLUMNS}
        result["gps_raw_nmea"] = line
        checksum_present, checksum_ok = checksum_status(line)
        result["gps_checksum_present"] = checksum_present
        result["gps_checksum_ok"] = "" if checksum_ok is None else checksum_ok

        try:
            if not line.startswith("$"):
                raise ValueError("line does not start with '$'")
            body = line[1:].split("*", 1)[0]
            fields = body.split(",")
            if not fields or not fields[0]:
                raise ValueError("empty NMEA sentence identifier")

            sentence_id = fields[0]
            if sentence_id == "PSSN" and len(fields) > 1 and fields[1] == "HRP":
                talker, message_type = "P", "HRP"
            else:
                talker = sentence_id[:2]
                message_type = sentence_id[2:]

            result["gps_talker"] = talker
            result["gps_message_type"] = message_type
            result["gps_raw_fields_json"] = json.dumps(
                fields[1:], ensure_ascii=False, separators=(",", ":")
            )

            parsed_time: Optional[dt_time] = None
            parsed_date: Optional[date] = None

            if message_type == "GGA":
                self._put_time(result, fields[1])
                parsed_time = parse_hhmmss(fields[1])[2]
                result["gps_latitude_deg"] = nmea_degrees(fields[2], fields[3])
                result["gps_longitude_deg"] = nmea_degrees(fields[4], fields[5])
                result["gps_fix_quality"] = safe_int(fields[6])
                result["gps_num_satellites"] = safe_int(fields[7])
                result["gps_hdop"] = safe_float(fields[8])
                altitude = safe_float(fields[9])
                geoid = safe_float(fields[11]) if len(fields) > 11 else None
                result["gps_altitude_msl_m"] = altitude
                result["gps_geoid_separation_m"] = geoid
                if altitude is not None and geoid is not None:
                    result["gps_ellipsoid_height_m"] = altitude + geoid

            elif message_type == "HDT":
                result["gps_heading_true_deg"] = safe_float(fields[1])

            elif message_type == "ZDA":
                self._put_time(result, fields[1])
                parsed_time = parse_hhmmss(fields[1])[2]
                parsed_date = date(int(fields[4]), int(fields[3]), int(fields[2]))

            elif message_type == "RMC":
                self._put_time(result, fields[1])
                parsed_time = parse_hhmmss(fields[1])[2]
                result["gps_rmc_status"] = fields[2]
                result["gps_latitude_deg"] = nmea_degrees(fields[3], fields[4])
                result["gps_longitude_deg"] = nmea_degrees(fields[5], fields[6])
                result["gps_speed_knots"] = safe_float(fields[7])
                result["gps_course_true_deg"] = safe_float(fields[8])
                parsed_date = parse_ddmmyy(fields[9])
                if len(fields) > 12:
                    result["gps_mode"] = fields[12]

            elif message_type == "VTG":
                result["gps_course_true_deg"] = safe_float(fields[1])
                result["gps_course_magnetic_deg"] = safe_float(fields[3])
                result["gps_speed_knots"] = safe_float(fields[5])
                result["gps_speed_kmh"] = safe_float(fields[7])
                if len(fields) > 9:
                    result["gps_mode"] = fields[9]

            elif message_type == "GST":
                self._put_time(result, fields[1])
                parsed_time = parse_hhmmss(fields[1])[2]
                result["gps_gst_rms_m"] = safe_float(fields[2])
                result["gps_gst_sigma_major_m"] = safe_float(fields[3])
                result["gps_gst_sigma_minor_m"] = safe_float(fields[4])
                result["gps_gst_orientation_deg"] = safe_float(fields[5])
                result["gps_lat_sigma_m"] = safe_float(fields[6])
                result["gps_lon_sigma_m"] = safe_float(fields[7])
                result["gps_alt_sigma_m"] = safe_float(fields[8])

            elif message_type == "HRP":
                # $PSSN,HRP,time,date,heading,,pitch,roll,,...,SV,mode,...
                self._put_time(result, fields[2])
                parsed_time = parse_hhmmss(fields[2])[2]
                parsed_date = parse_ddmmyy(fields[3])
                result["gps_heading_true_deg"] = safe_float(fields[4])
                result["gps_pitch_deg"] = safe_float(fields[6])
                result["gps_roll_deg"] = safe_float(fields[7])
                result["gps_num_satellites"] = safe_int(fields[10])
                result["gps_hrp_mode"] = safe_int(fields[11])

            if parsed_date is not None:
                self.last_date = parsed_date
            row_date = parsed_date or self.last_date
            if row_date is not None:
                result["gps_date"] = row_date.isoformat()
            if row_date is not None and parsed_time is not None:
                combined = datetime.combine(row_date, parsed_time)
                result["gps_datetime_utc"] = combined.isoformat(
                    timespec="microseconds"
                ).replace("+00:00", "Z")

            result["gps_parsed_ok"] = 1
        except (IndexError, TypeError, ValueError) as exc:
            result["gps_parsed_ok"] = 0
            result["gps_parse_error"] = str(exc)
        return result

    @staticmethod
    def _put_time(result: Dict[str, Any], value: str) -> None:
        text, seconds, _ = parse_hhmmss(value)
        result["gps_utc_time"] = "" if text is None else text
        result["gps_utc_seconds_of_day"] = "" if seconds is None else seconds


def mip_value(point: Any) -> Any:
    """Read an MSCL DataPoint without narrowing its stored value type.

    In particular, navigation latitude/longitude are stored as double and MIP
    reference times can be uint64.  Calling as_float() for every channel loses
    precision before the value ever reaches the CSV file.
    """
    scalar_readers = (
        ("valueType_float", "as_float"),
        ("valueType_double", "as_double"),
        ("valueType_uint8", "as_uint8"),
        ("valueType_uint16", "as_uint16"),
        ("valueType_uint32", "as_uint32"),
        ("valueType_uint64", "as_uint64"),
        ("valueType_int8", "as_int8"),
        ("valueType_int16", "as_int16"),
        ("valueType_int32", "as_int32"),
        ("valueType_bool", "as_bool"),
        ("valueType_string", "as_string"),
    )
    try:
        stored_as = point.storedAs()
        for type_name, accessor_name in scalar_readers:
            if hasattr(mscl, type_name) and stored_as == getattr(mscl, type_name):
                value = getattr(point, accessor_name)()
                if isinstance(value, float) and not math.isfinite(value):
                    return str(value)
                return value
    except Exception:
        # Some older MSCL builds do not expose storedAs() for every value.
        pass

    try:
        # Complex MSCL types are retained in the binding's string form rather
        # than being incorrectly coerced to a float.
        return point.as_string()
    except Exception as exc:
        return f"<unreadable:{exc}>"


def mip_source(descriptor_set: int) -> str:
    return {
        0x80: "CV7_IMU",
        0x81: "CV7_GNSS",
        0x82: "CV7_EKF",
    }.get(descriptor_set, "CV7_OTHER")


def packet_event(packet: Any, port: str, baud: int, start_mono_ns: int) -> Dict[str, Any]:
    host_ns = time.time_ns()
    descriptor_set = int(packet.descriptorSet())
    values: Dict[str, Any] = {}
    for point in packet.data():
        name = str(point.channelName())
        if name in values:
            suffix = 2
            while f"{name}__{suffix}" in values:
                suffix += 1
            name = f"{name}__{suffix}"
        values[name] = mip_value(point)

    try:
        collected_ns: Any = packet.collectedTimestamp().nanoseconds()
    except Exception:
        collected_ns = ""

    return {
        "host_unix_ns": host_ns,
        "host_unix_s": host_ns / 1e9,
        "host_utc_iso": utc_iso(host_ns),
        "elapsed_s": (time.monotonic_ns() - start_mono_ns) / 1e9,
        "source": mip_source(descriptor_set),
        "port": port,
        "baud": baud,
        "descriptor_set": f"0x{descriptor_set:02X}",
        "packet_collected_ns": collected_ns,
        "_mip": values,
    }


def is_nmea_line(line: str) -> bool:
    if not line.startswith("$") or len(line) < 6 or "," not in line:
        return False
    fields = line[1:].split("*", 1)[0].split(",")
    sentence_id = fields[0]
    if sentence_id == "PSSN" and len(fields) > 1:
        return fields[1] == "HRP"
    message_type = sentence_id[-3:] if len(sentence_id) >= 3 else sentence_id
    return message_type in GPS_NMEA_TYPES


def nmea_time_is_current(parsed: Dict[str, Any], host_ns: int) -> Optional[bool]:
    """Return whether a dated/time-tagged NMEA sample is near host UTC."""
    datetime_text = str(parsed.get("gps_datetime_utc", ""))
    if datetime_text:
        try:
            gps_unix_s = datetime.fromisoformat(
                datetime_text.replace("Z", "+00:00")
            ).timestamp()
            return abs(host_ns / 1e9 - gps_unix_s) <= GPS_LIVE_TOLERANCE_S
        except ValueError:
            pass

    gps_sod = safe_float(parsed.get("gps_utc_seconds_of_day"))
    if gps_sod is None:
        return None
    host_datetime = datetime.fromtimestamp(host_ns / 1e9, tz=timezone.utc)
    host_sod = (
        host_datetime.hour * 3600
        + host_datetime.minute * 60
        + host_datetime.second
        + host_datetime.microsecond / 1e6
    )
    # Circular difference handles samples close to the UTC midnight boundary.
    difference = (host_sod - gps_sod + 43200.0) % 86400.0 - 43200.0
    return abs(difference) <= GPS_LIVE_TOLERANCE_S


def port_identity_text(port_info: Any) -> str:
    return " ".join(
        str(value or "")
        for value in (
            port_info.device,
            port_info.description,
            port_info.manufacturer,
            port_info.product,
            port_info.hwid,
        )
    ).lower()


def is_septentrio_port(port_info: Any) -> bool:
    if (port_info.vid, port_info.pid) == SEPTENTRIO_USB_ID:
        return True
    text = port_identity_text(port_info)
    return "septentrio" in text or "mosaic" in text


def cv7_port_score(port_info: Any) -> int:
    text = port_identity_text(port_info)
    value = 0
    if any(key in text for key in ("microstrain", "lord", "hbk", "3dm", "cv7")):
        value += 100
    if "inertial" in text:
        value += 30
    if any(key in text for key in ("usb serial", "usb-serial", "acm", "cp210", "ftdi")):
        value += 10
    if is_septentrio_port(port_info):
        value -= 200
    if platform.system() == "Windows" and port_info.device.upper().startswith("COM"):
        value += 5
    if platform.system() != "Windows" and any(
        key in text for key in ("ttyacm", "ttyusb", "serial/by-id")
    ):
        value += 5
    return value


def gps_port_score(port_info: Any) -> int:
    text = port_identity_text(port_info)
    known_gps_ids = {
        SEPTENTRIO_USB_ID,
        (0x1546, 0x01A8),
        (0x10C4, 0xEA60),
        (0x067B, 0x2303),
        (0x0403, 0x6001),
    }
    value = 0
    if is_septentrio_port(port_info):
        value += 100
    if (port_info.vid, port_info.pid) in known_gps_ids:
        value += 60
    if any(key in text for key in ("nmea", "gnss", "gps", "receiver")):
        value += 20
    if any(key in text for key in ("microstrain", "lord", "hbk", "3dm", "cv7", "inertial")):
        value -= 100
    return value


def serial_ports() -> List[Any]:
    ports = list(list_ports.comports())
    if ports:
        return ports
    system_name = platform.system()
    hint = "COMx" if system_name == "Windows" else "/dev/ttyACM0 or /dev/ttyUSB0"
    raise RuntimeError(
        f"No serial ports found on {system_name}. Provide a port such as {hint}."
    )


def print_serial_ports() -> None:
    print(f"Operating system: {platform.system()} {platform.release()}")
    ports = serial_ports()
    for port_info in ports:
        if port_info.vid is None or port_info.pid is None:
            vid_pid = "VID:PID=None"
        else:
            vid_pid = f"VID:PID={port_info.vid:04X}:{port_info.pid:04X}"
        print(
            f"{port_info.device:24s} | {vid_pid:18s} | "
            f"{port_info.description or ''} | {port_info.manufacturer or ''}"
        )


def open_serial_without_writes(port: str, baud: int, timeout: float):
    handle = serial.Serial()
    handle.port = port
    handle.baudrate = baud
    handle.timeout = timeout
    handle.write_timeout = timeout
    handle.dtr = False
    handle.rts = False
    handle.open()
    return handle


def probe_mip_port(port: str, baud: int, timeout_s: float = 1.2) -> Tuple[int, List[int]]:
    """Count data packets using only the official MSCL receive call."""
    connection = None
    packet_count = 0
    descriptor_sets = set()
    try:
        connection = mscl.Connection.Serial(port, baud)
        node = mscl.InertialNode(connection)
        deadline = time.monotonic() + max(timeout_s, 0.1)
        while time.monotonic() < deadline:
            packets = node.getDataPackets(100, 200)
            packet_count += len(packets)
            descriptor_sets.update(int(packet.descriptorSet()) for packet in packets)
            if packet_count >= 2 and descriptor_sets.intersection({0x80, 0x82, 0xA0}):
                break
    finally:
        if connection is not None:
            try:
                connection.disconnect()
            except Exception:
                pass
    return packet_count, sorted(descriptor_sets)


def detect_cv7_mip_port(exclude: Iterable[str], baud: int) -> str:
    excluded = {value.upper() for value in exclude if value}
    ranked = sorted(serial_ports(), key=cv7_port_score, reverse=True)
    candidates = [
        port_info for port_info in ranked
        if port_info.device.upper() not in excluded
    ]
    print(
        f"Auto-detecting CV7 MIP port on {platform.system()}: "
        + ", ".join(port_info.device for port_info in candidates)
    )

    fallback = None
    for port_info in candidates:
        score = cv7_port_score(port_info)
        if fallback is None and score > 0 and not is_septentrio_port(port_info):
            fallback = port_info.device
        try:
            count, descriptor_sets = probe_mip_port(port_info.device, baud)
        except Exception as exc:
            print(f"  {port_info.device}: no readable MIP data ({exc})")
            continue
        descriptor_text = ",".join(f"0x{value:02X}" for value in descriptor_sets)
        print(
            f"  {port_info.device}: packets={count}, "
            f"descriptor_sets={descriptor_text or 'none'}"
        )
        if count and set(descriptor_sets).intersection({0x80, 0x82, 0xA0}):
            print(f"  selected CV7 port: {port_info.device}")
            return port_info.device

    if fallback:
        print(
            f"WARNING: no active MIP stream detected; selecting {fallback} from "
            "MicroStrain device metadata. The CV7 must already be streaming."
        )
        return fallback
    raise RuntimeError(
        "Could not identify a CV7 MIP port. Pass --cv7-port explicitly and "
        "verify that the existing CV7 configuration is streaming data."
    )


def detect_gps_nmea_port(exclude: Iterable[str], baud: int) -> str:
    excluded = {value.upper() for value in exclude}
    candidates = [
        port_info
        for port_info in sorted(serial_ports(), key=gps_port_score, reverse=True)
        if port_info.device.upper() not in excluded
    ]
    if not candidates:
        raise RuntimeError(
            "No candidate GPS serial ports found; pass --gps-port explicitly"
        )

    print(
        f"Auto-detecting GPS NMEA port on {platform.system()}: "
        + ", ".join(port_info.device for port_info in candidates)
    )
    best_port = None
    best_count = 0
    for port_info in candidates:
        candidate = port_info.device
        print(f"  checking {candidate} for an existing NMEA stream ...")
        try:
            handle = open_serial_without_writes(candidate, baud, 0.20)
        except Exception as exc:
            print(f"  {candidate}: cannot open ({exc})")
            continue
        count = 0
        deadline = time.monotonic() + 2.0
        try:
            while time.monotonic() < deadline:
                raw = handle.readline()
                if raw and is_nmea_line(raw.decode("ascii", errors="replace").strip()):
                    count += 1
                    if count >= 2:
                        print(f"  selected GPS port: {candidate}")
                        return candidate
        finally:
            handle.close()
        if count > best_count:
            best_port = candidate
            best_count = count
    if best_port:
        print(
            f"WARNING: selecting low-confidence GPS port {best_port}; "
            f"only {best_count} NMEA sentence(s) were observed."
        )
        return best_port
    raise RuntimeError("Serial ports were found, but none emitted NMEA")


def gps_reader_process(
    port: str,
    baud: int,
    start_mono_ns: int,
    events: Any,
    stop_event: Any,
) -> None:
    """Drain and parse NMEA in a process independent of MSCL's native calls."""
    # Ctrl+C is delivered to every Python process attached to the Windows
    # console.  Only the parent should handle it and set the shared stop event;
    # otherwise the child prints a harmless KeyboardInterrupt traceback while
    # blocked in pyserial.read().
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        pass

    parser = NmeaParser()
    sentence_index = 0
    try:
        handle = open_serial_without_writes(port, baud, 0.20)
    except Exception as exc:
        host_ns = time.time_ns()
        events.put([{
            "host_unix_ns": host_ns,
            "host_unix_s": host_ns / 1e9,
            "host_utc_iso": utc_iso(host_ns),
            "elapsed_s": (time.monotonic_ns() - start_mono_ns) / 1e9,
            "source": "ERROR",
            "port": port,
            "baud": baud,
            "gps_parse_error": f"GPS open failed: {exc}",
        }])
        stop_event.set()
        return

    try:
        # Auto-detection opens this port briefly before the recorder.  Discard
        # bytes left in the Windows serial buffer so recording starts live
        # instead of replaying an old backlog or a partial NMEA sentence.
        try:
            handle.reset_input_buffer()
        except Exception:
            pass

        pending = bytearray()
        live_stream = False
        live_fallback_deadline = time.monotonic() + GPS_LIVE_FALLBACK_S
        while not stop_event.is_set():
            # Fetch all bytes already buffered by the serial driver in one
            # operation.  This lets the process catch up quickly if Windows
            # delivers multiple 10 Hz NMEA sentences together.
            waiting = handle.in_waiting
            chunk = handle.read(waiting if waiting > 0 else 1)
            if not chunk:
                continue
            pending.extend(chunk)
            complete_lines = pending.split(b"\n")
            pending = bytearray(complete_lines.pop())
            batch: List[Dict[str, Any]] = []
            for raw in complete_lines:
                line = raw.decode("ascii", errors="replace").strip()
                if not line or not line.startswith("$"):
                    # The first bytes after opening may be the tail of a
                    # sentence.  Ignore that incomplete startup fragment.
                    continue
                host_ns = time.time_ns()
                parsed = parser.parse(line)
                if not live_stream:
                    is_current = nmea_time_is_current(parsed, host_ns)
                    if is_current is True or time.monotonic() >= live_fallback_deadline:
                        live_stream = True
                    else:
                        # Some Septentrio virtual COM endpoints preserve data
                        # while unopened.  Parse it to maintain NMEA date state,
                        # but do not mix pre-recording backlog into the new CSV.
                        continue
                sentence_index += 1
                parsed.update(
                    {
                        "host_unix_ns": host_ns,
                        "host_unix_s": host_ns / 1e9,
                        "host_utc_iso": utc_iso(host_ns),
                        "elapsed_s": (time.monotonic_ns() - start_mono_ns) / 1e9,
                        "source": "GPS_NMEA",
                        "port": port,
                        "baud": baud,
                        "gps_sentence_index": sentence_index,
                    }
                )
                batch.append(parsed)
            if batch:
                events.put(batch)
            if len(pending) > 65536:
                # Protect against an unplugged/corrupt source with no newline.
                pending.clear()
    except Exception as exc:
        host_ns = time.time_ns()
        events.put([{
            "host_unix_ns": host_ns,
            "host_unix_s": host_ns / 1e9,
            "host_utc_iso": utc_iso(host_ns),
            "elapsed_s": (time.monotonic_ns() - start_mono_ns) / 1e9,
            "source": "ERROR",
            "port": port,
            "baud": baud,
            "gps_parse_error": f"GPS read failed: {exc}",
        }])
        stop_event.set()
    finally:
        handle.close()


def make_output_path(requested: Optional[str]) -> Path:
    if requested:
        return Path(requested).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / f"data/cv7_all_readonly_{stamp}.csv"


def valid_geographic_point(lat: Optional[float], lon: Optional[float]) -> bool:
    if lat is None or lon is None:
        return False
    if abs(lat) < 1e-12 and abs(lon) < 1e-12:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def downsample_points(
    points: List[Tuple[float, float]], max_points: int
) -> List[Tuple[float, float]]:
    if max_points <= 0 or len(points) <= max_points:
        return points
    step = math.ceil(len(points) / max_points)
    return points[::step]


def track_to_local_meters(
    points: List[Tuple[float, float]], reference: Tuple[float, float]
) -> List[Tuple[float, float]]:
    ref_lat, ref_lon = reference
    cos_lat = math.cos(math.radians(ref_lat))
    return [
        (
            math.radians(lon - ref_lon) * EARTH_RADIUS_M * cos_lat,
            math.radians(lat - ref_lat) * EARTH_RADIUS_M,
        )
        for lat, lon in points
    ]


def plot_recorded_tracks(
    csv_path: Path,
    requested_plot_path: Optional[str],
    show_plot: bool,
    max_plot_points: int,
) -> Optional[Path]:
    """Plot raw GPS and fused CV7 EKF geographic points from the unified CSV."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "WARNING: matplotlib is not installed; skipping plot. Install with: "
            "python -m pip install matplotlib"
        )
        return None

    gps_gga: List[Tuple[float, float]] = []
    gps_fallback: List[Tuple[float, float]] = []
    ekf_points: List[Tuple[float, float]] = []
    ekf_state_counts: Dict[str, int] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = row.get("source", "")
            if source == "GPS_NMEA":
                lat = safe_float(row.get("gps_latitude_deg"))
                lon = safe_float(row.get("gps_longitude_deg"))
                if not valid_geographic_point(lat, lon):
                    continue
                point = (float(lat), float(lon))
                if row.get("gps_message_type") == "GGA":
                    gps_gga.append(point)
                else:
                    gps_fallback.append(point)
            elif source == "CV7_EKF":
                state = str(row.get("mip_estFilterState", "")).strip()
                if state:
                    ekf_state_counts[state] = ekf_state_counts.get(state, 0) + 1
                lat = safe_float(row.get("mip_estLatitude"))
                lon = safe_float(row.get("mip_estLongitude"))
                if valid_geographic_point(lat, lon):
                    ekf_points.append((float(lat), float(lon)))

    # GGA is the preferred raw position record.  RMC/other position messages
    # are used only when no GGA points exist, preventing duplicate GPS points.
    gps_points = gps_gga if gps_gga else gps_fallback
    if not gps_points and not ekf_points:
        print(
            "WARNING: no valid GPS or EKF latitude/longitude points were found; "
            "no comparison plot was created."
        )
        if ekf_state_counts:
            print(f"EKF filter states observed: {ekf_state_counts}")
        return None

    reference = ekf_points[0] if ekf_points else gps_points[0]
    gps_local = downsample_points(
        track_to_local_meters(gps_points, reference), max_plot_points
    )
    ekf_local = downsample_points(
        track_to_local_meters(ekf_points, reference), max_plot_points
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    if ekf_local:
        ax.plot(
            [point[0] for point in ekf_local],
            [point[1] for point in ekf_local],
            color=EKF_PLOT_COLOR,
            linewidth=1.0,
            marker=".",
            markersize=2.0,
            label=f"CV7 IMU/EKF fused ({len(ekf_points)} points)",
            zorder=2,
        )
        ax.scatter(
            [ekf_local[0][0]], [ekf_local[0][1]],
            color=EKF_PLOT_COLOR, edgecolors="black", s=55,
            label="EKF start", zorder=4,
        )
    else:
        ax.text(
            0.02,
            0.98,
            "No valid EKF latitude/longitude points\n"
            f"Filter states: {ekf_state_counts or 'not available'}",
            transform=ax.transAxes,
            va="top",
            color=EKF_PLOT_COLOR,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": EKF_PLOT_COLOR},
        )

    if gps_local:
        ax.scatter(
            [point[0] for point in gps_local],
            [point[1] for point in gps_local],
            color=GPS_PLOT_COLOR,
            s=22,
            alpha=0.80,
            label=f"GPS raw ({len(gps_points)} points)",
            zorder=3,
        )
    else:
        ax.text(
            0.02,
            0.83,
            "No valid GPS raw points",
            transform=ax.transAxes,
            va="top",
            color=GPS_PLOT_COLOR,
        )

    ax.set_title(
        "Raw GPS vs CV7 IMU/EKF Fused Position\n"
        f"Blue: EKF ({len(ekf_points)}) | Orange: GPS raw ({len(gps_points)})"
    )
    ax.set_xlabel("East from first valid point (m)")
    ax.set_ylabel("North from first valid point (m)")
    ax.axis("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = (
        Path(requested_plot_path).expanduser().resolve()
        if requested_plot_path
        else csv_path.with_name(csv_path.stem + "_gps_vs_ekf.png")
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    print(f"Track plot saved to: {plot_path}")
    print(
        f"Plot points: EKF={len(ekf_points)}, GPS raw={len(gps_points)}; "
        "blue=EKF, orange=GPS raw"
    )

    if show_plot:
        print("Close the plot window to finish.")
        plt.show()
    else:
        plt.close(fig)
    return plot_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read existing CV7 IMU/EKF streams plus mosaic-H NMEA into one CSV. "
            "No IMU or GPS configuration commands are sent."
        )
    )
    parser.add_argument(
        "--cv7-port",
        default=DEFAULT_CV7_PORT,
        help="CV7 MIP port or 'auto' (default: auto)",
    )
    parser.add_argument("--cv7-baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument(
        "--gps-port",
        default="auto",
        help="mosaic-H NMEA port, 'auto', or 'none' (default: auto)",
    )
    parser.add_argument("--gps-baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--output", help="output CSV path")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to record; 0 means until Ctrl+C",
    )
    parser.add_argument(
        "--schema-seconds",
        type=float,
        default=2.0,
        help="initial seconds used to discover MIP channel columns",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="show detected OS and serial-port identities, then exit",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="do not create the GPS-vs-EKF plot after recording",
    )
    parser.add_argument(
        "--no-show-plot",
        action="store_true",
        help="save the plot PNG without opening a matplotlib window",
    )
    parser.add_argument(
        "--plot-output",
        help="plot PNG path; default is beside the CSV",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=100000,
        help="downsample each plotted track above this count; 0 disables",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="publish live status and launch the C++ Pangolin viewer",
    )
    parser.add_argument(
        "--gui-no-launch",
        action="store_true",
        help=(
            "publish live GUI telemetry without launching a local viewer; "
            "also enables GUI telemetry when --gui is omitted"
        ),
    )
    parser.add_argument(
        "--gui-host",
        default="127.0.0.1",
        help="Pangolin viewer host for UDP telemetry (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--gui-port",
        type=int,
        default=5600,
        help="Pangolin viewer UDP telemetry port (default: 5600)",
    )
    parser.add_argument(
        "--gui-rate",
        type=float,
        default=10.0,
        help="GUI snapshot rate in Hz; CSV recording remains native-rate",
    )
    parser.add_argument(
        "--gui-executable",
        help="path to cv7_pangolin_viewer executable",
    )
    init_heading_group = parser.add_mutually_exclusive_group()
    init_heading_group.add_argument(
        "--init-heading-deg",
        type=float,
        default=None,
        help=(
            "send MIP Set Initial Heading (0x0D, 0x03) once, in degrees. "
            "Equivalent to clicking Apply Initial Values with only Heading/Yaw "
            "in SensorConnect. Only valid while the filter is in the "
            "Initialization state."
        ),
    )
    init_heading_group.add_argument(
        "--init-heading-rad",
        type=float,
        default=None,
        help="same as --init-heading-deg but the value is already in radians",
    )
    parser.add_argument(
        "--init-heading-ignore-errors",
        action="store_true",
        help=(
            "do not abort if Set Initial Heading fails (e.g. filter already past "
            "the Initialization state, or MSCL command timeout under heavy data load)"
        ),
    )
    parser.add_argument(
        "--init-heading-timeout-ms",
        type=int,
        default=3000,
        help=(
            "MSCL command response timeout in ms while sending Set Initial Heading. "
            "Defaults to 3000 because the CV7 is streaming data at up to 500 Hz on the "
            "same serial link and the stock MSCL timeout can be too short."
        ),
    )
    parser.add_argument(
        "--init-heading-retries",
        type=int,
        default=3,
        help="number of Set Initial Heading attempts before giving up",
    )
    return parser


def require_dependencies(serial_required: bool) -> None:
    missing = []
    if mscl is None:
        message = (
            "MicroStrain MSCL Python binding "
            "(module 'mscl' or pip package 'python-mscl')"
        )
        if MSCL_IMPORT_DETAILS:
            message += "; " + " | ".join(MSCL_IMPORT_DETAILS)
        missing.append(message)
    if serial_required and (serial is None or list_ports is None):
        missing.append("pyserial")
    if missing:
        raise SystemExit("Missing dependency: " + ", ".join(missing))


def drain_queue(events: Any) -> List[Dict[str, Any]]:
    drained = []
    while True:
        try:
            item = events.get_nowait()
            if isinstance(item, list):
                drained.extend(item)
            else:
                drained.append(item)
        except queue.Empty:
            return drained


def finalize_event(
    event: Dict[str, Any], mip_columns: Dict[str, str], record_index: int
) -> Dict[str, Any]:
    row = dict(event)
    mip_fields = row.pop("_mip", {})
    row["record_index"] = record_index
    if mip_fields:
        row["mip_fields_json"] = json.dumps(
            mip_fields, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        for channel_name, value in mip_fields.items():
            column = mip_columns.get(channel_name)
            if column is not None:
                row[column] = value
    return row


def run(args: argparse.Namespace) -> Path:
    requested_cv7 = args.cv7_port.strip()
    requested_gps = args.gps_port.strip()
    cv7_auto = requested_cv7.lower() == "auto"
    gps_enabled = requested_gps.lower() != "none"
    gps_auto = requested_gps.lower() == "auto"
    require_dependencies(serial_required=(cv7_auto or gps_enabled))
    if args.duration < 0 or args.schema_seconds < 0:
        raise SystemExit("--duration and --schema-seconds must be non-negative")

    gui_enabled = args.gui or args.gui_no_launch
    if gui_enabled and not (1 <= args.gui_port <= 65534):
        raise SystemExit("--gui-port must be between 1 and 65534")
    if gui_enabled and args.gui_rate <= 0:
        raise SystemExit("--gui-rate must be positive")
    viewer_executable = None
    if gui_enabled and not args.gui_no_launch:
        viewer_executable = find_viewer_executable(args.gui_executable)
        if viewer_executable is None:
            raise SystemExit(
                "Pangolin viewer executable was not found. Build "
                "IMU_EKF/pangolin_viewer first, pass --gui-executable PATH, "
                "or use --gui-no-launch to send telemetry to a viewer that "
                "you started separately."
            )

    output_path = make_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv7_exclude = []
    if gps_enabled and not gps_auto:
        cv7_exclude.append(requested_gps)
    cv7_port = (
        detect_cv7_mip_port(cv7_exclude, args.cv7_baud)
        if cv7_auto
        else requested_cv7
    )

    gps_port: Optional[str]
    if not gps_enabled:
        gps_port = None
    elif gps_auto:
        gps_port = detect_gps_nmea_port([cv7_port], args.gps_baud)
    else:
        gps_port = requested_gps

    print(f"Opening CV7 data stream: {cv7_port} @ {args.cv7_baud}")
    connection = mscl.Connection.Serial(cv7_port, args.cv7_baud)
    node = mscl.InertialNode(connection)

    # ---- Optional one-shot: MIP Set Initial Heading (0x0D, 0x03) ----
    # Mirrors SensorConnect's "Apply Initial Values" (Heading/Yaw only) button.
    # This is the sole exception to the read-only guarantee, and only fires when
    # the user explicitly passes --init-heading-deg or --init-heading-rad.
    init_heading_rad: Optional[float] = None
    if args.init_heading_rad is not None:
        init_heading_rad = float(args.init_heading_rad)
        init_heading_src = "--init-heading-rad"
    elif args.init_heading_deg is not None:
        init_heading_rad = math.radians(float(args.init_heading_deg))
        init_heading_src = "--init-heading-deg"

    if init_heading_rad is not None:
        # Wrap to (-pi, pi] so out-of-range user input (e.g. 370 deg) is sane.
        wrapped = math.atan2(math.sin(init_heading_rad), math.cos(init_heading_rad))
        print(
            "Read-only mode with ONE exception: sending MIP Set Initial Heading "
            f"(0x0D, 0x03) = {wrapped:.6f} rad (from {init_heading_src}). "
            "No other configuration, stream, or save-settings commands will be issued."
        )
        # Raise MSCL command timeout: the CV7 is already streaming 0x80/0x82 on the
        # same serial link, and the stock ~500 ms timeout can miss the ACK when the
        # response parser is competing with a flood of data packets. This is the
        # most common cause of Error_Communication here (not an actual NACK).
        prior_timeout_ms: Optional[int] = None
        try:
            prior_timeout_ms = int(node.commandsTimeout())
        except Exception:
            prior_timeout_ms = None
        try:
            node.commandsTimeout(int(max(500, args.init_heading_timeout_ms)))
        except Exception as timeout_exc:
            print(f"  warning: could not raise MSCL command timeout: {timeout_exc}")

        last_exc: Optional[BaseException] = None
        sent_ok = False
        attempts = max(1, int(args.init_heading_retries))
        for attempt in range(1, attempts + 1):
            try:
                node.setInitialHeading(float(wrapped))
                sent_ok = True
                print(f"  Set Initial Heading acknowledged (attempt {attempt}/{attempts}).")
                break
            except Exception as heading_exc:
                last_exc = heading_exc
                err_txt = str(heading_exc)
                # Error_Communication == timeout / no response parsed. Retrying often
                # succeeds because the ACK just got buried behind streaming data.
                is_comm = "Error_Communication" in err_txt or "Communication" in err_txt
                print(
                    f"  Set Initial Heading attempt {attempt}/{attempts} failed"
                    f" ({'timeout' if is_comm else 'error'}): {heading_exc}"
                )
                if attempt < attempts:
                    time.sleep(0.25)

        # Restore prior timeout regardless of outcome.
        if prior_timeout_ms is not None:
            try:
                node.commandsTimeout(prior_timeout_ms)
            except Exception:
                pass

        if not sent_ok:
            msg = (
                "Set Initial Heading failed after retries: "
                f"{last_exc}. Common causes: (1) the filter has already left the "
                "Initialization state (SensorConnect shows the same behavior; "
                "power-cycle the CV7 or reset the filter to re-apply), or "
                "(2) MSCL command timeout under heavy streaming load "
                "(try --init-heading-timeout-ms 5000 or --init-heading-retries 5). "
                "Pass --init-heading-ignore-errors to continue recording anyway."
            )
            if args.init_heading_ignore_errors:
                print(msg)
                print("  --init-heading-ignore-errors set; continuing.")
            else:
                try:
                    connection.disconnect()
                except Exception:
                    pass
                raise SystemExit(msg)
    else:
        print("Read-only mode: no CV7 or GPS configuration/stream-control calls will be made.")
    # ---- end Set Initial Heading block ----

    if gps_port:
        print(f"Opening GPS NMEA stream: {gps_port} @ {args.gps_baud}")
    else:
        print("GPS input disabled by --gps-port none")

    start_mono_ns = time.monotonic_ns()
    start_monotonic = start_mono_ns / 1e9
    stop_deadline = (
        time.monotonic() + args.duration if args.duration > 0 else None
    )
    process_context = mp.get_context("spawn")
    events = process_context.Queue()
    stop_event = process_context.Event()
    old_handlers = {}

    def request_stop(_signum=None, _frame=None) -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_stop)

    gps_process_obj: Optional[mp.Process] = None
    viewer_process = None
    telemetry_state: Optional[TelemetryState] = None
    telemetry_publisher: Optional[UdpTelemetryPublisher] = None
    if gui_enabled:
        telemetry_state = TelemetryState(
            cv7_port=cv7_port,
            gps_port=gps_port,
            output_path=output_path,
            start_monotonic=start_monotonic,
        )
        telemetry_publisher = UdpTelemetryPublisher(
            host=args.gui_host,
            port=args.gui_port,
            rate_hz=args.gui_rate,
        )
        if viewer_executable is not None:
            viewer_process = launch_viewer(viewer_executable, args.gui_port)
            print(f"Pangolin viewer started: {viewer_executable}")
        print(
            f"Live GUI telemetry: UDP {args.gui_host}:{args.gui_port} "
            f"@ {args.gui_rate:g} Hz"
        )
        telemetry_publisher.maybe_send(
            telemetry_state, recorder_status="Starting", force=True
        )

    def observe_event(event: Dict[str, Any]) -> None:
        if telemetry_state is not None:
            telemetry_state.update(event)

    def update_gui(force: bool = False) -> None:
        if telemetry_state is None or telemetry_publisher is None:
            return
        telemetry_publisher.maybe_send(telemetry_state, force=force)
        if telemetry_publisher.stop_requested():
            print("Stop requested from Pangolin viewer.")
            stop_event.set()

    if gps_port:
        gps_process_obj = process_context.Process(
            target=gps_reader_process,
            args=(gps_port, args.gps_baud, start_mono_ns, events, stop_event),
            name="mosaic-H-NMEA-reader-process",
            daemon=True,
        )
        gps_process_obj.start()

    buffered: List[Dict[str, Any]] = []
    mip_names = set()
    schema_deadline = time.monotonic() + args.schema_seconds
    counts: Dict[str, int] = {}
    last_report = time.monotonic()
    record_index = 0

    try:
        print(f"Discovering active MIP fields for {args.schema_seconds:.1f} s ...")
        while (
            not stop_event.is_set()
            and time.monotonic() < schema_deadline
            and (stop_deadline is None or time.monotonic() < stop_deadline)
        ):
            packets = node.getDataPackets(100, 1000)
            for packet in packets:
                event = packet_event(packet, cv7_port, args.cv7_baud, start_mono_ns)
                mip_names.update(event.get("_mip", {}).keys())
                buffered.append(event)
                observe_event(event)
            gps_events = drain_queue(events)
            for event in gps_events:
                observe_event(event)
            buffered.extend(gps_events)
            update_gui()

        mip_columns = {
            name: f"mip_{name}"
            for name in sorted(mip_names)
        }
        fieldnames = COMMON_COLUMNS + GPS_COLUMNS + list(mip_columns.values())

        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            def write_event(event: Dict[str, Any]) -> None:
                nonlocal record_index
                record_index += 1
                source = str(event.get("source", "UNKNOWN"))
                counts[source] = counts.get(source, 0) + 1
                writer.writerow(finalize_event(event, mip_columns, record_index))

            for event in buffered:
                write_event(event)
            buffered.clear()
            csv_file.flush()

            print(f"Recording to: {output_path}")
            if args.duration > 0:
                print(f"Duration: {args.duration:.1f} s")
            else:
                print("Press Ctrl+C to stop.")

            while not stop_event.is_set():
                elapsed = (time.monotonic_ns() - start_mono_ns) / 1e9
                if stop_deadline is not None and time.monotonic() >= stop_deadline:
                    break

                packets = node.getDataPackets(100, 1000)
                for packet in packets:
                    event = packet_event(
                        packet, cv7_port, args.cv7_baud, start_mono_ns
                    )
                    observe_event(event)
                    write_event(event)
                for event in drain_queue(events):
                    observe_event(event)
                    write_event(event)
                update_gui()

                now = time.monotonic()
                if now - last_report >= 1.0:
                    csv_file.flush()
                    summary = " ".join(
                        f"{name}={value}" for name, value in sorted(counts.items())
                    )
                    print(f"[{elapsed:8.1f}s] rows={record_index} {summary}")
                    last_report = now

            stop_event.set()
            if gps_process_obj is not None:
                gps_process_obj.join(timeout=2.0)
            for event in drain_queue(events):
                observe_event(event)
                write_event(event)
            csv_file.flush()
    finally:
        stop_event.set()
        if gps_process_obj is not None and gps_process_obj.is_alive():
            gps_process_obj.join(timeout=1.0)
        if gps_process_obj is not None and gps_process_obj.is_alive():
            gps_process_obj.terminate()
            gps_process_obj.join(timeout=1.0)
        try:
            connection.disconnect()
        except Exception:
            pass
        if telemetry_state is not None and telemetry_publisher is not None:
            telemetry_publisher.maybe_send(
                telemetry_state,
                recorder_status="Stopped",
                force=True,
                shutdown=True,
            )
            telemetry_publisher.close()
        if viewer_process is not None:
            try:
                viewer_process.wait(timeout=1.0)
            except Exception:
                # The viewer is intentionally independent; never terminate it
                # in a way that could affect CSV cleanup.
                pass
        for sig, old_handler in old_handlers.items():
            signal.signal(sig, old_handler)

    print(f"Stopped. Wrote {record_index} rows to {output_path}")
    print("Counts: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.list_ports:
            if serial is None or list_ports is None:
                raise SystemExit("Missing dependency: pyserial")
            print_serial_ports()
            return
        output_path = run(args)
        if not args.no_plot:
            plot_recorded_tracks(
                csv_path=output_path,
                requested_plot_path=args.plot_output,
                show_plot=not args.no_show_plot,
                max_plot_points=args.max_plot_points,
            )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    mp.freeze_support()
    main()
