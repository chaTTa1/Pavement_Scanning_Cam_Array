#!/usr/bin/env python3
"""
Read fused EKF/Navigation Filter output from a MicroStrain 3DM-CV7/CV7-INS.

The CV7 uses the MicroStrain MIP protocol. With the C-Series Development Kit
USB cable/board, the device appears as a virtual serial port:

    Windows: COM3, COM4, ...
    Linux  : /dev/ttyACM0, /dev/ttyUSB0, /dev/serial/by-id/...

Install runtime dependency:

    python -m pip install pyserial

Examples:

    python IMU_EKF/CV7_INS_EKF.py
    python IMU_EKF/CV7_INS_EKF.py --port COM5 --configure
    python IMU_EKF/CV7_INS_EKF.py --debug --configure --rate-hz 100
    python IMU_EKF/CV7_INS_EKF.py --port COM13 --status --pretty

Spyder IDE:

    1. Open this file in Spyder.
    2. Edit the "SPYDER / IDE SETTINGS" section below.
    3. Press Run. The script will use those settings automatically in Spyder.
    4. For one command-line style status snapshot from Spyder Console, run:
       run_cv7_status(port="COM13")

Default output is JSON lines. With --debug false, only EKF/filter data from
descriptor set 0x82 is printed. With --debug true, IMU sensor data, GNSS/GPS
data, and EKF/filter data are printed with an explicit "source" label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:
    raise SystemExit(
        "pyserial is required. Install it with: python -m pip install pyserial"
    ) from exc


# ---------------- SPYDER / IDE SETTINGS ----------------
# In Spyder, edit these values and press Run. No command line arguments needed.
USE_SPYDER_SETTINGS_WHEN_AVAILABLE = True

# Set to your actual COM port, for example "COM5". Leave as None for auto-detect.
SPYDER_PORT: Optional[str] = "COM13"
SPYDER_GPS_PORT: Optional[str] = None  # None means auto-detect GPS NMEA port when CSV recording is enabled.

SPYDER_BAUD = 115200
SPYDER_GPS_BAUD = 115200
SPYDER_DEBUG = False

# True: ask the CV7 to stream the needed MIP fields before reading.
# False: only listen to the current sampling setup stored on the device.
SPYDER_CONFIGURE = True

SPYDER_RATE_HZ = 500
SPYDER_ASSUME_BASE_RATE_HZ = 1000
SPYDER_STREAM_PRESET = "csv"
SPYDER_PRETTY_JSON = False
SPYDER_SUMMARY_OUTPUT = True

# Limit Spyder console output. 2 Hz means two summary lines per second.
# Set to 0.0 only if you really want to print every decoded field.
SPYDER_PRINT_HZ = 2.0

# Record CSV files in the same folder as this script.
SPYDER_RECORD_CSV = True
SPYDER_CSV_PREFIX = "cv7"

# Gap/skipping validation. A packet gap larger than threshold * expected period
# is written to the skip validation CSV.
SPYDER_SKIP_CHECK = True
SPYDER_SKIP_GAP_THRESHOLD = 1.5
SPYDER_EXPECTED_EKF_HZ = 500.0
SPYDER_EXPECTED_GPS_HZ = 10.0

# 0.0 means run forever. For a quick Spyder test, set this to 10.0.
SPYDER_DURATION_S = 5.0
SPYDER_READ_TIMEOUT_S = 1.0
# -------------------------------------------------------


SYNC = b"\x75\x65"

DESC_BASE = 0x01
DESC_3DM_CMD = 0x0C
DESC_SENSOR = 0x80
DESC_GNSS_LEGACY = 0x81
DESC_FILTER = 0x82
DESC_GNSS_MODULES = set(range(0x91, 0x96))

FIELD_ACK_NACK = 0xF1

CMD_BASE_PING = 0x01
CMD_BASE_SET_TO_IDLE = 0x02
CMD_BASE_RESUME = 0x06

CMD_3DM_POLL_IMU_MESSAGE = 0x01
CMD_3DM_POLL_GNSS_MESSAGE = 0x02
CMD_3DM_POLL_FILTER_MESSAGE = 0x03
CMD_3DM_GET_BASE_RATE = 0x0E
CMD_3DM_MESSAGE_FORMAT = 0x0F
CMD_3DM_DATASTREAM_CONTROL = 0x11
CMD_3DM_POLL_DATA = 0x0D
CMD_3DM_PPS_SOURCE = 0x28
CMD_3DM_GPIO_CONFIG = 0x41

DESC_FILTER_CMD = 0x0D
DESC_SYSTEM_CMD = 0x7F

CMD_FILTER_RESET_FILTER = 0x01
CMD_FILTER_RUN = 0x05
CMD_FILTER_AIDING_MEASUREMENT_ENABLE = 0x50
CMD_FILTER_INITIALIZATION_CONFIGURATION = 0x52
CMD_SYSTEM_INTERFACE_CONTROL = 0x02

MIP_FUNCTION_WRITE = 0x01
MIP_FUNCTION_READ = 0x02

ACK_OK = 0x00

REPLY_3DM_BASE_RATE = 0x8E
REPLY_3DM_PPS_SOURCE = 0xA8
REPLY_3DM_GPIO_CONFIG = 0xC1
REPLY_FILTER_AIDING_MEASUREMENT_ENABLE = 0xD0
REPLY_FILTER_INITIALIZATION_CONFIGURATION = 0xD2
REPLY_SYSTEM_INTERFACE_CONTROL = 0x82


FILTER_FIELDS = {
    0x01: "position_llh",
    0x02: "velocity_ned",
    0x03: "attitude_quaternion",
    0x05: "attitude_euler",
    0x08: "position_uncertainty",
    0x09: "velocity_uncertainty",
    0x0A: "euler_uncertainty",
    0x0D: "linear_accel",
    0x0E: "compensated_angular_rate",
    0x0F: "wgs84_gravity",
    0x10: "filter_status",
    0x11: "filter_timestamp",
    0x13: "gravity_vector",
    0x1C: "compensated_accel",
    0x40: "ecef_position",
    0x41: "ecef_velocity",
    0x42: "relative_position_ned",
    0x43: "gnss_pos_aid_status",
    0x44: "gnss_att_aid_status",
    0x45: "heading_aid_status",
    0x46: "aid_measurement_summary",
    0x50: "frame_config_error",
    0x51: "frame_config_error_uncertainty",
    0xD3: "shared_gps_timestamp",
    0xD5: "shared_reference_timestamp",
    0xD7: "shared_external_timestamp",
}

SENSOR_FIELDS = {
    0x01: "raw_accel",
    0x02: "raw_gyro",
    0x03: "raw_mag",
    0x04: "scaled_accel",
    0x05: "scaled_gyro",
    0x06: "scaled_mag",
    0x0A: "comp_quaternion",
    0x0C: "comp_euler",
    0x0E: "internal_timestamp",
    0x12: "gps_timestamp",
    0x14: "temperature_abs",
    0x17: "scaled_pressure",
    0x18: "overrange_status",
}

GNSS_FIELDS = {
    0x03: "gps_position_llh",
    0x05: "gps_velocity_ned",
    0x08: "gps_utc_time",
    0x09: "gps_time",
    0x0B: "gps_fix_info",
    0x0D: "gps_hw_status",
}

FILTER_MODE = {
    0: "startup",
    1: "init",
    2: "gx5_run_solution_valid_or_vertical_gyro",
    3: "gx5_run_solution_error_or_ahrs",
    4: "full_nav",
}

GNSS_FIX_TYPE = {
    0: "fix_unavailable",
    1: "fix_2d",
    2: "fix_3d",
    3: "fix_3d_with_sbas",
    4: "fix_3d_with_dgnss",
    5: "fix_3d_with_rtk_float",
    6: "fix_3d_with_rtk_fixed",
}

GPS_FIX_QUALITY_NAMES = {
    0: "invalid",
    1: "gps_fix",
    2: "dgps_fix",
    3: "pps_fix",
    4: "rtk_fixed",
    5: "rtk_float",
    6: "estimated_dead_reckoning",
    7: "manual_input",
    8: "simulation",
}

GPS_NMEA_SENTENCE_TYPES = ("GGA", "GST", "HDT", "RMC", "VTG", "ZDA", "GSV", "GSA")

PPS_SOURCE_NAMES = {
    0: "disabled",
    1: "receiver_1",
    2: "receiver_2",
    3: "gpio",
    4: "generated",
}

GPIO_FEATURE_NAMES = {
    0: "unused",
    1: "gpio",
    2: "pps",
    3: "encoder_odometer",
    4: "event_timestamp",
    5: "uart",
}

GPIO_BEHAVIOR_NAMES = {
    0: "unused",
    1: "input_or_pps_input_or_encoder_a_or_timestamp_rising",
    2: "output_low_or_pps_output_or_encoder_b_or_timestamp_falling",
    3: "output_high_or_timestamp_either",
    0x21: "uart_port2_tx",
    0x22: "uart_port2_rx",
    0x31: "uart_port3_tx",
    0x32: "uart_port3_rx",
}

GPIO_PIN_MODE_FLAGS = {
    0x01: "open_drain",
    0x02: "pulldown",
    0x04: "pullup",
}

AIDING_SOURCE_NAMES = {
    0: "gnss_position_velocity",
    1: "gnss_heading",
    2: "altimeter",
    3: "speed_odometer",
    4: "magnetometer",
    5: "external_heading",
    6: "external_altimeter",
    7: "external_magnetometer",
    8: "body_frame_velocity",
}

INTERFACE_NAMES = {
    0: "all",
    1: "main_usb_or_uart",
    17: "uart_1",
    18: "uart_2",
    19: "uart_3",
    33: "usb_1",
    34: "usb_2",
}

COMMS_PROTOCOL_BITS = {
    0x00000001: "mip",
    0x00000100: "nmea",
    0x00000200: "rtcm",
    0x01000000: "spartn",
}

AIDING_MEASUREMENT_TYPE_NAMES = {
    1: "gnss",
    2: "dual_antenna",
    3: "heading",
    4: "pressure",
    5: "magnetometer",
    6: "speed",
    33: "aiding_pos_ecef",
    34: "aiding_pos_llh",
    35: "aiding_height_above_ellipsoid",
    40: "aiding_vel_ecef",
    41: "aiding_vel_ned",
    42: "aiding_vel_body_frame",
    49: "aiding_heading_true",
    50: "aiding_magnetic_field",
    51: "aiding_pressure",
}

MEASUREMENT_INDICATOR_BITS = {
    0x01: "enabled",
    0x02: "used",
    0x04: "residual_high_warning",
    0x08: "sample_time_warning",
    0x10: "configuration_error",
    0x20: "max_num_measurements_exceeded",
}


@dataclass
class MipPacket:
    descriptor_set: int
    payload: bytes
    received_time: float


@dataclass
class MipField:
    descriptor_set: int
    field_descriptor: int
    payload: bytes


class MipError(Exception):
    pass


def checksum(packet_without_checksum: bytes) -> bytes:
    a = 0
    b = 0
    for value in packet_without_checksum:
        a = (a + value) & 0xFF
        b = (b + a) & 0xFF
    return bytes((a, b))


def build_packet(descriptor_set: int, fields: Iterable[Tuple[int, bytes]]) -> bytes:
    payload = bytearray()
    for field_descriptor, field_payload in fields:
        field_len = len(field_payload) + 2
        if field_len > 255:
            raise ValueError("MIP field is too large")
        payload.extend((field_len, field_descriptor))
        payload.extend(field_payload)
    if len(payload) > 255:
        raise ValueError("MIP packet payload is too large")
    header_and_payload = SYNC + bytes((descriptor_set, len(payload))) + payload
    return header_and_payload + checksum(header_and_payload)


def iter_fields(packet: MipPacket) -> Iterable[MipField]:
    pos = 0
    payload = packet.payload
    while pos + 2 <= len(payload):
        field_len = payload[pos]
        if field_len < 2 or pos + field_len > len(payload):
            raise MipError(
                f"bad field length {field_len} in descriptor set 0x{packet.descriptor_set:02X}"
            )
        field_desc = payload[pos + 1]
        field_payload = payload[pos + 2 : pos + field_len]
        yield MipField(packet.descriptor_set, field_desc, field_payload)
        pos += field_len
    if pos != len(payload):
        raise MipError("trailing bytes in MIP payload")


class MipReader:
    def __init__(self, ser: serial.Serial):
        self.ser = ser
        self.buffer = bytearray()

    def read_packet(self, timeout_s: float = 1.0) -> Optional[MipPacket]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            needed = 6
            chunk = self.ser.read(max(1, needed - len(self.buffer)))
            if chunk:
                self.buffer.extend(chunk)

            while True:
                sync_index = self.buffer.find(SYNC)
                if sync_index < 0:
                    del self.buffer[:-1]
                    break
                if sync_index:
                    del self.buffer[:sync_index]
                if len(self.buffer) < 4:
                    break

                payload_len = self.buffer[3]
                total_len = 4 + payload_len + 2
                if len(self.buffer) < total_len:
                    more = self.ser.read(total_len - len(self.buffer))
                    if more:
                        self.buffer.extend(more)
                        continue
                    break

                raw = bytes(self.buffer[:total_len])
                del self.buffer[:total_len]
                if checksum(raw[:-2]) != raw[-2:]:
                    continue
                return MipPacket(raw[2], raw[4:-2], time.time())
        return None


def unpack(fmt: str, payload: bytes, field_name: str) -> Tuple:
    size = struct.calcsize(fmt)
    if len(payload) < size:
        raise MipError(f"{field_name} payload too short: {len(payload)} < {size}")
    return struct.unpack(fmt, payload[:size])


def rad_to_deg(value: float) -> float:
    return value * 180.0 / math.pi


def vector3(prefix: str, xyz: Tuple[float, float, float]) -> Dict[str, float]:
    return {
        f"{prefix}_x": xyz[0],
        f"{prefix}_y": xyz[1],
        f"{prefix}_z": xyz[2],
    }


def named_bits(value: int, bit_names: Dict[int, str]) -> List[str]:
    return [name for bit, name in bit_names.items() if value & bit]


def decode_filter_field(field: MipField) -> Dict:
    p = field.payload
    d = field.field_descriptor
    name = FILTER_FIELDS.get(d, f"unknown_filter_0x{d:02X}")

    if d == 0x01:
        lat, lon, h, flags = unpack(">dddH", p, name)
        data = {
            "latitude_deg": lat,
            "longitude_deg": lon,
            "ellipsoid_height_m": h,
            "valid_flags": flags,
        }
    elif d == 0x02:
        north, east, down, flags = unpack(">fffH", p, name)
        data = {"vel_n_mps": north, "vel_e_mps": east, "vel_d_mps": down, "valid_flags": flags}
    elif d == 0x03:
        qw, qx, qy, qz, flags = unpack(">ffffH", p, name)
        data = {"q_w": qw, "q_x": qx, "q_y": qy, "q_z": qz, "valid_flags": flags}
    elif d == 0x05:
        roll, pitch, yaw, flags = unpack(">fffH", p, name)
        data = {
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "roll_deg": rad_to_deg(roll),
            "pitch_deg": rad_to_deg(pitch),
            "yaw_deg": rad_to_deg(yaw),
            "valid_flags": flags,
        }
    elif d in (0x08, 0x09):
        north, east, down, flags = unpack(">fffH", p, name)
        suffix = "m" if d == 0x08 else "mps"
        data = {
            f"uncert_n_{suffix}": north,
            f"uncert_e_{suffix}": east,
            f"uncert_d_{suffix}": down,
            "valid_flags": flags,
        }
    elif d == 0x0A:
        roll, pitch, yaw, flags = unpack(">fffH", p, name)
        data = {
            "roll_uncert_rad": roll,
            "pitch_uncert_rad": pitch,
            "yaw_uncert_rad": yaw,
            "roll_uncert_deg": rad_to_deg(roll),
            "pitch_uncert_deg": rad_to_deg(pitch),
            "yaw_uncert_deg": rad_to_deg(yaw),
            "valid_flags": flags,
        }
    elif d in (0x0D, 0x0E, 0x13, 0x1C):
        x, y, z, flags = unpack(">fffH", p, name)
        labels = {
            0x0D: ("linear_accel", "mps2"),
            0x0E: ("angular_rate", "radps"),
            0x13: ("gravity", "mps2"),
            0x1C: ("comp_accel", "mps2"),
        }[d]
        data = {
            f"{labels[0]}_x_{labels[1]}": x,
            f"{labels[0]}_y_{labels[1]}": y,
            f"{labels[0]}_z_{labels[1]}": z,
            "valid_flags": flags,
        }
    elif d == 0x0F:
        magnitude, flags = unpack(">fH", p, name)
        data = {"gravity_magnitude_mps2": magnitude, "valid_flags": flags}
    elif d == 0x10:
        state, dynamics, flags = unpack(">HHH", p, name)
        data = {
            "filter_state": state,
            "filter_state_name": FILTER_MODE.get(state, "unknown"),
            "dynamics_mode": dynamics,
            "status_flags": flags,
            "status_flags_hex": f"0x{flags:04X}",
        }
    elif d == 0x11:
        tow, week, flags = unpack(">dHH", p, name)
        data = {"gps_tow_s": tow, "gps_week": week, "valid_flags": flags}
    elif d in (0x40, 0x42):
        x, y, z, flags = unpack(">dddH", p, name)
        if d == 0x40:
            data = {
                "ecef_x_m": x,
                "ecef_y_m": y,
                "ecef_z_m": z,
                "valid_flags": flags,
            }
        else:
            data = {
                "relative_n_m": x,
                "relative_e_m": y,
                "relative_d_m": z,
                "valid_flags": flags,
            }
    elif d == 0x41:
        x, y, z, flags = unpack(">fffH", p, name)
        data = {
            "ecef_vx_mps": x,
            "ecef_vy_mps": y,
            "ecef_vz_mps": z,
            "valid_flags": flags,
        }
    elif d == 0x43:
        receiver_id, tow, status = unpack(">BfH", p, name)
        data = {
            "receiver_id": receiver_id,
            "gps_tow_s": tow,
            "gnss_aid_status_flags": status,
            "flags_hex": f"0x{status:04X}",
            "reserved_hex": p[7:].hex(),
        }
    elif d == 0x44:
        tow, status = unpack(">fH", p, name)
        data = {
            "gps_tow_s": tow,
            "gnss_att_aid_status_flags": status,
            "flags_hex": f"0x{status:04X}",
            "reserved_hex": p[6:].hex(),
        }
    elif d == 0x45:
        tow, heading_type, reserved_0, reserved_1 = unpack(">fBff", p, name)
        data = {
            "gps_tow_s": tow,
            "heading_aid_type": heading_type,
            "heading_aid_type_name": {
                1: "dual_antenna",
                2: "external_message",
            }.get(heading_type, "unknown"),
            "reserved_0": reserved_0,
            "reserved_1": reserved_1,
        }
    elif d == 0x46:
        tow, source, meas_type, indicator = unpack(">fBBB", p, name)
        data = {
            "gps_tow_s": tow,
            "measurement_source": source,
            "measurement_type": meas_type,
            "measurement_type_name": AIDING_MEASUREMENT_TYPE_NAMES.get(meas_type, "unknown"),
            "indicator": indicator,
            "indicator_hex": f"0x{indicator:02X}",
            "indicator_names": named_bits(indicator, MEASUREMENT_INDICATOR_BITS),
            "enabled": bool(indicator & 0x01),
            "used": bool(indicator & 0x02),
            "sample_time_warning": bool(indicator & 0x08),
            "configuration_error": bool(indicator & 0x10),
            "raw_hex": p.hex(),
        }
    elif d == 0x50:
        frame_id, tx, ty, tz, qw, qx, qy, qz = unpack(">Bfffffff", p, name)
        data = {
            "frame_id": frame_id,
            "translation_error_x_m": tx,
            "translation_error_y_m": ty,
            "translation_error_z_m": tz,
            "attitude_error_q_w": qw,
            "attitude_error_q_x": qx,
            "attitude_error_q_y": qy,
            "attitude_error_q_z": qz,
        }
    elif d == 0x51:
        frame_id, tx, ty, tz, ax, ay, az = unpack(">Bffffff", p, name)
        data = {
            "frame_id": frame_id,
            "translation_uncert_x_m": tx,
            "translation_uncert_y_m": ty,
            "translation_uncert_z_m": tz,
            "attitude_uncert_x_rad": ax,
            "attitude_uncert_y_rad": ay,
            "attitude_uncert_z_rad": az,
        }
    elif d == 0xD3:
        tow, week, flags = unpack(">dHH", p, name)
        data = {"gps_tow_s": tow, "gps_week": week, "valid_flags": flags}
    elif d == 0xD5:
        nanoseconds = unpack(">Q", p, name)[0]
        data = {
            "reference_time_ns": nanoseconds,
            "reference_time_s": nanoseconds / 1e9,
        }
    elif d == 0xD7:
        nanoseconds, flags = unpack(">QH", p, name)
        data = {
            "external_time_ns": nanoseconds,
            "external_time_s": nanoseconds / 1e9,
            "valid_flags": flags,
            "valid_flags_hex": f"0x{flags:04X}",
            "external_time_valid": bool(flags & 0x0001),
        }
    else:
        data = {"raw_hex": p.hex()}

    return {"source": "EKF", "descriptor_set": "0x82", "field": name, "field_descriptor": f"0x{d:02X}", **data}


def decode_sensor_field(field: MipField) -> Dict:
    p = field.payload
    d = field.field_descriptor
    name = SENSOR_FIELDS.get(d, f"unknown_sensor_0x{d:02X}")

    if d in (0x01, 0x02, 0x03, 0x04, 0x05, 0x06):
        x, y, z = unpack(">fff", p, name)
        prefix_units = {
            0x01: ("raw_accel", "counts"),
            0x02: ("raw_gyro", "counts"),
            0x03: ("raw_mag", "counts"),
            0x04: ("accel", "g"),
            0x05: ("gyro", "radps"),
            0x06: ("mag", "gauss"),
        }[d]
        data = {
            f"{prefix_units[0]}_x_{prefix_units[1]}": x,
            f"{prefix_units[0]}_y_{prefix_units[1]}": y,
            f"{prefix_units[0]}_z_{prefix_units[1]}": z,
        }
    elif d == 0x0A:
        qw, qx, qy, qz = unpack(">ffff", p, name)
        data = {"q_w": qw, "q_x": qx, "q_y": qy, "q_z": qz}
    elif d == 0x0C:
        roll, pitch, yaw = unpack(">fff", p, name)
        data = {
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "roll_deg": rad_to_deg(roll),
            "pitch_deg": rad_to_deg(pitch),
            "yaw_deg": rad_to_deg(yaw),
        }
    elif d == 0x0E:
        ticks = unpack(">I", p, name)[0]
        data = {"internal_ticks": ticks}
    elif d == 0x12:
        tow, week, flags = unpack(">dHH", p, name)
        data = {"gps_tow_s": tow, "gps_week": week, "valid_flags": flags}
    elif d == 0x14:
        min_temp, max_temp, mean_temp = unpack(">fff", p, name)
        data = {"min_temp_c": min_temp, "max_temp_c": max_temp, "mean_temp_c": mean_temp}
    elif d == 0x17:
        pressure = unpack(">f", p, name)[0]
        data = {"pressure_mbar": pressure}
    elif d == 0x18:
        status = unpack(">H", p, name)[0]
        data = {"overrange_status": status, "status_hex": f"0x{status:04X}"}
    else:
        data = {"raw_hex": p.hex()}

    return {
        "source": "IMU_SENSOR",
        "descriptor_set": "0x80",
        "field": name,
        "field_descriptor": f"0x{d:02X}",
        **data,
    }


def decode_gnss_field(field: MipField) -> Dict:
    p = field.payload
    d = field.field_descriptor
    name = GNSS_FIELDS.get(d, f"unknown_gnss_0x{d:02X}")

    if d == 0x03:
        lat, lon, ell_h, msl_h, h_acc, v_acc, flags = unpack(">ddddffH", p, name)
        data = {
            "latitude_deg": lat,
            "longitude_deg": lon,
            "ellipsoid_height_m": ell_h,
            "msl_height_m": msl_h,
            "horizontal_accuracy_m": h_acc,
            "vertical_accuracy_m": v_acc,
            "valid_flags": flags,
        }
    elif d == 0x05:
        vn, ve, vd, speed, ground_speed, heading, speed_acc, heading_acc, flags = unpack(">ffffffffH", p, name)
        data = {
            "vel_n_mps": vn,
            "vel_e_mps": ve,
            "vel_d_mps": vd,
            "speed_mps": speed,
            "ground_speed_mps": ground_speed,
            "heading_deg": heading,
            "speed_accuracy_mps": speed_acc,
            "heading_accuracy_deg": heading_acc,
            "valid_flags": flags,
        }
    elif d == 0x09:
        tow, week, flags = unpack(">dHH", p, name)
        data = {"gps_tow_s": tow, "gps_week": week, "valid_flags": flags}
    elif d == 0x0B:
        fix_type, num_sv, fix_flags, valid_flags = unpack(">BBHH", p, name)
        data = {
            "fix_type": fix_type,
            "fix_type_name": GNSS_FIX_TYPE.get(fix_type, "unknown"),
            "num_sv": num_sv,
            "fix_flags": fix_flags,
            "fix_flags_hex": f"0x{fix_flags:04X}",
            "valid_flags": valid_flags,
        }
    elif d == 0x0D:
        receiver_state, antenna_state, antenna_power, valid_flags = unpack(">BBBB", p, name)
        data = {
            "receiver_state": receiver_state,
            "antenna_state": antenna_state,
            "antenna_power": antenna_power,
            "valid_flags": valid_flags,
        }
    else:
        data = {"raw_hex": p.hex()}

    return {
        "source": "GPS",
        "descriptor_set": f"0x{field.descriptor_set:02X}",
        "field": name,
        "field_descriptor": f"0x{d:02X}",
        **data,
    }


def decode_field(field: MipField) -> Optional[Dict]:
    if field.descriptor_set == DESC_FILTER:
        return decode_filter_field(field)
    if field.descriptor_set == DESC_SENSOR:
        return decode_sensor_field(field)
    if field.descriptor_set == DESC_GNSS_LEGACY or field.descriptor_set in DESC_GNSS_MODULES:
        return decode_gnss_field(field)
    return None


def detect_serial_port() -> str:
    ports = list(list_ports.comports())
    if not ports:
        os_name = platform.system()
        hint = "COMx" if os_name == "Windows" else "/dev/ttyACM0 or /dev/ttyUSB0"
        raise SystemExit(f"No serial ports found. Provide one with --port {hint}.")

    keywords = ("microstrain", "lord", "hbk", "3dm", "cv7", "inertial")
    generic_usb = ("usb serial", "usb-serial", "acm", "cp210", "ftdi")

    def score(port_info) -> int:
        text = " ".join(
            str(x or "")
            for x in (
                port_info.device,
                port_info.description,
                port_info.manufacturer,
                port_info.product,
                port_info.hwid,
            )
        ).lower()
        value = 0
        if any(k in text for k in keywords):
            value += 100
        if any(k in text for k in generic_usb):
            value += 30
        if platform.system() == "Windows" and port_info.device.upper().startswith("COM"):
            value += 5
        if platform.system() != "Windows" and (
            "ttyacm" in text or "ttyusb" in text or "serial/by-id" in text
        ):
            value += 5
        return value

    ranked = sorted(ports, key=score, reverse=True)
    return ranked[0].device


def port_identity_text(port_info) -> str:
    return " ".join(
        str(x or "")
        for x in (
            port_info.device,
            port_info.description,
            port_info.manufacturer,
            port_info.product,
            port_info.hwid,
        )
    ).lower()


def is_septentrio_port(port_info) -> bool:
    if port_info.vid == 0x152A and port_info.pid == 0x85C0:
        return True
    text = port_identity_text(port_info)
    return "septentrio" in text or "mosaic" in text


def looks_like_nmea_sentence(line: str) -> bool:
    if not line.startswith("$") or len(line) < 6:
        return False
    sentence = strip_nmea(line)
    if sentence is None:
        return False
    sentence_id, _fields = sentence
    message_type = sentence_id[-3:] if len(sentence_id) >= 3 else sentence_id
    return message_type in GPS_NMEA_SENTENCE_TYPES


def nmea_count_on_port(port: str, baud: int, timeout_s: float, min_sentences: int) -> int:
    count = 0
    with serial.Serial(port=port, baudrate=baud, timeout=0.25) as ser:
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        deadline = time.time() + max(0.1, timeout_s)
        while time.time() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if looks_like_nmea_sentence(line):
                count += 1
                if count >= min_sentences:
                    break
    return count


def detect_gps_nmea_port(
    baud: int,
    timeout_s: float = 3.0,
    exclude_port: Optional[str] = None,
    min_sentences: int = 3,
) -> Optional[str]:
    try:
        ports = list(list_ports.comports())
    except Exception as exc:
        print(
            json.dumps({"event": "gps_auto_detect_error", "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return None

    if not ports:
        return None

    exclude = (exclude_port or "").upper()
    known_gps_ids = {
        (0x152A, 0x85C0),  # Septentrio mosaic-H USB serial interface.
        (0x1546, 0x01A8),
        (0x10C4, 0xEA60),
        (0x067B, 0x2303),
        (0x0403, 0x6001),
    }

    def score(port_info) -> int:
        text = port_identity_text(port_info)
        value = 0
        if is_septentrio_port(port_info):
            value += 100
        if port_info.vid is not None and port_info.pid is not None:
            if (port_info.vid, port_info.pid) in known_gps_ids:
                value += 60
        if any(k in text for k in ("nmea", "gnss", "gps", "receiver")):
            value += 20
        if any(k in text for k in ("microstrain", "lord", "hbk", "3dm", "cv7", "inertial")):
            value -= 100
        return value

    candidates = [
        p for p in sorted(ports, key=score, reverse=True)
        if not exclude or p.device.upper() != exclude
    ]

    print(
        json.dumps(
            {
                "event": "gps_auto_detect_start",
                "baud": baud,
                "exclude_port": exclude_port,
                "candidates": [p.device for p in candidates],
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )

    best_port: Optional[str] = None
    best_count = 0
    for port_info in candidates:
        try:
            count = nmea_count_on_port(port_info.device, baud, timeout_s, min_sentences)
        except serial.SerialException as exc:
            print(
                json.dumps(
                    {
                        "event": "gps_auto_detect_probe",
                        "port": port_info.device,
                        "nmea_sentences": 0,
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            continue

        print(
            json.dumps(
                {
                    "event": "gps_auto_detect_probe",
                    "port": port_info.device,
                    "nmea_sentences": count,
                    "septentrio_candidate": is_septentrio_port(port_info),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        if count > best_count:
            best_port = port_info.device
            best_count = count
        if count >= min_sentences:
            print(
                json.dumps(
                    {
                        "event": "gps_auto_detect_selected",
                        "port": port_info.device,
                        "nmea_sentences": count,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return port_info.device

    if best_port:
        print(
            json.dumps(
                {
                    "event": "gps_auto_detect_selected_low_confidence",
                    "port": best_port,
                    "nmea_sentences": best_count,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    return best_port


def print_serial_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    for port_info in ports:
        vid_pid = "VID:PID=None"
        if port_info.vid is not None and port_info.pid is not None:
            vid_pid = f"VID:PID={port_info.vid:04X}:{port_info.pid:04X}"
        print(
            f"{port_info.device:24s} | {vid_pid:18s} | "
            f"{port_info.description or ''} | {port_info.manufacturer or ''}"
        )


def descriptor_rates(fields: Iterable[int], decimation: int) -> bytes:
    out = bytearray()
    for field in fields:
        out.extend(struct.pack(">BH", field, decimation))
    return bytes(out)


def make_message_format_payload(descriptor_set: int, fields: Iterable[int], decimation: int) -> bytes:
    field_list = list(fields)
    return bytes((MIP_FUNCTION_WRITE, descriptor_set, len(field_list))) + descriptor_rates(field_list, decimation)


def make_datastream_payload(descriptor_set: int, enable: bool) -> bytes:
    return bytes((MIP_FUNCTION_WRITE, descriptor_set, 1 if enable else 0))


def send_command(
    ser: serial.Serial,
    reader: MipReader,
    command_set: int,
    command_descriptor: int,
    command_payload: bytes = b"",
    wait_ack: bool = True,
    timeout_s: float = 1.0,
) -> bool:
    ser.write(build_packet(command_set, [(command_descriptor, command_payload)]))
    ser.flush()
    if not wait_ack:
        return True

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        packet = reader.read_packet(timeout_s=max(0.05, deadline - time.time()))
        if packet is None or packet.descriptor_set != command_set:
            continue
        for field in iter_fields(packet):
            if field.field_descriptor != FIELD_ACK_NACK or len(field.payload) < 2:
                continue
            acked_descriptor = field.payload[0]
            result_code = field.payload[1]
            if acked_descriptor == command_descriptor:
                if result_code == ACK_OK:
                    return True
                raise MipError(
                    f"command 0x{command_set:02X},0x{command_descriptor:02X} NACK 0x{result_code:02X}"
                )
    raise MipError(f"timeout waiting for ACK to command 0x{command_set:02X},0x{command_descriptor:02X}")


def command_response(
    ser: serial.Serial,
    reader: MipReader,
    command_set: int,
    command_descriptor: int,
    command_payload: bytes,
    response_descriptor: int,
    timeout_s: float = 1.0,
) -> bytes:
    ser.write(build_packet(command_set, [(command_descriptor, command_payload)]))
    ser.flush()

    deadline = time.time() + timeout_s
    ack_ok = False
    response_payload: Optional[bytes] = None
    while time.time() < deadline:
        packet = reader.read_packet(timeout_s=max(0.05, deadline - time.time()))
        if packet is None:
            continue
        try:
            fields = list(iter_fields(packet))
        except MipError:
            continue

        for field in fields:
            if (
                packet.descriptor_set == command_set
                and field.field_descriptor == FIELD_ACK_NACK
                and len(field.payload) >= 2
                and field.payload[0] == command_descriptor
            ):
                result_code = field.payload[1]
                if result_code != ACK_OK:
                    raise MipError(
                        f"command 0x{command_set:02X},0x{command_descriptor:02X} NACK 0x{result_code:02X}"
                    )
                ack_ok = True
            elif packet.descriptor_set == command_set and field.field_descriptor == response_descriptor:
                response_payload = field.payload

        if ack_ok and response_payload is not None:
            return response_payload

    raise MipError(
        f"timeout waiting for response 0x{command_set:02X},0x{response_descriptor:02X} "
        f"to command 0x{command_descriptor:02X}"
    )


def alignment_selector_names(selector: int) -> List[str]:
    names = []
    if selector & 0x01:
        names.append("dual_antenna")
    if selector & 0x02:
        names.append("kinematic")
    if selector & 0x04:
        names.append("magnetometer")
    if selector & 0x08:
        names.append("external")
    return names


def parse_initialization_configuration(payload: bytes) -> Dict:
    if len(payload) < 40:
        raise MipError(f"initialization configuration response too short: {payload.hex()}")
    wait_for_run, initial_source, alignment_selector = struct.unpack(">BBB", payload[:3])
    heading, pitch, roll = struct.unpack(">fff", payload[3:15])
    position = struct.unpack(">fff", payload[15:27])
    velocity = struct.unpack(">fff", payload[27:39])
    reference_frame = payload[39]
    return {
        "wait_for_run_command": wait_for_run,
        "initial_condition_source": initial_source,
        "initial_condition_source_name": {
            0: "auto_pos_vel_att",
            1: "auto_pos_vel_pitch_roll_user_heading",
            2: "auto_pos_vel_user_attitude",
            3: "manual",
        }.get(initial_source, "unknown"),
        "auto_heading_alignment_selector": alignment_selector,
        "auto_heading_alignment_names": alignment_selector_names(alignment_selector),
        "initial_heading_rad": heading,
        "initial_heading_deg": rad_to_deg(heading),
        "initial_pitch_rad": pitch,
        "initial_pitch_deg": rad_to_deg(pitch),
        "initial_roll_rad": roll,
        "initial_roll_deg": rad_to_deg(roll),
        "initial_position": {
            "x_or_lat": position[0],
            "y_or_lon": position[1],
            "z_or_height": position[2],
        },
        "initial_velocity": {
            "x_or_north": velocity[0],
            "y_or_east": velocity[1],
            "z_or_down": velocity[2],
        },
        "reference_frame": reference_frame,
        "reference_frame_name": {1: "ecef", 2: "llh"}.get(reference_frame, "unused_or_unknown"),
    }


def read_initialization_configuration(
    ser: serial.Serial,
    reader: MipReader,
    timeout_s: float,
) -> Dict:
    payload = command_response(
        ser,
        reader,
        DESC_FILTER_CMD,
        CMD_FILTER_INITIALIZATION_CONFIGURATION,
        bytes((MIP_FUNCTION_READ,)),
        REPLY_FILTER_INITIALIZATION_CONFIGURATION,
        timeout_s=timeout_s,
    )
    out = parse_initialization_configuration(payload)
    out["payload_hex"] = payload.hex()
    return out


def make_user_heading_initialization_payload(heading_deg: float) -> bytes:
    heading_rad = math.radians(heading_deg)
    return (
        bytes((MIP_FUNCTION_WRITE, 0x00, 0x01, 0x00))
        + struct.pack(">fff", heading_rad, 0.0, 0.0)
        + struct.pack(">fff", 0.0, 0.0, 0.0)
        + struct.pack(">fff", 0.0, 0.0, 0.0)
        + bytes((0x02,))
    )


def send_user_heading_initialization_on_serial(
    ser: serial.Serial,
    reader: MipReader,
    timeout_s: float,
    heading_deg: Optional[float],
) -> List[Dict]:
    if heading_deg is None:
        return []
    payload = make_user_heading_initialization_payload(heading_deg)
    send_command(
        ser,
        reader,
        DESC_FILTER_CMD,
        CMD_FILTER_INITIALIZATION_CONFIGURATION,
        payload,
        timeout_s=timeout_s,
    )
    return [
        {
            "step": "initialization_user_heading",
            "descriptor_set": f"0x{DESC_FILTER_CMD:02X}",
            "field": f"0x{CMD_FILTER_INITIALIZATION_CONFIGURATION:02X}",
            "ack": "ACK_OK",
            "heading_deg": heading_deg,
            "payload_hex": payload.hex(),
        }
    ]


def send_filter_control_commands_on_serial(
    ser: serial.Serial,
    reader: MipReader,
    timeout_s: float,
    reset_filter: bool,
    run_filter: bool,
) -> List[Dict]:
    steps: List[Dict] = []
    if reset_filter:
        send_command(ser, reader, DESC_FILTER_CMD, CMD_FILTER_RESET_FILTER, b"", timeout_s=timeout_s)
        steps.append(
            {
                "step": "filter_reset",
                "descriptor_set": f"0x{DESC_FILTER_CMD:02X}",
                "field": f"0x{CMD_FILTER_RESET_FILTER:02X}",
                "ack": "ACK_OK",
            }
        )
        time.sleep(0.2)
    if run_filter:
        send_command(ser, reader, DESC_FILTER_CMD, CMD_FILTER_RUN, b"", timeout_s=timeout_s)
        steps.append(
            {
                "step": "filter_run",
                "descriptor_set": f"0x{DESC_FILTER_CMD:02X}",
                "field": f"0x{CMD_FILTER_RUN:02X}",
                "ack": "ACK_OK",
            }
        )
    return steps


def run_filter_control_commands(
    port: str,
    baud: int,
    read_timeout_s: float,
    init_heading_deg: Optional[float],
    reset_filter: bool,
    run_filter: bool,
    pretty: bool,
) -> int:
    with serial.Serial(port=port, baudrate=baud, timeout=read_timeout_s) as ser:
        reader = MipReader(ser)
        steps = send_user_heading_initialization_on_serial(
            ser,
            reader,
            read_timeout_s,
            init_heading_deg,
        )
        if steps:
            time.sleep(0.1)
        steps.extend(
            send_filter_control_commands_on_serial(
                ser,
                reader,
                read_timeout_s,
                reset_filter,
                run_filter,
            )
        )

    result = {
        "event": "filter_control",
        "port": port,
        "baud": baud,
        "init_heading_deg": init_heading_deg,
        "reset_filter": reset_filter,
        "run_filter": run_filter,
        "steps": steps,
    }
    if pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


def make_poll_data_payload(descriptor_set: int, fields: Iterable[int], suppress_ack: bool = False) -> bytes:
    field_list = list(fields)
    return bytes((descriptor_set, 1 if suppress_ack else 0, len(field_list))) + bytes(field_list)


def make_poll_message_payload(fields: Iterable[int], suppress_ack: bool = False) -> bytes:
    field_list = list(fields)
    payload = bytearray((1 if suppress_ack else 0, len(field_list)))
    for field in field_list:
        # Decimation must be zero for poll commands.
        payload.extend(struct.pack(">BH", field, 0))
    return bytes(payload)


def poll_data_set(
    ser: serial.Serial,
    reader: MipReader,
    descriptor_set: int,
    fields: Iterable[int],
    timeout_s: float,
) -> List[Dict]:
    ser.write(
        build_packet(
            DESC_3DM_CMD,
            [(CMD_3DM_POLL_DATA, make_poll_data_payload(descriptor_set, fields))],
        )
    )
    ser.flush()

    deadline = time.time() + timeout_s
    ack_ok = False
    decoded_items: List[Dict] = []
    while time.time() < deadline:
        packet = reader.read_packet(timeout_s=max(0.05, deadline - time.time()))
        if packet is None:
            continue

        try:
            packet_fields = list(iter_fields(packet))
        except MipError:
            continue

        if packet.descriptor_set == DESC_3DM_CMD:
            for field in packet_fields:
                if field.field_descriptor == FIELD_ACK_NACK and len(field.payload) >= 2:
                    if field.payload[0] != CMD_3DM_POLL_DATA:
                        continue
                    result_code = field.payload[1]
                    if result_code != ACK_OK:
                        raise MipError(
                            f"poll data 0x{descriptor_set:02X} NACK 0x{result_code:02X}"
                        )
                    ack_ok = True
            continue

        if packet.descriptor_set == descriptor_set:
            decoded_items.extend(decoded_packet_fields(packet, debug=True))
            if decoded_items and ack_ok:
                return decoded_items

    if decoded_items:
        return decoded_items
    raise MipError(f"timeout polling data descriptor set 0x{descriptor_set:02X}")


def poll_message_set(
    ser: serial.Serial,
    reader: MipReader,
    command_descriptor: int,
    expected_descriptor_sets: Iterable[int],
    fields: Iterable[int],
    timeout_s: float,
) -> List[Dict]:
    expected_sets = set(expected_descriptor_sets)
    ser.write(
        build_packet(
            DESC_3DM_CMD,
            [(command_descriptor, make_poll_message_payload(fields))],
        )
    )
    ser.flush()

    deadline = time.time() + timeout_s
    ack_ok = False
    decoded_items: List[Dict] = []
    while time.time() < deadline:
        packet = reader.read_packet(timeout_s=max(0.05, deadline - time.time()))
        if packet is None:
            continue

        try:
            packet_fields = list(iter_fields(packet))
        except MipError:
            continue

        if packet.descriptor_set == DESC_3DM_CMD:
            for field in packet_fields:
                if field.field_descriptor == FIELD_ACK_NACK and len(field.payload) >= 2:
                    if field.payload[0] != command_descriptor:
                        continue
                    result_code = field.payload[1]
                    if result_code != ACK_OK:
                        raise MipError(
                            f"poll command 0x{command_descriptor:02X} NACK 0x{result_code:02X}"
                        )
                    ack_ok = True
            continue

        if packet.descriptor_set in expected_sets:
            decoded_items.extend(decoded_packet_fields(packet, debug=True))
            if decoded_items and ack_ok:
                return decoded_items

    if decoded_items:
        return decoded_items
    raise MipError(
        "timeout polling message descriptor set(s) "
        + ",".join(f"0x{value:02X}" for value in sorted(expected_sets))
    )


def poll_with_fallback(
    primary_callback,
    fallback_callback,
) -> Dict:
    try:
        return primary_callback()
    except Exception as primary_exc:
        try:
            fallback = fallback_callback()
            if isinstance(fallback, dict):
                fallback["primary_error"] = str(primary_exc)
            return fallback
        except Exception as fallback_exc:
            return {
                "error": str(primary_exc),
                "fallback_error": str(fallback_exc),
            }


def protocol_names(mask: int) -> List[str]:
    names = named_bits(mask, COMMS_PROTOCOL_BITS)
    unknown = mask
    for bit in COMMS_PROTOCOL_BITS:
        unknown &= ~bit
    if unknown:
        names.append(f"unknown_0x{unknown:08X}")
    return names


def pin_mode_names(mask: int) -> List[str]:
    names = named_bits(mask, GPIO_PIN_MODE_FLAGS)
    unknown = mask
    for bit in GPIO_PIN_MODE_FLAGS:
        unknown &= ~bit
    if unknown:
        names.append(f"unknown_0x{unknown:02X}")
    return names


def safe_status_call(name: str, callback) -> Dict:
    try:
        value = callback()
        if isinstance(value, dict):
            return value
        return {"value": value}
    except Exception as exc:
        return {"error": str(exc)}


def read_pps_source(ser: serial.Serial, reader: MipReader, timeout_s: float) -> Dict:
    payload = command_response(
        ser,
        reader,
        DESC_3DM_CMD,
        CMD_3DM_PPS_SOURCE,
        bytes((MIP_FUNCTION_READ,)),
        REPLY_3DM_PPS_SOURCE,
        timeout_s=timeout_s,
    )
    source = payload[0] if payload else None
    return {
        "source": source,
        "source_name": PPS_SOURCE_NAMES.get(source, "unknown") if source is not None else None,
    }


def read_gpio_config(ser: serial.Serial, reader: MipReader, pin: int, timeout_s: float) -> Dict:
    payload = command_response(
        ser,
        reader,
        DESC_3DM_CMD,
        CMD_3DM_GPIO_CONFIG,
        bytes((MIP_FUNCTION_READ, pin)),
        REPLY_3DM_GPIO_CONFIG,
        timeout_s=timeout_s,
    )
    if len(payload) < 4:
        raise MipError(f"GPIO{pin} response too short: {payload.hex()}")
    resp_pin, feature, behavior, pin_mode = unpack(">BBBB", payload, f"gpio{pin}_config")
    return {
        "pin": resp_pin,
        "feature": feature,
        "feature_name": GPIO_FEATURE_NAMES.get(feature, "unknown"),
        "behavior": behavior,
        "behavior_name": GPIO_BEHAVIOR_NAMES.get(behavior, "unknown"),
        "pin_mode": pin_mode,
        "pin_mode_hex": f"0x{pin_mode:02X}",
        "pin_mode_names": pin_mode_names(pin_mode),
    }


def read_interface_control(
    ser: serial.Serial,
    reader: MipReader,
    interface: int,
    timeout_s: float,
) -> Dict:
    payload = command_response(
        ser,
        reader,
        DESC_SYSTEM_CMD,
        CMD_SYSTEM_INTERFACE_CONTROL,
        bytes((MIP_FUNCTION_READ, interface)),
        REPLY_SYSTEM_INTERFACE_CONTROL,
        timeout_s=timeout_s,
    )
    if len(payload) < 9:
        raise MipError(f"interface {interface} response too short: {payload.hex()}")
    port = payload[0]
    incoming = int.from_bytes(payload[1:5], "big")
    outgoing = int.from_bytes(payload[5:9], "big")
    return {
        "interface": port,
        "interface_name": INTERFACE_NAMES.get(port, "unknown"),
        "incoming_protocols_hex": f"0x{incoming:08X}",
        "incoming_protocols": protocol_names(incoming),
        "outgoing_protocols_hex": f"0x{outgoing:08X}",
        "outgoing_protocols": protocol_names(outgoing),
    }


def read_aiding_enable(
    ser: serial.Serial,
    reader: MipReader,
    source: int,
    timeout_s: float,
) -> Dict:
    payload = command_response(
        ser,
        reader,
        DESC_FILTER_CMD,
        CMD_FILTER_AIDING_MEASUREMENT_ENABLE,
        bytes((MIP_FUNCTION_READ,)) + source.to_bytes(2, "big"),
        REPLY_FILTER_AIDING_MEASUREMENT_ENABLE,
        timeout_s=timeout_s,
    )
    if len(payload) < 3:
        raise MipError(f"aiding source {source} response too short: {payload.hex()}")
    resp_source = int.from_bytes(payload[0:2], "big")
    enabled = bool(payload[2])
    return {
        "source": resp_source,
        "source_name": AIDING_SOURCE_NAMES.get(resp_source, "unknown"),
        "enabled": enabled,
    }


def read_base_rate(
    ser: serial.Serial,
    reader: MipReader,
    descriptor_set: int,
    timeout_s: float,
) -> Dict:
    payload = command_response(
        ser,
        reader,
        DESC_3DM_CMD,
        CMD_3DM_GET_BASE_RATE,
        bytes((descriptor_set,)),
        REPLY_3DM_BASE_RATE,
        timeout_s=timeout_s,
    )
    if len(payload) < 3:
        raise MipError(f"base rate response too short: {payload.hex()}")
    resp_desc = payload[0]
    rate = int.from_bytes(payload[1:3], "big")
    return {
        "descriptor_set": f"0x{resp_desc:02X}",
        "rate_hz": rate,
    }


def collect_latest_by_field(items: List[Dict]) -> Dict[str, Dict]:
    latest: Dict[str, Dict] = {}
    for item in items:
        field_name = str(item.get("field", item.get("field_descriptor", "unknown")))
        if field_name == "aid_measurement_summary":
            meas_name = item.get("measurement_type_name", "unknown")
            meas_source = item.get("measurement_source", "unknown")
            field_name = f"{field_name}_{meas_name}_src{meas_source}"
        latest[field_name] = item
    return latest


def collect_stream_snapshot(reader: MipReader, duration_s: float, read_timeout_s: float) -> Dict[str, Dict]:
    started = time.time()
    sections: Dict[str, List[Dict]] = {
        "ekf_filter": [],
        "imu_sensor": [],
        "gnss": [],
        "other": [],
    }
    packet_counts: Dict[str, int] = {}

    while time.time() - started < duration_s:
        remaining = max(0.0, duration_s - (time.time() - started))
        packet = reader.read_packet(timeout_s=min(read_timeout_s, max(0.05, remaining)))
        if packet is None:
            continue

        descriptor_key = f"0x{packet.descriptor_set:02X}"
        packet_counts[descriptor_key] = packet_counts.get(descriptor_key, 0) + 1

        try:
            decoded_items = decoded_packet_fields(packet, debug=True)
        except MipError:
            continue

        for item in decoded_items:
            source = item.get("source")
            if source == "EKF":
                sections["ekf_filter"].append(item)
            elif source == "IMU_SENSOR":
                sections["imu_sensor"].append(item)
            elif source == "GPS":
                sections["gnss"].append(item)
            else:
                sections["other"].append(item)

    snapshot: Dict[str, Dict] = {
        "meta": {
            "source": "existing_stream",
            "duration_s": duration_s,
            "packet_counts": packet_counts,
        }
    }
    for section_name, items in sections.items():
        snapshot[section_name] = {
            "source": "existing_stream",
            "decoded_fields": len(items),
            "fields": collect_latest_by_field(items),
        }
        if not items:
            snapshot[section_name]["error"] = f"no {section_name} fields seen in {duration_s:.2f}s"
    return snapshot


def prefer_valid_status(primary: Dict, fallback: Dict) -> Dict:
    fields = primary.get("fields")
    if isinstance(fields, dict) and fields:
        if fallback.get("fields"):
            primary["stream_fallback_available"] = True
        return primary
    if fallback.get("fields"):
        out = dict(fallback)
        if primary.get("error"):
            out["primary_error"] = primary.get("error")
        if primary.get("fallback_error"):
            out["primary_fallback_error"] = primary.get("fallback_error")
        return out
    return primary


def run_status_snapshot(
    port: str,
    baud: int,
    read_timeout_s: float,
    status_timeout_s: float,
    status_listen_s: float,
    configure_streams_for_status: bool,
    rate_hz: int,
    assume_base_rate_hz: int,
    stream_preset: str,
    init_heading_deg: Optional[float],
    reset_filter: bool,
    run_filter: bool,
    pretty: bool,
) -> int:
    with serial.Serial(port=port, baudrate=baud, timeout=read_timeout_s) as ser:
        reader = MipReader(ser)
        if configure_streams_for_status:
            configure_streams(
                ser=ser,
                reader=reader,
                rate_hz=rate_hz,
                debug=True,
                assume_base_rate_hz=assume_base_rate_hz,
                stream_preset=stream_preset,
            )

        filter_control_steps: List[Dict] = []
        if init_heading_deg is not None:
            filter_control_steps.extend(
                send_user_heading_initialization_on_serial(
                    ser,
                    reader,
                    status_timeout_s,
                    init_heading_deg,
                )
            )
            time.sleep(0.1)
        if reset_filter or run_filter:
            filter_control_steps.extend(
                send_filter_control_commands_on_serial(
                    ser,
                    reader,
                    status_timeout_s,
                    reset_filter,
                    run_filter,
                )
            )
            time.sleep(0.5)

        config = {
            "pps_source": safe_status_call(
                "pps_source", lambda: read_pps_source(ser, reader, status_timeout_s)
            ),
            "gpio": {
                f"gpio{pin}": safe_status_call(
                    f"gpio{pin}", lambda pin=pin: read_gpio_config(ser, reader, pin, status_timeout_s)
                )
                for pin in (1, 2, 3, 4)
            },
            "interfaces": {
                "main": safe_status_call(
                    "interface_main",
                    lambda: read_interface_control(ser, reader, 1, status_timeout_s),
                ),
                "uart2": safe_status_call(
                    "interface_uart2",
                    lambda: read_interface_control(ser, reader, 18, status_timeout_s),
                ),
                "usb1": safe_status_call(
                    "interface_usb1",
                    lambda: read_interface_control(ser, reader, 33, status_timeout_s),
                ),
                "usb2": safe_status_call(
                    "interface_usb2",
                    lambda: read_interface_control(ser, reader, 34, status_timeout_s),
                ),
            },
            "aiding_enable": {
                AIDING_SOURCE_NAMES[source]: safe_status_call(
                    AIDING_SOURCE_NAMES[source],
                    lambda source=source: read_aiding_enable(ser, reader, source, status_timeout_s),
                )
                for source in sorted(AIDING_SOURCE_NAMES)
            },
            "initialization": safe_status_call(
                "initialization_configuration",
                lambda: read_initialization_configuration(ser, reader, status_timeout_s),
            ),
            "base_rates": {
                "imu_sensor": safe_status_call(
                    "base_rate_sensor",
                    lambda: read_base_rate(ser, reader, DESC_SENSOR, status_timeout_s),
                ),
                "gnss_legacy": safe_status_call(
                    "base_rate_gnss_legacy",
                    lambda: read_base_rate(ser, reader, DESC_GNSS_LEGACY, status_timeout_s),
                ),
                "filter": safe_status_call(
                    "base_rate_filter",
                    lambda: read_base_rate(ser, reader, DESC_FILTER, status_timeout_s),
                ),
            },
        }

        filter_fields = [
            0x11,
            0x10,
            0x01,
            0x02,
            0x05,
            0x08,
            0x09,
            0x0A,
            0x46,
            0x50,
            0x51,
            0xD3,
            0xD5,
            0xD7,
        ]
        filter_core_fields = [
            0x11,
            0x10,
            0x01,
            0x02,
            0x05,
            0x08,
            0x09,
            0x0A,
            0x46,
        ]
        sensor_fields = [0x04, 0x05, 0x06, 0x0C, 0x12, 0x14, 0x17, 0x18]
        gnss_fields = [0x03, 0x05, 0x09, 0x0B, 0x0D]

        latest_filter_poll = poll_with_fallback(
            lambda: {
                "fields": collect_latest_by_field(
                    poll_data_set(
                        ser,
                        reader,
                        DESC_FILTER,
                        filter_fields,
                        status_timeout_s,
                    )
                )
            },
            lambda: {
                "fields": collect_latest_by_field(
                    poll_data_set(
                        ser,
                        reader,
                        DESC_FILTER,
                        filter_core_fields,
                        status_timeout_s,
                    )
                )
            },
        )
        latest_sensor_poll = safe_status_call(
            "poll_sensor",
            lambda: {
                "fields": collect_latest_by_field(
                    poll_data_set(
                        ser,
                        reader,
                        DESC_SENSOR,
                        sensor_fields,
                        status_timeout_s,
                    )
                )
            },
        )
        latest_gnss_poll = poll_with_fallback(
            lambda: {
                "fields": collect_latest_by_field(
                    poll_data_set(ser, reader, DESC_GNSS_LEGACY, gnss_fields, status_timeout_s)
                )
            },
            lambda: {
                "fields": collect_latest_by_field(
                    poll_data_set(ser, reader, 0x91, gnss_fields, status_timeout_s)
                )
            },
        )

        stream_snapshot = collect_stream_snapshot(
            reader=reader,
            duration_s=status_listen_s,
            read_timeout_s=read_timeout_s,
        )

        latest_filter = prefer_valid_status(latest_filter_poll, stream_snapshot["ekf_filter"])
        latest_sensor = prefer_valid_status(latest_sensor_poll, stream_snapshot["imu_sensor"])
        latest_gnss = prefer_valid_status(latest_gnss_poll, stream_snapshot["gnss"])

        latest_poll_attempts = {
            "ekf_filter": latest_filter_poll,
            "imu_sensor": latest_sensor_poll,
            "gnss": latest_gnss_poll,
        }

        latest_stream = {
            "meta": stream_snapshot["meta"],
            "ekf_filter": stream_snapshot["ekf_filter"],
            "imu_sensor": stream_snapshot["imu_sensor"],
            "gnss": stream_snapshot["gnss"],
        }

        decoded_for_summary: List[Dict] = []
        for section in (latest_filter, latest_sensor, latest_gnss):
            fields = section.get("fields")
            if isinstance(fields, dict):
                decoded_for_summary.extend(fields.values())

        result = {
            "event": "imu_status_snapshot",
            "host_time_unix_s": time.time(),
            "port": port,
            "baud": baud,
            "configured_streams_for_status": configure_streams_for_status,
            "filter_control": {
                "init_heading_deg": init_heading_deg,
                "reset_filter": reset_filter,
                "run_filter": run_filter,
                "steps": filter_control_steps,
            },
            "config": config,
            "latest": {
                "ekf_filter": latest_filter,
                "imu_sensor": latest_sensor,
                "gnss": latest_gnss,
                "poll_attempts": latest_poll_attempts,
                "stream_snapshot": latest_stream,
            },
            "ekf_summary": summarize_decoded_fields(decoded_for_summary),
        }

    if pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


def configure_streams(
    ser: serial.Serial,
    reader: MipReader,
    rate_hz: int,
    debug: bool,
    assume_base_rate_hz: int,
    stream_preset: str,
) -> None:
    decimation = max(1, int(round(assume_base_rate_hz / max(1, rate_hz))))

    send_command(ser, reader, DESC_BASE, CMD_BASE_SET_TO_IDLE)

    core_filter_fields = [
        0x11,
        0x10,
        0x01,
        0x02,
        0x05,
    ]
    csv_filter_fields = core_filter_fields + [
        0x08,
        0x09,
        0x0A,
    ]
    full_filter_fields = core_filter_fields + [
        0x03,
        0x08,
        0x09,
        0x0A,
        0x0D,
        0x0E,
        0x13,
    ]
    if stream_preset == "core":
        filter_field_candidates = [core_filter_fields]
    elif stream_preset == "csv":
        filter_field_candidates = [csv_filter_fields, core_filter_fields]
    else:
        filter_field_candidates = [
            full_filter_fields + [0x46, 0x50, 0x51, 0xD3, 0xD5, 0xD7],
            full_filter_fields + [0x46, 0xD3, 0xD7],
            full_filter_fields,
            csv_filter_fields,
            core_filter_fields,
        ]
    last_format_error: Optional[MipError] = None
    for filter_fields in filter_field_candidates:
        try:
            send_command(
                ser,
                reader,
                DESC_3DM_CMD,
                CMD_3DM_MESSAGE_FORMAT,
                make_message_format_payload(DESC_FILTER, filter_fields, decimation),
            )
            break
        except MipError as exc:
            last_format_error = exc
    else:
        assert last_format_error is not None
        raise last_format_error

    send_command(
        ser,
        reader,
        DESC_3DM_CMD,
        CMD_3DM_DATASTREAM_CONTROL,
        make_datastream_payload(DESC_FILTER, True),
    )

    if debug:
        sensor_fields = [0x04, 0x05, 0x06, 0x0A, 0x0C, 0x12, 0x14, 0x17, 0x18]
        gnss_fields = [0x03, 0x05, 0x09, 0x0B, 0x0D]
        send_command(
            ser,
            reader,
            DESC_3DM_CMD,
            CMD_3DM_MESSAGE_FORMAT,
            make_message_format_payload(DESC_SENSOR, sensor_fields, decimation),
        )
        send_command(
            ser,
            reader,
            DESC_3DM_CMD,
            CMD_3DM_DATASTREAM_CONTROL,
            make_datastream_payload(DESC_SENSOR, True),
        )
        # On 7-series products GNSS module streams are usually 0x91..0x95.
        # Some legacy tools use 0x81, so enable both forms if the device accepts them.
        for desc_set in (DESC_GNSS_LEGACY, 0x91):
            try:
                send_command(
                    ser,
                    reader,
                    DESC_3DM_CMD,
                    CMD_3DM_MESSAGE_FORMAT,
                    make_message_format_payload(desc_set, gnss_fields, decimation),
                    timeout_s=0.5,
                )
                send_command(
                    ser,
                    reader,
                    DESC_3DM_CMD,
                    CMD_3DM_DATASTREAM_CONTROL,
                    make_datastream_payload(desc_set, True),
                    timeout_s=0.5,
                )
            except MipError:
                pass

    send_command(ser, reader, DESC_BASE, CMD_BASE_RESUME)


def decoded_packet_fields(packet: MipPacket, debug: bool) -> List[Dict]:
    decoded_items: List[Dict] = []
    for field in iter_fields(packet):
        if not debug and field.descriptor_set != DESC_FILTER:
            continue
        decoded = decode_field(field)
        if decoded is None:
            if not debug:
                continue
            decoded = {
                "source": "OTHER",
                "descriptor_set": f"0x{field.descriptor_set:02X}",
                "field_descriptor": f"0x{field.field_descriptor:02X}",
                "raw_hex": field.payload.hex(),
            }
        decoded["host_time_unix_s"] = packet.received_time
        decoded_items.append(decoded)
    return decoded_items


def summarize_decoded_fields(decoded_items: List[Dict]) -> Optional[Dict]:
    if not decoded_items:
        return None

    summary: Dict[str, object] = {
        "source": "EKF_SUMMARY",
        "host_time_unix_s": decoded_items[-1].get("host_time_unix_s"),
    }
    found = False

    for item in decoded_items:
        field = item.get("field")
        if field == "filter_status":
            summary["filter_state"] = item.get("filter_state")
            summary["filter_state_name"] = item.get("filter_state_name")
            summary["status_flags_hex"] = item.get("status_flags_hex")
            found = True
        elif field == "attitude_euler":
            summary["roll_deg"] = item.get("roll_deg")
            summary["pitch_deg"] = item.get("pitch_deg")
            summary["yaw_deg"] = item.get("yaw_deg")
            summary["attitude_valid"] = item.get("valid_flags")
            found = True
        elif field == "position_llh":
            summary["latitude_deg"] = item.get("latitude_deg")
            summary["longitude_deg"] = item.get("longitude_deg")
            summary["ellipsoid_height_m"] = item.get("ellipsoid_height_m")
            summary["position_valid"] = item.get("valid_flags")
            found = True
        elif field == "velocity_ned":
            summary["vel_n_mps"] = item.get("vel_n_mps")
            summary["vel_e_mps"] = item.get("vel_e_mps")
            summary["vel_d_mps"] = item.get("vel_d_mps")
            summary["velocity_valid"] = item.get("valid_flags")
            found = True
        elif field == "euler_uncertainty":
            summary["roll_uncert_deg"] = item.get("roll_uncert_deg")
            summary["pitch_uncert_deg"] = item.get("pitch_uncert_deg")
            summary["yaw_uncert_deg"] = item.get("yaw_uncert_deg")
        elif field == "gps_fix_info":
            summary["gps_fix_type"] = item.get("fix_type_name")
            summary["gps_num_sv"] = item.get("num_sv")
            found = True

    return summary if found else None


EKF_CSV_FIELDS = [
    "host_time_unix_s",
    "sample_time_source",
    "sample_time_s",
    "skip_detected",
    "skip_count_estimate",
    "gap_s",
    "expected_period_s",
    "fields",
    "filter_state",
    "filter_state_name",
    "status_flags_hex",
    "gps_tow_s",
    "gps_week",
    "reference_time_ns",
    "reference_time_s",
    "latitude_deg",
    "longitude_deg",
    "ellipsoid_height_m",
    "position_valid",
    "vel_n_mps",
    "vel_e_mps",
    "vel_d_mps",
    "velocity_valid",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "attitude_valid",
    "roll_uncert_deg",
    "pitch_uncert_deg",
    "yaw_uncert_deg",
    "uncert_n_m",
    "uncert_e_m",
    "uncert_d_m",
    "uncert_n_mps",
    "uncert_e_mps",
    "uncert_d_mps",
    "ecef_x_m",
    "ecef_y_m",
    "ecef_z_m",
    "ecef_vx_mps",
    "ecef_vy_mps",
    "ecef_vz_mps",
    "relative_n_m",
    "relative_e_m",
    "relative_d_m",
    "decoded_json",
]

GPS_CSV_FIELDS = [
    "host_time_unix_s",
    "sample_time_source",
    "sample_time_s",
    "skip_detected",
    "skip_count_estimate",
    "gap_s",
    "expected_period_s",
    "gps_data_source",
    "port",
    "baud",
    "sentence_index",
    "raw_nmea",
    "talker",
    "message_type",
    "checksum_present",
    "checksum_ok",
    "parsed_ok",
    "valid_fix",
    "nmea_time_utc",
    "nmea_date",
    "gps_datetime_utc",
    "fields",
    "gps_tow_s",
    "gps_week",
    "latitude_deg",
    "longitude_deg",
    "ellipsoid_height_m",
    "altitude_m",
    "geoid_separation_m",
    "msl_height_m",
    "horizontal_accuracy_m",
    "vertical_accuracy_m",
    "lat_sigma_m",
    "lon_sigma_m",
    "alt_sigma_m",
    "gst_rms_m",
    "gst_sigma_major_m",
    "gst_sigma_minor_m",
    "gst_orientation_deg",
    "vel_n_mps",
    "vel_e_mps",
    "vel_d_mps",
    "speed_mps",
    "ground_speed_mps",
    "ground_speed_knots",
    "ground_speed_kmph",
    "track_true_deg",
    "track_magnetic_deg",
    "heading_deg",
    "heading_true_deg",
    "magnetic_variation_deg",
    "speed_accuracy_mps",
    "heading_accuracy_deg",
    "fix_type",
    "fix_type_name",
    "fix_quality",
    "fix_quality_name",
    "num_sv",
    "num_sats",
    "fix_flags_hex",
    "rmc_status",
    "rmc_mode",
    "vtg_mode",
    "dgps_age_s",
    "dgps_station_id",
    "zda_day",
    "zda_month",
    "zda_year",
    "zda_local_zone_hours",
    "zda_local_zone_minutes",
    "receiver_state",
    "antenna_state",
    "antenna_power",
    "valid_flags",
    "raw_fields_json",
    "decoded_json",
]

SKIP_CSV_FIELDS = [
    "host_time_unix_s",
    "stream",
    "sample_time_source",
    "previous_sample_time_s",
    "sample_time_s",
    "gap_s",
    "expected_period_s",
    "skip_count_estimate",
    "threshold",
    "fields",
]


def parse_float_maybe(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def parse_int_maybe(value: object) -> Optional[int]:
    number = parse_float_maybe(value)
    return int(number) if number is not None else None


def nmea_checksum(line: str) -> Tuple[bool, bool]:
    text = line.strip()
    if not text.startswith("$") or "*" not in text:
        return False, False
    body, checksum = text[1:].split("*", 1)
    if len(checksum) < 2:
        return True, False
    calc = 0
    for char in body:
        calc ^= ord(char)
    try:
        expected = int(checksum[:2], 16)
    except ValueError:
        return True, False
    return True, calc == expected


def strip_nmea(line: str) -> Optional[Tuple[str, List[str]]]:
    text = line.strip()
    if not text.startswith("$"):
        return None
    body = text[1:].split("*", 1)[0]
    parts = body.split(",")
    if not parts or not parts[0]:
        return None
    return parts[0], parts[1:]


def nmea_coord_to_decimal(value: str, direction: str, is_latitude: bool) -> Optional[float]:
    raw = parse_float_maybe(value)
    if raw is None:
        return None
    degrees = int(raw // 100)
    minutes = raw - degrees * 100
    decimal = degrees + minutes / 60.0
    if direction in ("S", "W"):
        decimal = -decimal
    if is_latitude and not (-90.0 <= decimal <= 90.0):
        return None
    if not is_latitude and not (-180.0 <= decimal <= 180.0):
        return None
    return decimal


def nmea_time_to_text(value: str) -> str:
    text = str(value or "").strip()
    if len(text) < 6:
        return text
    return f"{text[0:2]}:{text[2:4]}:{text[4:]}"


def nmea_time_to_seconds(value: str) -> Optional[float]:
    text = str(value or "").strip()
    if len(text) < 6:
        return None
    try:
        hours = int(text[0:2])
        minutes = int(text[2:4])
        seconds = float(text[4:])
    except ValueError:
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds


def nmea_date_to_text(value: str) -> str:
    text = str(value or "").strip()
    if len(text) != 6:
        return text
    day = text[0:2]
    month = text[2:4]
    year2 = int(text[4:6])
    year = 2000 + year2 if year2 < 80 else 1900 + year2
    return f"{year:04d}-{month}-{day}"


def combine_gps_datetime(date_text: str, time_text: str) -> str:
    if not date_text or not time_text:
        return ""
    return f"{date_text}T{time_text}Z"


def parse_external_gps_nmea(
    line: str,
    port: str,
    baud: int,
    sentence_index: int,
    received_time: float,
) -> Optional[Dict[str, object]]:
    stripped = strip_nmea(line)
    if stripped is None:
        return None

    sentence_id, fields = stripped
    checksum_present, checksum_ok = nmea_checksum(line)
    talker = sentence_id[:2] if len(sentence_id) >= 5 else ""
    message_type = sentence_id[-3:] if len(sentence_id) >= 3 else sentence_id
    row: Dict[str, object] = {
        "host_time_unix_s": received_time,
        "sample_time_source": "host_time_unix_s",
        "sample_time_s": received_time,
        "gps_data_source": "external_nmea",
        "port": port,
        "baud": baud,
        "sentence_index": sentence_index,
        "raw_nmea": line.strip(),
        "talker": talker,
        "message_type": message_type,
        "checksum_present": int(checksum_present),
        "checksum_ok": int(checksum_ok),
        "parsed_ok": 0,
        "valid_fix": "",
        "fields": message_type,
        "raw_fields_json": json.dumps(fields, separators=(",", ":")),
    }

    if checksum_present and not checksum_ok:
        return row

    try:
        if message_type == "GGA":
            parse_external_gga(row, fields)
        elif message_type == "RMC":
            parse_external_rmc(row, fields)
        elif message_type == "VTG":
            parse_external_vtg(row, fields)
        elif message_type == "GST":
            parse_external_gst(row, fields)
        elif message_type == "HDT":
            parse_external_hdt(row, fields)
        elif message_type == "ZDA":
            parse_external_zda(row, fields)
        else:
            row["parsed_ok"] = 1
        return row
    except Exception as exc:
        row["raw_fields_json"] = json.dumps(
            {"fields": fields, "parse_error": str(exc)},
            separators=(",", ":"),
        )
        return row


def set_nmea_sample_time(row: Dict[str, object], raw_time: str) -> None:
    seconds = nmea_time_to_seconds(raw_time)
    if seconds is not None:
        row["sample_time_source"] = "nmea_time_utc"
        row["sample_time_s"] = seconds
        row["nmea_time_utc"] = nmea_time_to_text(raw_time)


def parse_external_gga(row: Dict[str, object], fields: List[str]) -> None:
    raw_time = fields[0] if len(fields) > 0 else ""
    lat = nmea_coord_to_decimal(fields[1], fields[2], True) if len(fields) > 2 else None
    lon = nmea_coord_to_decimal(fields[3], fields[4], False) if len(fields) > 4 else None
    fix_quality = parse_int_maybe(fields[5]) if len(fields) > 5 else None
    num_sats = parse_int_maybe(fields[6]) if len(fields) > 6 else None
    hdop = parse_float_maybe(fields[7]) if len(fields) > 7 else None
    altitude = parse_float_maybe(fields[8]) if len(fields) > 8 else None
    geoid = parse_float_maybe(fields[10]) if len(fields) > 10 else None
    dgps_age = parse_float_maybe(fields[12]) if len(fields) > 12 else None
    station = fields[13] if len(fields) > 13 else ""

    set_nmea_sample_time(row, raw_time)
    row.update(
        {
            "parsed_ok": 1,
            "valid_fix": int(fix_quality is not None and fix_quality > 0 and lat is not None and lon is not None),
            "latitude_deg": lat,
            "longitude_deg": lon,
            "altitude_m": altitude,
            "msl_height_m": altitude,
            "geoid_separation_m": geoid,
            "fix_quality": fix_quality,
            "fix_quality_name": GPS_FIX_QUALITY_NAMES.get(fix_quality or 0, ""),
            "num_sats": num_sats,
            "num_sv": num_sats,
            "hdop": hdop,
            "dgps_age_s": dgps_age,
            "dgps_station_id": station,
        }
    )


def parse_external_rmc(row: Dict[str, object], fields: List[str]) -> None:
    raw_time = fields[0] if len(fields) > 0 else ""
    status = fields[1] if len(fields) > 1 else ""
    lat = nmea_coord_to_decimal(fields[2], fields[3], True) if len(fields) > 3 else None
    lon = nmea_coord_to_decimal(fields[4], fields[5], False) if len(fields) > 5 else None
    speed_knots = parse_float_maybe(fields[6]) if len(fields) > 6 else None
    track = parse_float_maybe(fields[7]) if len(fields) > 7 else None
    date_text = nmea_date_to_text(fields[8]) if len(fields) > 8 else ""
    mag_var = parse_float_maybe(fields[9]) if len(fields) > 9 else None
    if mag_var is not None and len(fields) > 10 and fields[10] == "W":
        mag_var = -mag_var
    mode = fields[11] if len(fields) > 11 else ""

    set_nmea_sample_time(row, raw_time)
    time_text = str(row.get("nmea_time_utc") or "")
    row.update(
        {
            "parsed_ok": 1,
            "valid_fix": int(status == "A" and lat is not None and lon is not None),
            "nmea_date": date_text,
            "gps_datetime_utc": combine_gps_datetime(date_text, time_text),
            "latitude_deg": lat,
            "longitude_deg": lon,
            "ground_speed_knots": speed_knots,
            "ground_speed_mps": speed_knots * 0.514444 if speed_knots is not None else None,
            "speed_mps": speed_knots * 0.514444 if speed_knots is not None else None,
            "track_true_deg": track,
            "heading_deg": track,
            "magnetic_variation_deg": mag_var,
            "rmc_status": status,
            "rmc_mode": mode,
        }
    )


def parse_external_vtg(row: Dict[str, object], fields: List[str]) -> None:
    true_track = parse_float_maybe(fields[0]) if len(fields) > 0 else None
    mag_track = parse_float_maybe(fields[2]) if len(fields) > 2 else None
    speed_knots = parse_float_maybe(fields[4]) if len(fields) > 4 else None
    speed_kmph = parse_float_maybe(fields[6]) if len(fields) > 6 else None
    mode = fields[8] if len(fields) > 8 else ""
    row.update(
        {
            "parsed_ok": 1,
            "track_true_deg": true_track,
            "track_magnetic_deg": mag_track,
            "heading_deg": true_track,
            "ground_speed_knots": speed_knots,
            "ground_speed_mps": speed_knots * 0.514444 if speed_knots is not None else None,
            "speed_mps": speed_knots * 0.514444 if speed_knots is not None else None,
            "ground_speed_kmph": speed_kmph,
            "vtg_mode": mode,
        }
    )


def parse_external_gst(row: Dict[str, object], fields: List[str]) -> None:
    raw_time = fields[0] if len(fields) > 0 else ""
    set_nmea_sample_time(row, raw_time)
    row.update(
        {
            "parsed_ok": 1,
            "gst_rms_m": parse_float_maybe(fields[1]) if len(fields) > 1 else None,
            "gst_sigma_major_m": parse_float_maybe(fields[2]) if len(fields) > 2 else None,
            "gst_sigma_minor_m": parse_float_maybe(fields[3]) if len(fields) > 3 else None,
            "gst_orientation_deg": parse_float_maybe(fields[4]) if len(fields) > 4 else None,
            "lat_sigma_m": parse_float_maybe(fields[5]) if len(fields) > 5 else None,
            "lon_sigma_m": parse_float_maybe(fields[6]) if len(fields) > 6 else None,
            "alt_sigma_m": parse_float_maybe(fields[7]) if len(fields) > 7 else None,
            "horizontal_accuracy_m": parse_float_maybe(fields[1]) if len(fields) > 1 else None,
            "vertical_accuracy_m": parse_float_maybe(fields[7]) if len(fields) > 7 else None,
        }
    )


def parse_external_hdt(row: Dict[str, object], fields: List[str]) -> None:
    heading = parse_float_maybe(fields[0]) if fields else None
    row.update({"parsed_ok": 1, "heading_true_deg": heading, "heading_deg": heading})


def parse_external_zda(row: Dict[str, object], fields: List[str]) -> None:
    raw_time = fields[0] if len(fields) > 0 else ""
    day = parse_int_maybe(fields[1]) if len(fields) > 1 else None
    month = parse_int_maybe(fields[2]) if len(fields) > 2 else None
    year = parse_int_maybe(fields[3]) if len(fields) > 3 else None
    date_text = f"{year:04d}-{month:02d}-{day:02d}" if day and month and year else ""
    set_nmea_sample_time(row, raw_time)
    time_text = str(row.get("nmea_time_utc") or "")
    row.update(
        {
            "parsed_ok": 1,
            "nmea_date": date_text,
            "gps_datetime_utc": combine_gps_datetime(date_text, time_text),
            "zda_day": day,
            "zda_month": month,
            "zda_year": year,
            "zda_local_zone_hours": parse_int_maybe(fields[4]) if len(fields) > 4 else None,
            "zda_local_zone_minutes": parse_int_maybe(fields[5]) if len(fields) > 5 else None,
        }
    )


class CsvRecorder:
    def __init__(
        self,
        output_dir: str,
        prefix: str,
        expected_ekf_hz: float,
        expected_gps_hz: float,
        skip_threshold: float,
        skip_check: bool,
    ):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.ekf_path = os.path.join(output_dir, f"{prefix}_ekf_fused_{stamp}.csv")
        self.gps_path = os.path.join(output_dir, f"{prefix}_gps_raw_{stamp}.csv")
        self.skip_path = os.path.join(output_dir, f"{prefix}_skip_validation_{stamp}.csv")
        self.expected_periods = {
            "EKF": 1.0 / expected_ekf_hz if expected_ekf_hz > 0 else 0.0,
            "GPS": 1.0 / expected_gps_hz if expected_gps_hz > 0 else 0.0,
        }
        self.skip_threshold = skip_threshold
        self.skip_check = skip_check
        self.previous_sample_times: Dict[str, float] = {}
        self._lock = threading.Lock()

        self._ekf_file = open(self.ekf_path, "w", newline="", encoding="utf-8")
        self._gps_file = open(self.gps_path, "w", newline="", encoding="utf-8")
        self._skip_file = open(self.skip_path, "w", newline="", encoding="utf-8")
        self.ekf_writer = csv.DictWriter(self._ekf_file, fieldnames=EKF_CSV_FIELDS, extrasaction="ignore")
        self.gps_writer = csv.DictWriter(self._gps_file, fieldnames=GPS_CSV_FIELDS, extrasaction="ignore")
        self.skip_writer = csv.DictWriter(self._skip_file, fieldnames=SKIP_CSV_FIELDS, extrasaction="ignore")
        self.ekf_writer.writeheader()
        self.gps_writer.writeheader()
        self.skip_writer.writeheader()

    def close(self) -> None:
        with self._lock:
            for handle in (self._ekf_file, self._gps_file, self._skip_file):
                handle.flush()
                handle.close()

    def record(self, decoded_items: List[Dict]) -> Tuple[int, int, int]:
        with self._lock:
            ekf_items = [item for item in decoded_items if item.get("source") == "EKF"]
            gps_items = [item for item in decoded_items if item.get("source") == "GPS"]
            ekf_rows = gps_rows = skip_rows = 0

            if ekf_items:
                row = self._build_row("EKF", ekf_items)
                skip = self._skip_row("EKF", row)
                self.ekf_writer.writerow(row)
                ekf_rows += 1
                if skip:
                    self.skip_writer.writerow(skip)
                    skip_rows += 1

            if gps_items:
                row = self._build_row("GPS", gps_items)
                row["gps_data_source"] = "cv7_mip_gnss"
                skip = self._skip_row("GPS", row)
                self.gps_writer.writerow(row)
                gps_rows += 1
                if skip:
                    self.skip_writer.writerow(skip)
                    skip_rows += 1

        return ekf_rows, gps_rows, skip_rows

    def record_external_gps(self, row: Dict[str, object]) -> Tuple[int, int]:
        with self._lock:
            skip = self._skip_row("GPS", row)
            self.gps_writer.writerow(row)
            skip_rows = 0
            if skip:
                self.skip_writer.writerow(skip)
                skip_rows = 1
            return 1, skip_rows

    def _build_row(self, stream: str, items: List[Dict]) -> Dict:
        host_time = items[-1].get("host_time_unix_s")
        fields = ",".join(str(item.get("field", "")) for item in items)
        row: Dict[str, object] = {
            "host_time_unix_s": host_time,
            "fields": fields,
            "decoded_json": json.dumps(items, separators=(",", ":"), sort_keys=True),
        }

        for item in items:
            field = item.get("field")
            for key, value in item.items():
                if key in ("source", "descriptor_set", "field", "field_descriptor", "host_time_unix_s"):
                    continue
                if key not in row or row.get(key) in (None, ""):
                    row[key] = value

            if stream == "EKF":
                self._copy_ekf_aliases(row, item, field)

        sample_source, sample_time = self._sample_time(stream, row)
        row["sample_time_source"] = sample_source
        row["sample_time_s"] = sample_time
        return row

    @staticmethod
    def _copy_ekf_aliases(row: Dict, item: Dict, field: object) -> None:
        if field == "position_llh":
            row["position_valid"] = item.get("valid_flags")
        elif field == "velocity_ned":
            row["velocity_valid"] = item.get("valid_flags")
        elif field == "attitude_euler":
            row["attitude_valid"] = item.get("valid_flags")

    @staticmethod
    def _sample_time(stream: str, row: Dict) -> Tuple[str, Optional[float]]:
        if stream == "EKF" and row.get("reference_time_s") is not None:
            return "reference_time_s", float(row["reference_time_s"])
        if row.get("gps_tow_s") is not None:
            return "gps_tow_s", float(row["gps_tow_s"])
        if row.get("host_time_unix_s") is not None:
            return "host_time_unix_s", float(row["host_time_unix_s"])
        return "none", None

    def _skip_row(self, stream: str, row: Dict) -> Optional[Dict]:
        if not self.skip_check:
            row["skip_detected"] = 0
            return None

        sample_time = row.get("sample_time_s")
        expected = self.expected_periods.get(stream, 0.0)
        sample_source = str(row.get("sample_time_source", "none"))
        time_key = f"{stream}:{sample_source}"
        previous = self.previous_sample_times.get(time_key)
        row["expected_period_s"] = expected
        row["skip_detected"] = 0

        if sample_time is None:
            return None
        sample_time = float(sample_time)
        self.previous_sample_times[time_key] = sample_time

        if previous is None or expected <= 0:
            return None

        gap = sample_time - previous
        row["gap_s"] = gap
        if gap <= expected * self.skip_threshold:
            return None

        skip_count = max(1, int(round(gap / expected)) - 1)
        row["skip_detected"] = 1
        row["skip_count_estimate"] = skip_count
        return {
            "host_time_unix_s": row.get("host_time_unix_s"),
            "stream": stream,
            "sample_time_source": row.get("sample_time_source"),
            "previous_sample_time_s": previous,
            "sample_time_s": sample_time,
            "gap_s": gap,
            "expected_period_s": expected,
            "skip_count_estimate": skip_count,
            "threshold": self.skip_threshold,
            "fields": row.get("fields"),
        }

    def flush(self) -> None:
        with self._lock:
            self._ekf_file.flush()
            self._gps_file.flush()
            self._skip_file.flush()


class ExternalGpsReaderThread(threading.Thread):
    def __init__(
        self,
        port: str,
        baud: int,
        read_timeout_s: float,
        recorder: CsvRecorder,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True, name="ExternalGpsReader")
        self.port = port
        self.baud = baud
        self.read_timeout_s = read_timeout_s
        self.recorder = recorder
        self.stop_event = stop_event
        self.lines_read = 0
        self.rows_written = 0
        self.skip_rows = 0
        self.bad_checksum = 0
        self.open_error = ""
        self.last_error = ""

    def run(self) -> None:
        try:
            with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.read_timeout_s) as ser:
                try:
                    ser.set_buffer_size(rx_size=65536, tx_size=4096)
                except Exception:
                    pass
                while not self.stop_event.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("ascii", errors="ignore").strip()
                    if not line.startswith("$"):
                        continue

                    self.lines_read += 1
                    row = parse_external_gps_nmea(
                        line=line,
                        port=self.port,
                        baud=self.baud,
                        sentence_index=self.lines_read,
                        received_time=time.time(),
                    )
                    if row is None:
                        continue
                    if row.get("checksum_present") and not row.get("checksum_ok"):
                        self.bad_checksum += 1
                    rows, skips = self.recorder.record_external_gps(row)
                    self.rows_written += rows
                    self.skip_rows += skips
        except serial.SerialException as exc:
            self.open_error = str(exc)
            self.last_error = str(exc)
        except Exception as exc:
            self.last_error = str(exc)


def print_decoded_items(decoded_items: List[Dict], pretty: bool, summary: bool) -> int:
    if summary:
        summary_item = summarize_decoded_fields(decoded_items)
        if not summary_item:
            return 0
        if pretty:
            print(json.dumps(summary_item, indent=2, sort_keys=True))
        else:
            print(json.dumps(summary_item, separators=(",", ":"), sort_keys=True))
        return 1

    printed = 0
    for decoded in decoded_items:
        if pretty:
            print(json.dumps(decoded, indent=2, sort_keys=True))
        else:
            print(json.dumps(decoded, separators=(",", ":"), sort_keys=True))
        printed += 1
    return printed


def print_packet(packet: MipPacket, debug: bool, pretty: bool, summary: bool) -> int:
    return print_decoded_items(
        decoded_packet_fields(packet, debug=debug),
        pretty=pretty,
        summary=summary,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read MicroStrain 3DM-CV7 fused EKF/filter output over USB serial."
    )
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit.")
    parser.add_argument("--port", help="Serial port, e.g. COM5 on Windows or /dev/ttyACM0 on Linux.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate. USB CDC often ignores this.")
    parser.add_argument(
        "--gps-port",
        help='External GPS NMEA serial port to record at the same time, e.g. COM8. Use "auto" to detect it.',
    )
    parser.add_argument("--gps-baud", type=int, default=115200, help="External GPS NMEA serial baud rate.")
    parser.add_argument("--gps-read-timeout-s", type=float, default=0.25, help="External GPS serial read timeout.")
    parser.add_argument("--gps-auto-detect-s", type=float, default=3.0, help="Seconds to listen on each candidate GPS port during auto-detect.")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Pull one non-destructive command-line IMU status snapshot, including config and latest polled data.",
    )
    parser.add_argument(
        "--status-timeout-s",
        type=float,
        default=1.5,
        help="Timeout per status read/poll command.",
    )
    parser.add_argument(
        "--status-listen-s",
        type=float,
        default=2.0,
        help="Seconds to listen to existing MIP streaming during --status for latest EKF/IMU/GNSS data.",
    )
    parser.add_argument(
        "--reset-filter",
        action="store_true",
        help="Send the CV7 EKF Reset Filter command before reading/status output.",
    )
    parser.add_argument(
        "--run-filter",
        action="store_true",
        help="Send the CV7 EKF Run Filter command before reading/status output.",
    )
    parser.add_argument(
        "--init-heading-deg",
        type=float,
        help=(
            "Temporarily set EKF initialization to automatic position/velocity/pitch/roll "
            "with this user heading before reset/run."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Print IMU sensor, GPS/GNSS, EKF, and unknown fields.")
    parser.add_argument(
        "--configure",
        action="store_true",
        help="Configure CV7 MIP output streams before reading. Otherwise only listens to current device config.",
    )
    parser.add_argument("--rate-hz", type=int, default=100, help="Requested output rate for --configure.")
    parser.add_argument(
        "--assume-base-rate-hz",
        type=int,
        default=1000,
        help="CV7 base rate used to compute MIP decimation when --configure is set.",
    )
    parser.add_argument(
        "--stream-preset",
        choices=("full", "csv", "core"),
        default="full",
        help=(
            "Fields requested from the EKF stream when --configure is set. "
            "'csv' is recommended for 500 Hz logging."
        ),
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of compact JSON lines.")
    parser.add_argument("--summary", action="store_true", help="Print one compact summary per packet instead of every field.")
    parser.add_argument(
        "--print-hz",
        type=float,
        default=0.0,
        help="Limit console printing rate. 0 means print every decoded packet.",
    )
    parser.add_argument("--record-csv", action="store_true", help="Write GPS raw, EKF fused, and skip validation CSV files.")
    parser.add_argument("--csv-prefix", default="cv7", help="Prefix for CSV files written beside this script.")
    parser.add_argument("--skip-check", action="store_true", help="Enable skipped-sample/gap validation CSV logging.")
    parser.add_argument("--expected-ekf-hz", type=float, default=100.0, help="Expected EKF output rate for skip validation.")
    parser.add_argument("--expected-gps-hz", type=float, default=10.0, help="Expected GPS output rate for skip validation.")
    parser.add_argument(
        "--skip-gap-threshold",
        type=float,
        default=1.5,
        help="Flag skipping when sample gap exceeds threshold times expected period.",
    )
    parser.add_argument("--duration-s", type=float, default=0.0, help="Stop after this many seconds. 0 means forever.")
    parser.add_argument("--read-timeout-s", type=float, default=1.0, help="Serial read timeout.")
    return parser.parse_args(argv)


def running_in_spyder() -> bool:
    spyder_env_keys = ("SPYDER_KERNEL_ID", "SPYDER_PARENT_PID", "SPYDER_ARGS")
    return any(os.environ.get(key) for key in spyder_env_keys) or any(
        name.startswith("spyder_kernels") for name in sys.modules
    )


def build_argv_from_spyder_settings() -> List[str]:
    argv: List[str] = [
        "--baud",
        str(SPYDER_BAUD),
        "--gps-baud",
        str(SPYDER_GPS_BAUD),
        "--rate-hz",
        str(SPYDER_RATE_HZ),
        "--assume-base-rate-hz",
        str(SPYDER_ASSUME_BASE_RATE_HZ),
        "--stream-preset",
        str(SPYDER_STREAM_PRESET),
        "--print-hz",
        str(SPYDER_PRINT_HZ),
        "--csv-prefix",
        str(SPYDER_CSV_PREFIX),
        "--expected-ekf-hz",
        str(SPYDER_EXPECTED_EKF_HZ),
        "--expected-gps-hz",
        str(SPYDER_EXPECTED_GPS_HZ),
        "--skip-gap-threshold",
        str(SPYDER_SKIP_GAP_THRESHOLD),
        "--duration-s",
        str(SPYDER_DURATION_S),
        "--read-timeout-s",
        str(SPYDER_READ_TIMEOUT_S),
    ]
    if SPYDER_PORT:
        argv.extend(["--port", SPYDER_PORT])
    if SPYDER_GPS_PORT:
        argv.extend(["--gps-port", SPYDER_GPS_PORT])
    if SPYDER_DEBUG:
        argv.append("--debug")
    if SPYDER_CONFIGURE:
        argv.append("--configure")
    if SPYDER_PRETTY_JSON:
        argv.append("--pretty")
    if SPYDER_SUMMARY_OUTPUT:
        argv.append("--summary")
    if SPYDER_RECORD_CSV:
        argv.append("--record-csv")
    if SPYDER_SKIP_CHECK:
        argv.append("--skip-check")
    return argv


def run_cv7_reader(
    port: Optional[str] = None,
    gps_port: Optional[str] = None,
    debug: bool = False,
    configure: bool = False,
    rate_hz: int = 100,
    stream_preset: str = "csv",
    duration_s: float = 0.0,
    baud: int = 115200,
    gps_baud: int = 115200,
    pretty_json: bool = False,
    summary: bool = True,
    print_hz: float = 2.0,
    record_csv: bool = True,
    expected_ekf_hz: Optional[float] = None,
) -> int:
    """Convenience function for Spyder Console or notebook-style use."""
    argv: List[str] = [
        "--baud",
        str(baud),
        "--gps-baud",
        str(gps_baud),
        "--rate-hz",
        str(rate_hz),
        "--stream-preset",
        stream_preset,
        "--print-hz",
        str(print_hz),
        "--duration-s",
        str(duration_s),
    ]
    if port:
        argv.extend(["--port", port])
    if gps_port:
        argv.extend(["--gps-port", gps_port])
    if debug:
        argv.append("--debug")
    if configure:
        argv.append("--configure")
    if pretty_json:
        argv.append("--pretty")
    if summary:
        argv.append("--summary")
    if record_csv:
        argv.append("--record-csv")
        argv.append("--skip-check")
        argv.extend(["--expected-ekf-hz", str(expected_ekf_hz or rate_hz)])
    return main(argv)


def run_cv7_status(
    port: Optional[str] = None,
    baud: int = 115200,
    pretty_json: bool = True,
    status_timeout_s: float = 1.5,
    status_listen_s: float = 2.0,
    init_heading_deg: Optional[float] = None,
    reset_filter: bool = False,
    run_filter: bool = False,
) -> int:
    """Convenience function for Spyder Console to pull one IMU status snapshot."""
    argv: List[str] = [
        "--status",
        "--baud",
        str(baud),
        "--status-timeout-s",
        str(status_timeout_s),
        "--status-listen-s",
        str(status_listen_s),
    ]
    if port:
        argv.extend(["--port", port])
    if pretty_json:
        argv.append("--pretty")
    if init_heading_deg is not None:
        argv.extend(["--init-heading-deg", str(init_heading_deg)])
    if reset_filter:
        argv.append("--reset-filter")
    if run_filter:
        argv.append("--run-filter")
    return main(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_ports:
        print_serial_ports()
        return 0

    os_name = platform.system()
    port = args.port or detect_serial_port()
    gps_port = args.gps_port
    gps_port_source = "command-line" if gps_port else "not_requested"
    if gps_port and gps_port.strip().lower() == "auto":
        gps_port = detect_gps_nmea_port(
            baud=args.gps_baud,
            timeout_s=args.gps_auto_detect_s,
            exclude_port=port,
        )
        gps_port_source = "auto-detected" if gps_port else "auto-detect-failed"
    elif args.record_csv and not gps_port:
        gps_port = detect_gps_nmea_port(
            baud=args.gps_baud,
            timeout_s=args.gps_auto_detect_s,
            exclude_port=port,
        )
        gps_port_source = "auto-detected" if gps_port else "auto-detect-failed"

    if args.record_csv and not gps_port:
        print(
            json.dumps(
                {
                    "event": "external_gps_auto_detect_failed",
                    "message": "GPS CSV will still be created, but only CV7 GNSS rows can be written unless a GPS NMEA port is found.",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )

    print(
        json.dumps(
            {
                "event": "open_serial",
                "os": os_name,
                "port": port,
                "baud": args.baud,
                "gps_port": gps_port,
                "gps_port_source": gps_port_source,
                "gps_baud": args.gps_baud,
                "status": args.status,
                "debug": args.debug,
                "configure": args.configure,
                "stream_preset": args.stream_preset,
                "init_heading_deg": args.init_heading_deg,
                "reset_filter": args.reset_filter,
                "run_filter": args.run_filter,
                "summary": args.summary,
                "print_hz": args.print_hz,
                "record_csv": args.record_csv,
                "skip_check": args.skip_check,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )

    if args.status:
        return run_status_snapshot(
            port=port,
            baud=args.baud,
            read_timeout_s=args.read_timeout_s,
            status_timeout_s=args.status_timeout_s,
            status_listen_s=args.status_listen_s,
            configure_streams_for_status=args.configure,
            rate_hz=args.rate_hz,
            assume_base_rate_hz=args.assume_base_rate_hz,
            stream_preset=args.stream_preset,
            init_heading_deg=args.init_heading_deg,
            reset_filter=args.reset_filter,
            run_filter=args.run_filter,
            pretty=args.pretty,
        )

    if args.init_heading_deg is not None or args.reset_filter or args.run_filter:
        return run_filter_control_commands(
            port=port,
            baud=args.baud,
            read_timeout_s=args.read_timeout_s,
            init_heading_deg=args.init_heading_deg,
            reset_filter=args.reset_filter,
            run_filter=args.run_filter,
            pretty=args.pretty,
        )

    recorder: Optional[CsvRecorder] = None
    gps_stop_event = threading.Event()
    gps_thread: Optional[ExternalGpsReaderThread] = None
    if args.record_csv:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        recorder = CsvRecorder(
            output_dir=output_dir,
            prefix=args.csv_prefix,
            expected_ekf_hz=args.expected_ekf_hz,
            expected_gps_hz=args.expected_gps_hz,
            skip_threshold=args.skip_gap_threshold,
            skip_check=args.skip_check,
        )
        print(
            json.dumps(
                {
                    "event": "csv_recording",
                    "ekf_csv": recorder.ekf_path,
                    "gps_csv": recorder.gps_path,
                    "skip_csv": recorder.skip_path,
                    "external_gps_port": gps_port,
                    "external_gps_port_source": gps_port_source,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    elif gps_port:
        print(
            json.dumps(
                {
                    "event": "external_gps_not_recorded",
                    "reason": "--gps-port requires --record-csv to write GPS rows",
                    "gps_port": gps_port,
                    "gps_port_source": gps_port_source,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )

    ekf_rows = 0
    gps_rows = 0
    skip_rows = 0
    packets = 0
    decoded_fields = 0

    with serial.Serial(port=port, baudrate=args.baud, timeout=args.read_timeout_s) as ser:
        try:
            reader = MipReader(ser)
            if recorder and gps_port:
                gps_thread = ExternalGpsReaderThread(
                    port=gps_port,
                    baud=args.gps_baud,
                    read_timeout_s=args.gps_read_timeout_s,
                    recorder=recorder,
                    stop_event=gps_stop_event,
                )
                gps_thread.start()
                print(
                    json.dumps(
                        {
                            "event": "external_gps_recording_started",
                            "gps_port": gps_port,
                            "gps_port_source": gps_port_source,
                            "gps_baud": args.gps_baud,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
            if args.configure:
                configure_streams(
                    ser=ser,
                    reader=reader,
                    rate_hz=args.rate_hz,
                    debug=args.debug,
                    assume_base_rate_hz=args.assume_base_rate_hz,
                    stream_preset=args.stream_preset,
                )
                print(
                    json.dumps(
                        {
                            "event": "configured",
                            "rate_hz": args.rate_hz,
                            "assume_base_rate_hz": args.assume_base_rate_hz,
                            "stream_preset": args.stream_preset,
                            "debug_streams": args.debug,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )

            started = time.time()
            last_print_time = 0.0
            last_flush_time = time.time()
            while True:
                if args.duration_s > 0 and time.time() - started >= args.duration_s:
                    break
                packet = reader.read_packet(timeout_s=args.read_timeout_s)
                if packet is None:
                    continue
                packets += 1
                try:
                    decoded_items = decoded_packet_fields(packet, debug=True)
                    if not args.debug:
                        printable_items = [item for item in decoded_items if item.get("source") == "EKF"]
                    else:
                        printable_items = decoded_items

                    if recorder:
                        new_ekf_rows, new_gps_rows, new_skip_rows = recorder.record(decoded_items)
                        ekf_rows += new_ekf_rows
                        gps_rows += new_gps_rows
                        skip_rows += new_skip_rows
                        if time.time() - last_flush_time >= 1.0:
                            recorder.flush()
                            last_flush_time = time.time()

                    now = time.time()
                    if args.print_hz > 0:
                        min_print_period = 1.0 / args.print_hz
                        if now - last_print_time < min_print_period:
                            continue
                        last_print_time = now
                    decoded_fields += print_decoded_items(
                        printable_items,
                        pretty=args.pretty,
                        summary=args.summary,
                    )
                except MipError as exc:
                    if args.debug:
                        print(
                            json.dumps(
                                {
                                    "source": "PARSER",
                                    "error": str(exc),
                                    "descriptor_set": f"0x{packet.descriptor_set:02X}",
                                },
                                sort_keys=True,
                            ),
                            file=sys.stderr,
                        )
        except KeyboardInterrupt:
            print(json.dumps({"event": "interrupted_by_user"}, sort_keys=True), file=sys.stderr)
        finally:
            gps_stop_event.set()
            if gps_thread:
                gps_thread.join(timeout=2.0)
            if recorder:
                recorder.flush()

    print(
        json.dumps(
            {
                "event": "done",
                "packets": packets,
                "decoded_fields": decoded_fields,
                "ekf_csv_rows": ekf_rows,
                "gps_csv_rows": gps_rows,
                "external_gps_csv_rows": gps_thread.rows_written if gps_thread else 0,
                "external_gps_lines": gps_thread.lines_read if gps_thread else 0,
                "external_gps_bad_checksum": gps_thread.bad_checksum if gps_thread else 0,
                "external_gps_skip_events": gps_thread.skip_rows if gps_thread else 0,
                "external_gps_error": gps_thread.last_error if gps_thread else "",
                "skip_events": skip_rows + (gps_thread.skip_rows if gps_thread else 0),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    if recorder:
        recorder.close()
    return 0


if __name__ == "__main__":
    is_spyder = running_in_spyder()
    if USE_SPYDER_SETTINGS_WHEN_AVAILABLE and is_spyder:
        exit_code = main(build_argv_from_spyder_settings())
    else:
        exit_code = main()
    if not is_spyder:
        raise SystemExit(exit_code)
