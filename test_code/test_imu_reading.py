# -*- coding: utf-8 -*-

import argparse
import struct
import sys
import time
import math
import numpy as np
import serial
import serial.tools.list_ports
from scipy.spatial.transform import Rotation as R


IMU_SYNC_BYTE_1 = 0xAA
IMU_SYNC_BYTE_2 = 0x55

IMU_CMD_QUAT = 32
IMU_CMD_RPY = 35
IMU_CMD_GRAVITY = 36
IMU_CMD_RAW = 41

IMU_CRC_BYTES = 2
G_TO_MPS2 = 9.81


def detect_os():
    if sys.platform.startswith("win"):
        return "Windows"
    if sys.platform.startswith("linux"):
        return "Linux"
    return "Unknown"


def get_default_imu_port(os_name):
    if os_name == "Windows":
        return "COM12"
    return "/dev/ttyACM2"


def list_serial_ports():
    try:
        ports = serial.tools.list_ports.comports()
    except Exception as exc:
        print(f"Error listing ports: {exc}")
        return

    if not ports:
        print("No serial ports found.")
        return

    for p in ports:
        if p.vid is not None:
            print(
                f"{p.device:20s} | "
                f"VID:PID={p.vid:04X}:{p.pid:04X} | "
                f"{p.description}"
            )
        else:
            print(
                f"{p.device:20s} | "
                f"VID:PID=None         | "
                f"{p.description}"
            )


def auto_detect_imu_port():
    imu_usb_ids = [
        (0x0483, 0x5740),
    ]

    try:
        ports = serial.tools.list_ports.comports()
    except Exception:
        return None

    print("Scanning serial ports for IMU...")

    for p in ports:
        vid = p.vid
        pid = p.pid

        if vid is None:
            print(
                f"{p.device:20s} | "
                f"VID:PID=None         | "
                f"{p.description}"
            )
            continue

        print(
            f"{p.device:20s} | "
            f"VID:PID={vid:04X}:{pid:04X} | "
            f"{p.description}"
        )

        for known_vid, known_pid in imu_usb_ids:
            if vid == known_vid and pid == known_pid:
                print(f"Detected IMU: {p.device}")
                return p.device

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="TransducerM IMU RAW and Quaternion logger in IMU Assistant style"
    )
    parser.add_argument("--imu-port", type=str, default=None)
    parser.add_argument("--imu-baud", type=int, default=115200)
    parser.add_argument("--output", type=str, default="rawData.csv")
    parser.add_argument("--quat-output", type=str, default="quaternionData.csv")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--print-every", type=float, default=0.25)
    parser.add_argument("--skip-crc", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    return parser.parse_args()


def normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


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
            self._buf.extend(self.ser.read(min(avail, 4096)))

    def read_packet(self):
        self._refill()

        while True:
            sync_idx = -1

            for i in range(len(self._buf) - 1):
                if (
                    self._buf[i] == IMU_SYNC_BYTE_1
                    and self._buf[i + 1] == IMU_SYNC_BYTE_2
                ):
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
            crc_bytes = bytes(self._buf[3 + length:3 + length + 2])

            self._buf = self._buf[total_needed:]

            if self.check_crc:
                crc_received = struct.unpack("<H", crc_bytes)[0]
                crc_calculated = crc16_modbus(length_byte + payload)

                if crc_received != crc_calculated:
                    self.crc_fail_count += 1
                    continue

            if len(payload) < 4:
                self.bad_packet_count += 1
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
                if len(data) < 28:
                    return None, None

                num_floats = (len(data) - 4) // 4
                fmt = "<I" + ("f" * num_floats)
                required_bytes = 4 + num_floats * 4
                vals = struct.unpack(fmt, data[:required_bytes])

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
                    "q1": vals[1],
                    "q2": vals[2],
                    "q3": vals[3],
                    "q4": vals[4],
                }

            if cmd_id == IMU_CMD_RPY:
                if len(data) < 16:
                    return None, None

                vals = struct.unpack("<Ifff", data[:16])

                return vals[0], {
                    "packet_type": "RPY",
                    "roll_deg": vals[1],
                    "pitch_deg": vals[2],
                    "yaw_deg": vals[3],
                }

            if cmd_id == IMU_CMD_GRAVITY:
                if len(data) < 16:
                    return None, None

                vals = struct.unpack("<Ifff", data[:16])

                return vals[0], {
                    "packet_type": "GRAVITY",
                    "gravity_x_g": vals[1],
                    "gravity_y_g": vals[2],
                    "gravity_z_g": vals[3],
                }

        except struct.error:
            return None, None

        return None, None


class RawDataMatStyleLogger:
    def __init__(self, filename="rawData.csv"):
        self.filename = filename
        self.f = open(filename, "w", newline="")
        self.count = 0

        self.f.write("rawData = [ \n")
        self.f.write("%   File created by Python IMU raw reader. \n")
        self.f.write("%   NAME:  timeStamp,   temperature,     gyroX,   gyroY,   gyroZ,     accX,   accY,   accZ,     magX,   magY,   magZ ;  \n")
        self.f.write("%   UNIT:   Second  ,       oC     ,     rad/s,   rad/s,   rad/s,       g ,     g ,     g ,     unit ,  unit ,  unit ; \n")

    def write_raw(
        self,
        imu_timestamp_us,
        gyro_rad_s,
        accel_g,
        mag_unit,
        temperature_c=None,
    ):
        time_s = float(imu_timestamp_us) / 1e6

        if temperature_c is None:
            temperature_c = math.nan

        if mag_unit is None:
            mag_unit = np.array([math.nan, math.nan, math.nan], dtype=float)

        self.f.write(
            f"         {time_s:12.6f},"
            f"       {temperature_c:12.6f},"
            f"      {gyro_rad_s[0]:14.9f},"
            f"    {gyro_rad_s[1]:14.9f},"
            f"    {gyro_rad_s[2]:14.9f},"
            f"       {accel_g[0]:14.9f},"
            f"    {accel_g[1]:14.9f},"
            f"    {accel_g[2]:14.9f},"
            f"       {mag_unit[0]:14.9f},"
            f"    {mag_unit[1]:14.9f},"
            f"    {mag_unit[2]:14.9f} ; \n"
        )

        self.count += 1

        if self.count % 500 == 0:
            self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.write("];\n")
            self.f.flush()
            self.f.close()
            self.f = None


class QuaternionMatStyleLogger:
    def __init__(self, filename="quaternionData.csv"):
        self.filename = filename
        self.f = open(filename, "w", newline="")
        self.count = 0

        self.f.write("quaternionData = [ \n")
        self.f.write("%   File created by Python IMU quaternion reader. \n")
        self.f.write("%   NAME:  timeStamp,    q1,   q2,   q3,   q4; \n")
        self.f.write("%   UNIT:   Second  ,    Quaternion(no unit) ; \n")

    def write_quat(self, imu_timestamp_us, q1, q2, q3, q4):
        time_s = float(imu_timestamp_us) / 1e6

        self.f.write(
            f"        {time_s:12.6f},"
            f"      {q1:14.9f},"
            f"    {q2:14.9f},"
            f"    {q3:14.9f},"
            f"    {q4:14.9f}; \n"
        )

        self.count += 1

        if self.count % 500 == 0:
            self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.write("];\n")
            self.f.flush()
            self.f.close()
            self.f = None


def validate_imu_connection(reader, timeout=2.0):
    print(f"Validating IMU for {timeout:.1f} seconds...")
    start = time.time()
    packets = 0

    while time.time() - start < timeout:
        ts, data = reader.read_packet()
        if data is not None:
            packets += 1
            if packets >= 5:
                print(f"IMU OK: received {packets} packets")
                return True

        time.sleep(0.0002)

    print(f"IMU validation failed: received {packets} packets")
    return False


def quat_to_scipy_xyzw(q1, q2, q3, q4):
    return normalize_quaternion(np.array([q2, q3, q4, q1], dtype=float))


def format_vec(v, precision=5):
    return "[" + ", ".join(f"{x:.{precision}f}" for x in v) + "]"


def main():
    args = parse_args()

    if args.list_ports:
        list_serial_ports()
        return

    os_name = detect_os()
    default_port = get_default_imu_port(os_name)
    auto_port = auto_detect_imu_port()

    imu_port = args.imu_port or auto_port or default_port
    imu_baud = args.imu_baud

    print(f"OS: {os_name}")
    print(f"IMU port: {imu_port}")
    print(f"IMU baud: {imu_baud}")
    print(f"Raw output file: {args.output}")
    print(f"Quaternion output file: {args.quat_output}")
    print(f"CRC check: {'off' if args.skip_crc else 'on'}")

    ser = serial.Serial(
        port=imu_port,
        baudrate=imu_baud,
        timeout=0,
    )

    reader = IMUPacketReader(
        ser,
        check_crc=not args.skip_crc,
    )

    raw_logger = RawDataMatStyleLogger(args.output)
    quat_logger = QuaternionMatStyleLogger(args.quat_output)

    if not validate_imu_connection(reader):
        raw_logger.close()
        quat_logger.close()
        ser.close()
        return

    latest_raw = None
    latest_gravity = None
    latest_quat = None
    latest_rpy = None

    raw_count = 0
    gravity_count = 0
    quat_count = 0
    rpy_count = 0

    start_time = time.time()
    rate_start_time = time.time()
    last_rate_time = time.time()
    last_print_time = time.time()

    print("")
    print("Start recording IMU RAW and Quaternion data.")
    print("Static validation target:")
    print("RAW accel norm should be close to 1 g.")
    print("Gravity norm should be close to 1 g.")
    print("RAW accel minus Gravity should be close to 0 g.")
    print("Gyro should be close to 0 rad/s when stationary.")
    print("")

    try:
        while True:
            if args.max_seconds > 0:
                if time.time() - start_time >= args.max_seconds:
                    print("Reached max recording time.")
                    break

            ts, data = reader.read_packet()

            if data is None:
                time.sleep(0.0002)
                continue

            packet_type = data.get("packet_type")

            if packet_type == "RAW":
                raw_count += 1

                gyro = np.array([
                    data["gyro_x_rad_s"],
                    data["gyro_y_rad_s"],
                    data["gyro_z_rad_s"],
                ], dtype=float)

                accel = np.array([
                    data["accel_x_g"],
                    data["accel_y_g"],
                    data["accel_z_g"],
                ], dtype=float)

                mag = np.array([
                    data.get("mag_x", math.nan),
                    data.get("mag_y", math.nan),
                    data.get("mag_z", math.nan),
                ], dtype=float)

                raw_logger.write_raw(
                    imu_timestamp_us=ts,
                    gyro_rad_s=gyro,
                    accel_g=accel,
                    mag_unit=mag,
                    temperature_c=None,
                )

                latest_raw = {
                    "timestamp": ts,
                    "gyro": gyro,
                    "accel": accel,
                    "mag": mag,
                }

            elif packet_type == "GRAVITY":
                gravity_count += 1

                gravity = np.array([
                    data["gravity_x_g"],
                    data["gravity_y_g"],
                    data["gravity_z_g"],
                ], dtype=float)

                latest_gravity = {
                    "timestamp": ts,
                    "gravity": gravity,
                }

            elif packet_type == "QUAT":
                quat_count += 1

                q1 = data["q1"]
                q2 = data["q2"]
                q3 = data["q3"]
                q4 = data["q4"]

                quat_logger.write_quat(
                    imu_timestamp_us=ts,
                    q1=q1,
                    q2=q2,
                    q3=q3,
                    q4=q4,
                )

                q_scipy = quat_to_scipy_xyzw(q1, q2, q3, q4)

                euler_zyx = R.from_quat(q_scipy).as_euler(
                    "zyx",
                    degrees=True,
                )

                latest_quat = {
                    "timestamp": ts,
                    "q1q2q3q4": np.array([q1, q2, q3, q4], dtype=float),
                    "xyzw": q_scipy,
                    "euler_zyx_deg": euler_zyx,
                }

            elif packet_type == "RPY":
                rpy_count += 1

                latest_rpy = {
                    "timestamp": ts,
                    "roll": data["roll_deg"],
                    "pitch": data["pitch_deg"],
                    "yaw": data["yaw_deg"],
                }

            now = time.time()

            if now - last_print_time >= args.print_every:
                last_print_time = now

                print("=" * 72)

                if latest_raw is not None:
                    accel = latest_raw["accel"]
                    gyro = latest_raw["gyro"]
                    mag = latest_raw["mag"]

                    accel_norm = np.linalg.norm(accel)
                    gyro_norm = np.linalg.norm(gyro)

                    print(f"RAW ts: {latest_raw['timestamp']}")
                    print(f"RAW gyro rad/s: {format_vec(gyro, 6)}")
                    print(f"RAW gyro norm rad/s: {gyro_norm:.6f}")
                    print(f"RAW accel g: {format_vec(accel, 6)}")
                    print(f"RAW accel m/s^2: {format_vec(accel * G_TO_MPS2, 6)}")
                    print(f"RAW accel norm g: {accel_norm:.6f}")
                    print(f"RAW mag: {format_vec(mag, 6)}")
                    print(f"RAW mag norm: {np.linalg.norm(mag):.6f}")

                if latest_gravity is not None:
                    gravity = latest_gravity["gravity"]
                    gravity_norm = np.linalg.norm(gravity)

                    print(f"GRAVITY ts: {latest_gravity['timestamp']}")
                    print(f"GRAVITY g: {format_vec(gravity, 6)}")
                    print(f"GRAVITY norm g: {gravity_norm:.6f}")

                if latest_raw is not None and latest_gravity is not None:
                    accel = latest_raw["accel"]
                    gravity = latest_gravity["gravity"]

                    linear_accel_body_g = accel - gravity
                    linear_norm = np.linalg.norm(linear_accel_body_g)

                    print("CHECK raw accel minus gravity:")
                    print(f"linear accel body g: {format_vec(linear_accel_body_g, 6)}")
                    print(f"linear accel body norm g: {linear_norm:.6f}")
                    print(f"linear accel body m/s^2: {format_vec(linear_accel_body_g * G_TO_MPS2, 6)}")

                    if linear_norm < 0.05:
                        print("RESULT: PASS. Stationary RAW and GRAVITY are aligned.")
                    elif linear_norm < 0.15:
                        print("RESULT: WARN. Small mismatch or vibration.")
                    else:
                        print("RESULT: CHECK. Large mismatch. Check motion or gravity sign.")

                if latest_quat is not None:
                    q1q2q3q4 = latest_quat["q1q2q3q4"]
                    xyzw = latest_quat["xyzw"]
                    euler = latest_quat["euler_zyx_deg"]

                    print(f"QUAT ts: {latest_quat['timestamp']}")
                    print(f"QUAT official q1 q2 q3 q4: {format_vec(q1q2q3q4, 6)}")
                    print(f"QUAT norm: {np.linalg.norm(q1q2q3q4):.6f}")
                    print(f"QUAT scipy x y z w: {format_vec(xyzw, 6)}")
                    print(
                        f"QUAT euler zyx deg: "
                        f"yaw={euler[0]:.3f}, "
                        f"pitch={euler[1]:.3f}, "
                        f"roll={euler[2]:.3f}"
                    )

                if latest_rpy is not None:
                    print(f"RPY ts: {latest_rpy['timestamp']}")
                    print(
                        f"RPY official deg: "
                        f"roll={latest_rpy['roll']:.3f}, "
                        f"pitch={latest_rpy['pitch']:.3f}, "
                        f"yaw={latest_rpy['yaw']:.3f}"
                    )

                if latest_quat is not None and latest_rpy is not None:
                    quat_euler = latest_quat["euler_zyx_deg"]
                    rpy_vec = np.array([
                        latest_rpy["yaw"],
                        latest_rpy["pitch"],
                        latest_rpy["roll"],
                    ])

                    diff = quat_euler - rpy_vec
                    diff = (diff + 180.0) % 360.0 - 180.0

                    print("CHECK QUAT Euler versus official RPY:")
                    print(f"diff yaw pitch roll deg: {format_vec(diff, 6)}")

                if latest_quat is not None and latest_raw is not None:
                    accel_body_g = latest_raw["accel"]
                    q = latest_quat["xyzw"]
                    rot = R.from_quat(q)

                    acc_apply = rot.apply(accel_body_g)
                    acc_inv = rot.inv().apply(accel_body_g)

                    print("CHECK QUAT gravity rotation:")
                    print(f"rot.apply(accel_body_g): {format_vec(acc_apply, 6)}")
                    print(f"rot.inv().apply(accel_body_g): {format_vec(acc_inv, 6)}")
                    print(f"apply xy norm: {np.linalg.norm(acc_apply[:2]):.6f}")
                    print(f"inv xy norm: {np.linalg.norm(acc_inv[:2]):.6f}")

                if now - last_rate_time >= 2.0:
                    elapsed = now - rate_start_time

                    print("PACKET RATE:")
                    print(f"RAW: {raw_count / elapsed:.1f} Hz")
                    print(f"GRAVITY: {gravity_count / elapsed:.1f} Hz")
                    print(f"QUAT: {quat_count / elapsed:.1f} Hz")
                    print(f"RPY: {rpy_count / elapsed:.1f} Hz")
                    print(f"Saved RAW rows: {raw_logger.count}")
                    print(f"Saved QUAT rows: {quat_logger.count}")
                    print(f"CRC fail count: {reader.crc_fail_count}")
                    print(f"Bad packet count: {reader.bad_packet_count}")

                    last_rate_time = now

    except KeyboardInterrupt:
        print("")
        print("Stopped by user.")

    finally:
        raw_logger.close()
        quat_logger.close()
        ser.close()
        print(f"Saved raw file: {args.output}")
        print(f"Saved quaternion file: {args.quat_output}")
        print("Serial closed.")


if __name__ == "__main__":
    main()