# -*- coding: utf-8 -*-
"""
Receive NMEA data from one XBee Socket Server and save one CSV file.

XBee settings:
    Socket server: ON
    TCP port: 5000
    Socket client: OFF

The CSV contains one row for each GGA position. A GST sentence with the
same GPS UTC time is attached to that position when available.
"""

import csv
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple


# =========================
# User configuration
# =========================

XBEE_IP = "192.168.0.166"
SOCKET_PORT = 5000
CSV_LOG_FILE = "gps_position_log.csv"

CONNECT_TIMEOUT_S = 5
RECV_TIMEOUT_S = 1
RECONNECT_DELAY_S = 3

# Wait briefly for a GST sentence with the same GPS UTC time.
# If no matching GST arrives, the GGA position is still saved and the
# error columns are left empty.
PAIR_WAIT_S = 2.0

# The host only needs to receive data from the XBee.
SEND_HELLO_ON_CONNECT = False
HELLO_MESSAGE = b"Hello from host\r\n"


# =========================
# Shared control
# =========================

stop_event = threading.Event()
file_lock = threading.Lock()


# =========================
# NMEA parsing helpers
# =========================


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def nmea_to_decimal(coord: str, direction: str) -> Optional[float]:
    """Convert NMEA latitude or longitude to decimal degrees."""
    if not coord or not direction:
        return None

    try:
        if direction in ("N", "S"):
            degree_length = 2
        elif direction in ("E", "W"):
            degree_length = 3
        else:
            return None

        degrees = float(coord[:degree_length])
        minutes = float(coord[degree_length:])
        decimal = degrees + minutes / 60.0

        if direction in ("S", "W"):
            decimal *= -1.0

        return decimal

    except ValueError:
        return None


def parse_gga(sentence: str) -> Optional[dict]:
    """Parse position, altitude, GPS UTC time, and fix quality from GGA."""
    parts = sentence.split(",")

    if len(parts) < 10:
        return None

    try:
        return {
            "gps_utc_time": parts[1],
            "latitude": nmea_to_decimal(parts[2], parts[3]),
            "longitude": nmea_to_decimal(parts[4], parts[5]),
            "altitude_m": float(parts[9]) if parts[9] else None,
            "fix_quality": int(parts[6]) if parts[6] else None,
            "raw_gga": sentence,
        }

    except (ValueError, IndexError):
        return None


def parse_gst(sentence: str) -> Optional[dict]:
    """Parse latitude, longitude, and altitude error estimates from GST."""
    parts = sentence.split(",")

    if len(parts) < 9:
        return None

    try:
        altitude_text = parts[8].split("*")[0] if parts[8] else ""

        return {
            "gps_utc_time": parts[1],
            "lat_error_m": float(parts[6]) if parts[6] else None,
            "lon_error_m": float(parts[7]) if parts[7] else None,
            "alt_error_m": float(altitude_text) if altitude_text else None,
            "raw_gst": sentence,
        }

    except (ValueError, IndexError):
        return None


def is_gga(sentence: str) -> bool:
    return sentence.startswith("$") and "GGA" in sentence[:10]


def is_gst(sentence: str) -> bool:
    return sentence.startswith("$") and "GST" in sentence[:10]


# =========================
# CSV pairing and writing
# =========================


class PositionLogger:
    """Pair GGA and GST by GPS UTC time and write one CSV row per GGA."""

    def __init__(self, writer: csv.DictWriter, file_object):
        self.writer = writer
        self.file_object = file_object
        self.pending_gga: Dict[str, Tuple[dict, float, str]] = {}
        self.pending_gst: Dict[str, Tuple[dict, float]] = {}

    def process_line(self, line: str) -> None:
        if is_gga(line):
            gga = parse_gga(line)
            if gga is None:
                print(f"[WARN] Failed to parse GGA: {line}")
                return

            gps_time = gga["gps_utc_time"]
            self.pending_gga[gps_time] = (gga, time.monotonic(), utc_now_iso())
            self._write_matching_pair(gps_time)

            print(
                f"[GGA] time={gps_time} fix={gga['fix_quality']} "
                f"lat={gga['latitude']} lon={gga['longitude']} "
                f"alt={gga['altitude_m']}"
            )

        elif is_gst(line):
            gst = parse_gst(line)
            if gst is None:
                print(f"[WARN] Failed to parse GST: {line}")
                return

            gps_time = gst["gps_utc_time"]
            self.pending_gst[gps_time] = (gst, time.monotonic())
            self._write_matching_pair(gps_time)

            print(
                f"[GST] time={gps_time} lat_err={gst['lat_error_m']} "
                f"lon_err={gst['lon_error_m']} alt_err={gst['alt_error_m']}"
            )

        self.flush_stale()

    def _write_matching_pair(self, gps_time: str) -> None:
        if gps_time not in self.pending_gga or gps_time not in self.pending_gst:
            return

        gga, _, pc_timestamp = self.pending_gga.pop(gps_time)
        gst, _ = self.pending_gst.pop(gps_time)
        self._write_row(gga, gst, pc_timestamp)

    def _write_row(
        self,
        gga: dict,
        gst: Optional[dict],
        pc_timestamp: str,
    ) -> None:
        raw_parts = [gga["raw_gga"]]
        if gst is not None:
            raw_parts.append(gst["raw_gst"])

        row = {
            "timestamp_utc": pc_timestamp,
            "gps_utc_time": gga["gps_utc_time"],
            "latitude": gga["latitude"],
            "longitude": gga["longitude"],
            "altitude_m": gga["altitude_m"],
            "lat_error_m": gst["lat_error_m"] if gst else None,
            "lon_error_m": gst["lon_error_m"] if gst else None,
            "alt_error_m": gst["alt_error_m"] if gst else None,
            "fix_quality": gga["fix_quality"],
            "raw_data": " || ".join(raw_parts),
        }

        with file_lock:
            self.writer.writerow(row)
            self.file_object.flush()

    def flush_stale(self, force: bool = False) -> None:
        """Save old unmatched GGA records with empty GST error fields."""
        now = time.monotonic()
        stale_times = []

        for gps_time, (_, received_time, _) in self.pending_gga.items():
            if force or now - received_time >= PAIR_WAIT_S:
                stale_times.append(gps_time)

        for gps_time in stale_times:
            gga, _, pc_timestamp = self.pending_gga.pop(gps_time)
            gst_tuple = self.pending_gst.pop(gps_time, None)
            gst = gst_tuple[0] if gst_tuple else None
            self._write_row(gga, gst, pc_timestamp)

        # Remove GST records that have no corresponding recent GGA.
        old_gst_times = []
        for gps_time, (_, received_time) in self.pending_gst.items():
            if force or now - received_time >= 10.0:
                old_gst_times.append(gps_time)

        for gps_time in old_gst_times:
            self.pending_gst.pop(gps_time, None)


# =========================
# TCP receiver
# =========================


def receive_from_xbee(logger: PositionLogger) -> None:
    buffer = ""

    while not stop_event.is_set():
        try:
            print(f"[INFO] Connecting to {XBEE_IP}:{SOCKET_PORT} ...")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(CONNECT_TIMEOUT_S)
                sock.connect((XBEE_IP, SOCKET_PORT))
                sock.settimeout(RECV_TIMEOUT_S)

                local_ip, local_port = sock.getsockname()
                remote_ip, remote_port = sock.getpeername()

                print(
                    f"[INFO] Connected: local={local_ip}:{local_port} "
                    f"remote={remote_ip}:{remote_port}"
                )

                if SEND_HELLO_ON_CONNECT:
                    sock.sendall(HELLO_MESSAGE)

                while not stop_event.is_set():
                    try:
                        data = sock.recv(4096)

                        if not data:
                            print("[WARN] XBee closed the connection.")
                            break

                        buffer += data.decode("utf-8", errors="ignore")

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if line:
                                logger.process_line(line)

                    except socket.timeout:
                        logger.flush_stale()
                        continue

        except (ConnectionRefusedError, TimeoutError, OSError) as error:
            if not stop_event.is_set():
                print(f"[WARN] Connection error: {error}")
                print(f"[INFO] Reconnecting in {RECONNECT_DELAY_S} seconds...")
                time.sleep(RECONNECT_DELAY_S)

        except Exception as error:
            if not stop_event.is_set():
                print(f"[ERROR] Unexpected error: {error}")
                time.sleep(RECONNECT_DELAY_S)


# =========================
# Main
# =========================


def main() -> None:
    fieldnames = [
        "timestamp_utc",
        "gps_utc_time",
        "latitude",
        "longitude",
        "altitude_m",
        "lat_error_m",
        "lon_error_m",
        "alt_error_m",
        "fix_quality",
        "raw_data",
    ]

    print("[INFO] Single XBee Socket Server receiver")
    print(f"[INFO] Target: {XBEE_IP}:{SOCKET_PORT}")
    print(f"[INFO] CSV file: {CSV_LOG_FILE}")
    print("[INFO] Press Ctrl+C to stop.")

    with open(CSV_LOG_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        logger = PositionLogger(writer, csv_file)

        try:
            receive_from_xbee(logger)

        except KeyboardInterrupt:
            print("\n[INFO] Stopping receiver...")
            stop_event.set()

        finally:
            logger.flush_stale(force=True)
            print(f"[INFO] GPS log saved to {CSV_LOG_FILE}")


if __name__ == "__main__":
    main()
