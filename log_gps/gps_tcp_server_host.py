# -*- coding: utf-8 -*-
"""
Receive NMEA data from one XBee Socket Server and save one CSV file.

Behavior:
    1. Every received NMEA sentence is preserved in raw_data.
    2. Valid GGA and matching GST sentences are combined into one row.
    3. Invalid GGA sentences are saved immediately.
    4. Unmatched GST and other NMEA sentences are also saved as raw rows.
"""

import csv
import os
import socket
import threading
import time
from datetime import datetime, timezone


# =========================
# User configuration
# =========================

XBEE_IP = "10.83.203.186"
SOCKET_PORT = 5000

CSV_FILE_NAME = "gps_position_log.csv"
CSV_LOG_FILE = os.path.join(os.getcwd(), CSV_FILE_NAME)

CONNECT_TIMEOUT_S = 5
RECV_TIMEOUT_S = 1
RECONNECT_DELAY_S = 3

PAIR_WAIT_S = 2.0
NO_DATA_WARNING_S = 5.0

SHOW_RECEIVE_PREVIEW = True
RECEIVE_PREVIEW_LENGTH = 180

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

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def nmea_to_decimal(coord, direction):
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


def parse_gga(sentence):
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


def parse_gst(sentence):
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


def is_gga(sentence):
    return sentence.startswith("$") and "GGA" in sentence[:10]


def is_gst(sentence):
    return sentence.startswith("$") and "GST" in sentence[:10]


# =========================
# CSV pairing and writing
# =========================

class PositionLogger:

    def __init__(self, writer, file_object):
        self.writer = writer
        self.file_object = file_object

        self.pending_gga = {}
        self.pending_gst = {}

        self.saved_count = 0
        self.nmea_line_count = 0

    def process_line(self, line):
        self.nmea_line_count += 1

        if line.startswith("$"):
            sentence_type = line[1:6]
        else:
            sentence_type = "UNKNOWN"

        print(
            f"[NMEA] line={self.nmea_line_count} "
            f"type={sentence_type}"
        )

        if is_gga(line):
            self.process_gga(line)

        elif is_gst(line):
            self.process_gst(line)

        else:
            self.write_raw_only(line)
            print(
                f"[RAW SAVED] Unsupported NMEA type: "
                f"{sentence_type}"
            )

        self.flush_stale()

    def process_gga(self, line):
        gga = parse_gga(line)

        if gga is None:
            self.write_raw_only(line)
            print(f"[RAW SAVED] Failed to parse GGA: {line}")
            return

        gps_time = gga["gps_utc_time"]

        print(
            f"[GGA] time={gps_time} "
            f"fix={gga['fix_quality']} "
            f"lat={gga['latitude']} "
            f"lon={gga['longitude']} "
            f"alt={gga['altitude_m']}"
        )

        valid_for_pairing = (
            bool(gps_time)
            and gga["latitude"] is not None
            and gga["longitude"] is not None
            and gga["fix_quality"] is not None
            and gga["fix_quality"] > 0
        )

        if not valid_for_pairing:
            self.write_gga_row(
                gga=gga,
                gst=None,
                pc_timestamp=utc_now_iso(),
            )

            print(
                "[RAW SAVED] Invalid GGA was saved "
                "without waiting for GST."
            )
            return

        if gps_time not in self.pending_gga:
            self.pending_gga[gps_time] = []

        self.pending_gga[gps_time].append(
            (
                gga,
                time.monotonic(),
                utc_now_iso(),
            )
        )

        self.write_matching_pairs(gps_time)

    def process_gst(self, line):
        gst = parse_gst(line)

        if gst is None:
            self.write_raw_only(line)
            print(f"[RAW SAVED] Failed to parse GST: {line}")
            return

        gps_time = gst["gps_utc_time"]

        print(
            f"[GST] time={gps_time} "
            f"lat_err={gst['lat_error_m']} "
            f"lon_err={gst['lon_error_m']} "
            f"alt_err={gst['alt_error_m']}"
        )

        if not gps_time:
            self.write_gst_only_row(
                gst=gst,
                pc_timestamp=utc_now_iso(),
            )

            print(
                "[RAW SAVED] GST without GPS time "
                "was saved immediately."
            )
            return

        if gps_time not in self.pending_gst:
            self.pending_gst[gps_time] = []

        self.pending_gst[gps_time].append(
            (
                gst,
                time.monotonic(),
            )
        )

        self.write_matching_pairs(gps_time)

    def write_matching_pairs(self, gps_time):
        gga_queue = self.pending_gga.get(gps_time, [])
        gst_queue = self.pending_gst.get(gps_time, [])

        while gga_queue and gst_queue:
            gga, _, pc_timestamp = gga_queue.pop(0)
            gst, _ = gst_queue.pop(0)

            self.write_gga_row(
                gga=gga,
                gst=gst,
                pc_timestamp=pc_timestamp,
            )

        if not gga_queue:
            self.pending_gga.pop(gps_time, None)

        if not gst_queue:
            self.pending_gst.pop(gps_time, None)

    def write_gga_row(self, gga, gst, pc_timestamp):
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

        self.write_row(row)

        if gst is None:
            gst_status = "missing"
        else:
            gst_status = "matched"

        print(
            f"[SAVED] row={self.saved_count} "
            f"time={row['gps_utc_time']} "
            f"fix={row['fix_quality']} "
            f"gst={gst_status}"
        )

    def write_gst_only_row(self, gst, pc_timestamp):
        row = {
            "timestamp_utc": pc_timestamp,
            "gps_utc_time": gst["gps_utc_time"],
            "latitude": None,
            "longitude": None,
            "altitude_m": None,
            "lat_error_m": gst["lat_error_m"],
            "lon_error_m": gst["lon_error_m"],
            "alt_error_m": gst["alt_error_m"],
            "fix_quality": None,
            "raw_data": gst["raw_gst"],
        }

        self.write_row(row)

        print(
            f"[SAVED GST RAW] row={self.saved_count} "
            f"time={row['gps_utc_time']}"
        )

    def write_raw_only(self, line):
        row = {
            "timestamp_utc": utc_now_iso(),
            "gps_utc_time": None,
            "latitude": None,
            "longitude": None,
            "altitude_m": None,
            "lat_error_m": None,
            "lon_error_m": None,
            "alt_error_m": None,
            "fix_quality": None,
            "raw_data": line,
        }

        self.write_row(row)

        print(
            f"[SAVED RAW] row={self.saved_count}"
        )

    def write_row(self, row):
        with file_lock:
            self.writer.writerow(row)
            self.file_object.flush()

        self.saved_count += 1

    def flush_stale(self, force=False):
        now = time.monotonic()

        stale_gga_items = []

        for gps_time, queue in list(self.pending_gga.items()):
            remaining = []

            for gga, received_time, pc_timestamp in queue:
                if force or now - received_time >= PAIR_WAIT_S:
                    stale_gga_items.append(
                        (
                            gps_time,
                            gga,
                            pc_timestamp,
                        )
                    )
                else:
                    remaining.append(
                        (
                            gga,
                            received_time,
                            pc_timestamp,
                        )
                    )

            if remaining:
                self.pending_gga[gps_time] = remaining
            else:
                self.pending_gga.pop(gps_time, None)

        for _, gga, pc_timestamp in stale_gga_items:
            self.write_gga_row(
                gga=gga,
                gst=None,
                pc_timestamp=pc_timestamp,
            )

        stale_gst_items = []

        for gps_time, queue in list(self.pending_gst.items()):
            remaining = []

            for gst, received_time in queue:
                if force or now - received_time >= 10.0:
                    stale_gst_items.append(gst)
                else:
                    remaining.append(
                        (
                            gst,
                            received_time,
                        )
                    )

            if remaining:
                self.pending_gst[gps_time] = remaining
            else:
                self.pending_gst.pop(gps_time, None)

        for gst in stale_gst_items:
            self.write_gst_only_row(
                gst=gst,
                pc_timestamp=utc_now_iso(),
            )


# =========================
# TCP receiver
# =========================

def make_preview(data):
    text = data.decode("utf-8", errors="ignore")
    text = text.replace("\r", "\\r")
    text = text.replace("\n", "\\n")

    if len(text) > RECEIVE_PREVIEW_LENGTH:
        text = text[:RECEIVE_PREVIEW_LENGTH] + "..."

    return text


def receive_from_xbee(logger):
    buffer = ""
    receive_block_count = 0
    total_received_bytes = 0

    while not stop_event.is_set():
        try:
            print(
                f"[INFO] Connecting to "
                f"{XBEE_IP}:{SOCKET_PORT} ..."
            )

            with socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM,
            ) as sock:
                sock.settimeout(CONNECT_TIMEOUT_S)
                sock.connect((XBEE_IP, SOCKET_PORT))
                sock.settimeout(RECV_TIMEOUT_S)

                local_ip, local_port = sock.getsockname()
                remote_ip, remote_port = sock.getpeername()

                print(
                    f"[CONNECTED] "
                    f"local={local_ip}:{local_port} "
                    f"remote={remote_ip}:{remote_port}"
                )

                print(
                    "[INFO] TCP connection is open. "
                    "Waiting for XBee data..."
                )

                if SEND_HELLO_ON_CONNECT:
                    sock.sendall(HELLO_MESSAGE)

                last_data_time = time.monotonic()
                last_warning_time = 0.0

                while not stop_event.is_set():
                    try:
                        data = sock.recv(4096)

                        if not data:
                            print(
                                "[WARN] XBee closed "
                                "the connection."
                            )
                            break

                        receive_block_count += 1
                        total_received_bytes += len(data)
                        last_data_time = time.monotonic()

                        print(
                            f"[RX] block={receive_block_count} "
                            f"bytes={len(data)} "
                            f"total_bytes={total_received_bytes}"
                        )

                        if SHOW_RECEIVE_PREVIEW:
                            print(
                                f"[RX DATA] "
                                f"{make_preview(data)}"
                            )

                        buffer += data.decode(
                            "utf-8",
                            errors="ignore",
                        )

                        while "\n" in buffer:
                            line, buffer = buffer.split(
                                "\n",
                                1,
                            )

                            line = line.strip()

                            if line:
                                logger.process_line(line)

                    except socket.timeout:
                        logger.flush_stale()

                        now = time.monotonic()
                        silent_seconds = now - last_data_time

                        if silent_seconds >= NO_DATA_WARNING_S:
                            if (
                                now - last_warning_time
                                >= NO_DATA_WARNING_S
                            ):
                                print(
                                    "[WAIT] Connected, but no "
                                    "XBee data received for "
                                    f"{silent_seconds:.1f} seconds."
                                )

                                last_warning_time = now

        except (
            ConnectionRefusedError,
            TimeoutError,
            OSError,
        ) as error:
            if not stop_event.is_set():
                print(
                    f"[WARN] Connection error: "
                    f"{error}"
                )

                print(
                    f"[INFO] Reconnecting in "
                    f"{RECONNECT_DELAY_S} seconds..."
                )

                time.sleep(RECONNECT_DELAY_S)

        except Exception as error:
            if not stop_event.is_set():
                print(
                    f"[ERROR] Unexpected error: "
                    f"{error}"
                )

                time.sleep(RECONNECT_DELAY_S)


# =========================
# Main
# =========================

def main():
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
    print(f"[INFO] Current directory: {os.getcwd()}")
    print(f"[INFO] CSV file: {os.path.abspath(CSV_LOG_FILE)}")
    print("[INFO] Press Ctrl+C to stop.")

    with open(
        CSV_LOG_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        csv_file.flush()

        logger = PositionLogger(
            writer,
            csv_file,
        )

        try:
            receive_from_xbee(logger)

        except KeyboardInterrupt:
            print("\n[INFO] Stopping receiver...")
            stop_event.set()

        finally:
            logger.flush_stale(force=True)

            print(
                f"[INFO] Total saved rows: "
                f"{logger.saved_count}"
            )

            print(
                f"[INFO] GPS log saved to: "
                f"{os.path.abspath(CSV_LOG_FILE)}"
            )


if __name__ == "__main__":
    main()
