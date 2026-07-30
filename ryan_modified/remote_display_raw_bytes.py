# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:50:44 2025

@author: nicho
"""

import socket
import cv2
import numpy as np
import pygame
import subprocess
import threading
import queue
import os
import time
from PIL import Image
import io
import json
import struct
import matplotlib.pyplot as plt
import piexif
import Jetson.GPIO as GPIO
from datetime import datetime, timezone
from collections import deque

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None
# =====================
# Configuration
# =====================
LISTEN_IP = "0.0.0.0"
PORT = 5000
STAT_PORT1 = 6000
STAT_PORT2 = 6001
STAT_PORT3 = 6002
PPS_PIN = 7
CONTROL_PORT = 7000
CONTROL_ACK_PORT = 7001
CONTROL_BROADCAST_IP = "192.168.1.255"
CONTROL_RETRY_SECONDS = 0.1
CONTROL_DEADLINE_SECONDS = 0.85
CONTROL_PPS_GUARD_SECONDS = 0.1

# UTC configuration:
#   "GPS"    -> outdoor PPS + GPS RMC/ZDA serial UTC
#   "NO_GPS" -> indoor PPS + chrony-synchronized AGX system UTC
UTC_MODE = "NO_GPS"

# Used only when UTC_MODE == "GPS". Set GPS_NMEA_PORT = "" to auto-detect.
GPS_NMEA_PORT = "/dev/ttyUSB0"
GPS_NMEA_BAUD = 115200

GPS_POSITION_MAX_AGE_NS = 2_000_000_000
UTC_MODE = UTC_MODE.strip().upper()
if UTC_MODE not in ("GPS", "NO_GPS"):
    raise ValueError(
        "UTC_MODE must be GPS (outdoor) or NO_GPS (indoor)"
        )
UTC_SOURCE = "GPS_NMEA" if UTC_MODE == "GPS" else "SYSTEM_NTP"

# RMC/ZDA is assumed to arrive after the PPS edge whose UTC second it labels.
# Set this to 1 only when receiver documentation says the sentence time is one
# second behind the most recent PPS edge.
NMEA_PPS_UTC_OFFSET_SECONDS = 0
NMEA_PPS_MAX_ASSOCIATION_DELAY_NS = 900_000_000
NMEA_PPS_PAIR_WAIT_SECONDS = 1.1
# Window size to display images on
WIDTH = 1920
HEIGHT = 1080

# Contains the raw bytes recieved from the jetsons
raw_q = {
    "left": queue.Queue(500),
    "mid": queue.Queue(500),
    "right": queue.Queue(500),
}

# Contains the decoded frames to display on screen
decoded_q = {
    "left": queue.Queue(100),
    "mid": queue.Queue(100),
    "right": queue.Queue(100),
}

# Contains the decoded frames to be saved to disk
save_q = {
    "left": queue.Queue(500),
    "mid": queue.Queue(500),
    "right": queue.Queue(500),
}


frame_lock = threading.Lock()
stats_lock = threading.Lock()

# Latest_left/mid/right are the updated images from each camera to display
latest_left = None
latest_mid = None
latest_right = None

# each camera stat m_capt = middle camera's image capture rate
m_capt = None
m_enc = None
m_stream = None
m_save = None
m_time = None
m_exif = None
m_send = None
l_capt = None
l_enc = None
l_stream = None
l_save = None
l_time = None
l_exif = None
l_send = None
r_capt = None
r_enc = None
r_stream = None
r_save = None
r_time = None
r_exif = None
r_send = None
l_rec = None
m_rec = None
r_rec = None
L_tot_fps = []
M_tot_fps = []
R_tot_fps = []
R_capt = []
M_capt = []
L_capt = []
R_send = []
M_send = []
L_send = []

CAMERA_CONFIGS = {
    "192.168.1.12": {"label": "left", "ssh_user": "ryan4"},
    "192.168.1.11": {"label": "mid",  "ssh_user": "ryan5"},
    "192.168.1.13": {"label": "right","ssh_user": "ryan6"},
}
EXPECTED_CAMERAS = {config["label"] for config in CAMERA_CONFIGS.values()}
pps_lock = threading.Lock()
pps_condition = threading.Condition(pps_lock)
agx_pps_state = {
    "edge_count": 0,
    "armed_session_id": None,
    "active_session_id": None,
    "sequence": None,
    "monotonic_ns": None,
    "system_time_ns": None,
    "interval_ns": None,
    }
agx_pps_table = {}
sync_status = {
    "state": "WAITING_FOR_PPS",
    "session_id": None,
    "attempt": 0,
    "acknowledged_cameras": set(),
    "missing_cameras": set(EXPECTED_CAMERAS),
    }
recording_control = {
    "state": "PREVIEW",
    "target_session_id": None,
    "target_pps_sequence": None,
    }
gps_lock = threading.Lock()
gps_state = {
    "status": "NOT_STARTED",
    "utc_source": UTC_SOURCE,
    "system_ntp_synchronized": False,
    "port": None,
    "connected": False,
    "receiving_nmea": False,
    "fix_valid": False,
    "last_sentence_monotonic_ns": None,
    "last_utc_ns": None,
    "last_date": None,
    "utc_pair_status": "WAITING_FOR_PPS_AND_NMEA",
    "utc_pair_count": 0,
    "last_paired_session_id": None,
    "last_paired_pps_sequence": None,
    "last_paired_pps_utc_ns": None,
    "latitude": None,
    "longitude": None,
    "altitude": None,
    "message": "NMEA listener has not started",
    }
gps_samples = deque(maxlen=600)


# ===============
# Camera Stats
# ===============
def stat_thread(STAT_PORT, label):
    """Receives and stores one Nano's periodic capture and queue statistics."""
    global m_capt, m_save, m_send, m_enc, m_time, m_exif, m_stream, l_capt, l_enc, l_send, l_stream, l_save, l_exif, l_time, r_capt, r_enc, r_send, r_stream, r_time, r_exif, r_save
    
    # creates a socket to receive information from the jetson
    stat_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stat_server.bind((LISTEN_IP, STAT_PORT))
    stat_server.listen(5)
    conn, addr = stat_server.accept()
    print('connected:', addr)
    buffer = ""
    
    # recieves the camera statistics from the jetson
    while not stop_event.is_set():
        data = conn.recv(4096)

        if not data:
            break
        buffer += data.decode()
        
        # separtates the fields of the message on the comma ","
        while "\n" in buffer:
            stats, buffer = buffer.split("\n", 1)

            parts = [p.strip() for p in stats.split(",")]
            
            # checks to ensure that the proper number of fields have been recieved before moving on
            if len(parts) != 8:
                print(f"{label}: malformed stats: {stats}")
                continue
            
            # locates each number by splitting the field on the equals sign "=" and taking the 1 index [1]
            #fps_label = parts[0]
            capture = str(parts[1].split("=")[1])
            encode = str(parts[2].split("=")[1])
            send = str(parts[3].split("=")[1])
            save_q = str(parts[4].split("=")[1])
            stream_q = str(parts[5].split("=")[1])
            #exif_q = str(parts[6].split("=")[1])
            #time_q = str(parts[7].split("=")[1])
            
            # puts the recieved camera stats into their corresponding global variables
            with stats_lock:
                if label =="mid":
                    m_capt = capture
                    m_enc = encode
                    m_stream = stream_q
                    m_save = save_q
                    #m_time = time_q
                    #m_exif = exif_q
                    m_send = send
                    M_send.append(m_send)
                    M_capt.append(m_capt)
                elif label =="left":
                    l_capt = capture
                    l_enc = encode
                    l_stream = stream_q
                    l_save = save_q
                    #l_time = time_q
                    #l_exif = exif_q
                    l_send = send
                    L_send.append(l_send)
                    L_capt.append(l_capt)
                else:
                    r_capt = capture
                    r_enc = encode
                    r_stream = stream_q
                    r_save = save_q
                    #r_time = time_q
                    #r_exif = exif_q
                    r_send = send
                    R_capt.append(r_capt)
                    R_send.append(r_send)
        #print(data.decode(), end="")
 
# ====================
# TCP server creation
# ====================
def create_tcp_server(port):
    """Creates a low-latency TCP server for one camera image stream."""
    # Creates a TCP server using the inputed port and using "0.0.0.0" for the IP
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.bind((LISTEN_IP, port))
    s.listen(1)
    s.settimeout(1.0)
    return s

# =========================
# accepting the connection
# =========================
def accept_client(server_sock, label):
    """Waits for one Nano connection while allowing clean program shutdown."""
    # attempts to establish the socket connection for the inputed server_sock
    while not stop_event.is_set():
        try:
            conn, addr = server_sock.accept()
            conn.settimeout(1.0)
            print(f"{label} connected: {addr}")
            return conn, addr
        except socket.timeout:
            continue

# ==============
# receive exact
# ==============
def recv_exact(sock, n, timeout = 1.0):
    """Receives exactly the requested byte count unless the peer disconnects."""
    # creates the buffer and socket timeout
    buf = b''
    sock.settimeout(timeout)
    # while the buffer is shorter than the expected package chunks that are received will be added to the buffer
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        except socket.timeout:
            continue
    return buf

# =====================
# UDP Socket
# =====================

stop_event = threading.Event()

# ======================
# inserting into Queues
# ======================
def put_latest(q, item, label):
    """Adds one item to a queue and removes its oldest item when full."""
    # attempts to insert a given item into a given Queue
    # if the Queue is full it will pull out the oldest frame and drop it
    # if the Queue is still full after dropping the oldest frame it will also drop the newest frame
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
            print(f"{label} queue full; dropped oldest frame")
            return True
        except queue.Full:
            print(f"{label} queue still full; dropped newest frame")
            return False


def _nmea_checksum_valid(sentence):
    """Returns True for a valid NMEA checksum or a sentence without one."""
    if not sentence.startswith("$"):
        return False
    if "*" not in sentence:
        return True
    body, supplied = sentence[1:].split("*", 1)
    checksum = 0
    for character in body:
        checksum ^= ord(character)
    try:
        return checksum == int(supplied[:2], 16)
    except ValueError:
        return False


def _nmea_coordinate(value, hemisphere, degree_digits):
    """Converts an NMEA degrees/minutes coordinate to signed decimal degrees."""
    if not value or not hemisphere:
        return None
    try:
        degrees = int(value[:degree_digits])
        minutes = float(value[degree_digits:])
    except (TypeError, ValueError):
        return None
    coordinate = degrees + minutes / 60.0
    if hemisphere.upper() in ("S", "W"):
        coordinate = -coordinate
    return coordinate


def _nmea_date_from_rmc(value):
    """Parses an RMC DDMMYY date into a (year, month, day) tuple."""
    if len(value) != 6 or not value.isdigit():
        return None
    day = int(value[0:2])
    month = int(value[2:4])
    short_year = int(value[4:6])
    year = 2000 + short_year if short_year < 80 else 1900 + short_year
    return year, month, day


def _nmea_utc_ns(time_value, date_value):
    """Combines an NMEA UTC time and date into Unix nanoseconds exactly."""
    if not time_value or date_value is None:
        return None
    try:
        time_text = str(time_value).strip()
        if len(time_text) < 6:
            return None
        hour = int(time_text[0:2])
        minute = int(time_text[2:4])
        seconds_text = time_text[4:]
        if "." in seconds_text:
            whole_seconds, fractional_seconds = seconds_text.split(".", 1)
        else:
            whole_seconds, fractional_seconds = seconds_text, ""
        second = int(whole_seconds)
        fractional_ns = int(
            (fractional_seconds[:9] + "000000000")[:9]
            )
        year, month, day = date_value
        timestamp = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            tzinfo=timezone.utc,
            )
    except (TypeError, ValueError, IndexError):
        return None
    return int(timestamp.timestamp()) * 1_000_000_000 + fractional_ns


def associate_nmea_utc_to_pps(
        utc_ns,
        receive_monotonic_ns,
        sentence_type,
        ):
    """
    Labels the most recent AGX PPS edge with the UTC second from RMC/ZDA.

    The GPS receiver is expected to emit RMC/ZDA after the PPS edge represented
    by that sentence. Network time and AGX wall-clock time are not used for the
    association, so serial latency does not become part of the image timestamp.
    """
    if (
        sentence_type not in ("RMC", "ZDA")
        or not isinstance(utc_ns, int)
        or not isinstance(receive_monotonic_ns, int)
    ):
        return False

    pps_utc_ns = (
        (utc_ns // 1_000_000_000 + NMEA_PPS_UTC_OFFSET_SECONDS)
        * 1_000_000_000
        )
    pair_result = "NO_RECENT_PPS"
    paired_key = None
    newly_paired = False

    with pps_condition:
        candidates = []
        for trigger_key, trigger_record in agx_pps_table.items():
            pps_monotonic_ns = trigger_record.get("agx_pps_monotonic_ns")
            if not isinstance(pps_monotonic_ns, int):
                continue
            association_delay_ns = receive_monotonic_ns - pps_monotonic_ns
            if 0 <= association_delay_ns <= NMEA_PPS_MAX_ASSOCIATION_DELAY_NS:
                candidates.append(
                    (pps_monotonic_ns, trigger_key, trigger_record)
                    )

        if candidates:
            _, paired_key, trigger_record = max(
                candidates,
                key=lambda item: item[0],
                )
            existing_utc_ns = trigger_record.get("pps_utc_ns")

            if isinstance(existing_utc_ns, int):
                if existing_utc_ns == pps_utc_ns:
                    pair_result = "PAIRED"
                else:
                    pair_result = "UTC_CONFLICT"
            else:
                session_id, pps_sequence = paired_key
                previous_pairs = [
                    record
                    for (record_session_id, record_sequence), record
                    in agx_pps_table.items()
                    if (
                        record_session_id == session_id
                        and isinstance(record_sequence, int)
                        and record_sequence < pps_sequence
                        and isinstance(record.get("pps_utc_ns"), int)
                    )
                    ]
                if previous_pairs:
                    previous_record = max(
                        previous_pairs,
                        key=lambda record: record["pps_sequence"],
                        )
                    sequence_difference = (
                        pps_sequence - previous_record["pps_sequence"]
                        )
                    expected_utc_ns = (
                        previous_record["pps_utc_ns"]
                        + sequence_difference * 1_000_000_000
                        )
                    if pps_utc_ns != expected_utc_ns:
                        pair_result = "UTC_SEQUENCE_MISMATCH"
                    else:
                        pair_result = "PAIRED"
                else:
                    pair_result = "PAIRED"

                if pair_result == "PAIRED":
                    trigger_record["pps_utc_ns"] = pps_utc_ns
                    trigger_record["utc_pair_valid"] = True
                    trigger_record["nmea_sentence_type"] = sentence_type
                    trigger_record["nmea_receive_monotonic_ns"] = (
                        receive_monotonic_ns
                        )
                    trigger_record["nmea_association_delay_ns"] = (
                        receive_monotonic_ns
                        - trigger_record["agx_pps_monotonic_ns"]
                        )
                    newly_paired = True
                    pps_condition.notify_all()

    with gps_lock:
        gps_state["utc_pair_status"] = pair_result
        if paired_key is not None:
            gps_state["last_paired_session_id"] = paired_key[0]
            gps_state["last_paired_pps_sequence"] = paired_key[1]
        if pair_result == "PAIRED":
            gps_state["last_paired_pps_utc_ns"] = pps_utc_ns
            if newly_paired:
                gps_state["utc_pair_count"] += 1

    if newly_paired:
        print(
            "[utc] Paired "
            f"session={paired_key[0]}, PPS={paired_key[1]}, "
            f"UTC_NS={pps_utc_ns}, source={sentence_type}"
            )
    elif pair_result not in ("PAIRED", "NO_RECENT_PPS"):
        print(f"[utc] PPS/NMEA pairing rejected: {pair_result}")

    return pair_result == "PAIRED"


def parse_nmea_sentence(
        sentence,
        receive_system_time_ns,
        receive_monotonic_ns=None,
        ):
    """
    Parses GGA, RMC, and ZDA without requiring a separate NMEA package.

    Position samples retain both GPS UTC and AGX receive time. Frame geotagging
    uses the receive-time axis so it still works when the AGX clock has not
    already been disciplined to GPS UTC.
    """
    sentence = sentence.strip()
    if receive_monotonic_ns is None:
        receive_monotonic_ns = time.monotonic_ns()
    if not _nmea_checksum_valid(sentence):
        return False

    body = sentence[1:].split("*", 1)[0]
    fields = body.split(",")
    if not fields or len(fields[0]) < 3:
        return False
    sentence_type = fields[0][-3:].upper()
    if sentence_type not in ("GGA", "RMC", "ZDA"):
        with gps_lock:
            gps_state["receiving_nmea"] = True
            gps_state["last_sentence_monotonic_ns"] = time.monotonic_ns()
            gps_state["status"] = "NMEA_NO_SUPPORTED_FIX"
            gps_state["message"] = f"Receiving ${fields[0]}"
        return True

    with gps_lock:
        date_value = gps_state["last_date"]
        latitude = gps_state["latitude"]
        longitude = gps_state["longitude"]
        altitude = gps_state["altitude"]
        fix_valid = gps_state["fix_valid"]

        if sentence_type == "RMC" and len(fields) > 9:
            rmc_date = _nmea_date_from_rmc(fields[9])
            if rmc_date is not None:
                date_value = rmc_date
            fix_valid = fields[2].upper() == "A"
            parsed_latitude = _nmea_coordinate(fields[3], fields[4], 2)
            parsed_longitude = _nmea_coordinate(fields[5], fields[6], 3)
            if parsed_latitude is not None and parsed_longitude is not None:
                latitude = parsed_latitude
                longitude = parsed_longitude

        elif sentence_type == "GGA" and len(fields) > 9:
            try:
                fix_valid = int(fields[6] or "0") > 0
            except ValueError:
                fix_valid = False
            parsed_latitude = _nmea_coordinate(fields[2], fields[3], 2)
            parsed_longitude = _nmea_coordinate(fields[4], fields[5], 3)
            if parsed_latitude is not None and parsed_longitude is not None:
                latitude = parsed_latitude
                longitude = parsed_longitude
            try:
                altitude = float(fields[9]) if fields[9] else None
            except ValueError:
                altitude = None

        elif sentence_type == "ZDA" and len(fields) > 4:
            try:
                date_value = (int(fields[4]), int(fields[3]), int(fields[2]))
            except ValueError:
                pass

        utc_ns = _nmea_utc_ns(fields[1] if len(fields) > 1 else "", date_value)
        gps_state.update({
            "status": "GPS_FIX" if fix_valid else "NMEA_NO_FIX",
            "receiving_nmea": True,
            "fix_valid": fix_valid,
            "last_sentence_monotonic_ns": time.monotonic_ns(),
            "last_utc_ns": utc_ns,
            "last_date": date_value,
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "message": f"Receiving {sentence_type}",
            })

        if fix_valid and latitude is not None and longitude is not None:
            gps_samples.append({
                "receive_system_time_ns": receive_system_time_ns,
                "gps_utc_ns": utc_ns,
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                })

    associate_nmea_utc_to_pps(
        utc_ns,
        receive_monotonic_ns,
        sentence_type,
        )
    return True


def _nmea_candidate_ports():
    """Returns the configured port or currently enumerated serial ports."""
    if GPS_NMEA_PORT:
        return [port.strip() for port in GPS_NMEA_PORT.split(",") if port.strip()]
    if list_ports is None:
        return []
    ports = [port.device for port in list_ports.comports()]
    return sorted(
        ports,
        key=lambda port: (
            not any(token in port.lower() for token in ("usb", "acm", "gps")),
            port,
            ),
        )


def system_clock_ntp_synchronized():
    """Returns Ubuntu's current NTP synchronization state."""
    try:
        result = subprocess.run(
            [
                "timedatectl",
                "show",
                "--property=NTPSynchronized",
                "--value",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            )
    except (OSError, subprocess.SubprocessError):
        return False
    return (
        result.returncode == 0
        and result.stdout.strip().lower() == "yes"
        )


def system_ntp_monitor(stop_event):
    """Monitors chrony/systemd UTC readiness for indoor SYSTEM_NTP mode."""
    previous_state = None
    while not stop_event.is_set():
        synchronized = system_clock_ntp_synchronized()
        with gps_lock:
            gps_state.update({
                "status": (
                    "SYSTEM_NTP_SYNCED"
                    if synchronized
                    else "SYSTEM_NTP_NOT_SYNCED"
                    ),
                "connected": synchronized,
                "receiving_nmea": False,
                "system_ntp_synchronized": synchronized,
                "utc_pair_status": (
                    "PAIRED"
                    if synchronized and gps_state["utc_pair_count"] > 0
                    else "READY_FOR_PPS"
                    if synchronized
                    else "WAITING_FOR_SYSTEM_NTP"
                    ),
                "message": (
                    "AGX system UTC is synchronized"
                    if synchronized
                    else "Waiting for chrony/systemd NTP synchronization"
                    ),
                })
        if synchronized != previous_state:
            print(
                "[utc] SYSTEM_NTP "
                + ("synchronized" if synchronized else "not synchronized")
                )
            previous_state = synchronized
        stop_event.wait(2.0)


def utc_source_ready():
    """Returns whether the selected UTC source can label a future PPS."""
    with gps_lock:
        if UTC_SOURCE == "SYSTEM_NTP":
            return gps_state["system_ntp_synchronized"] is True
        return (
            gps_state["utc_pair_count"] > 0
            and gps_state["utc_pair_status"] == "PAIRED"
            )


def nmea_listener(stop_event):
    """Finds an NMEA serial stream and maintains recent GPS position samples."""
    if serial is None:
        with gps_lock:
            gps_state.update({
                "status": "PYserial_MISSING",
                "message": "Install pyserial or configure GPS_NMEA_PORT",
                })
        while not stop_event.wait(2.0):
            pass
        return

    while not stop_event.is_set():
        candidates = _nmea_candidate_ports()
        if not candidates:
            with gps_lock:
                gps_state.update({
                    "status": "NO_SERIAL_PORT",
                    "connected": False,
                    "receiving_nmea": False,
                    "port": None,
                    "message": "No candidate NMEA serial port found",
                    })
            stop_event.wait(2.0)
            continue

        found_stream = False
        for port in candidates:
            if stop_event.is_set():
                return
            try:
                gps_serial = serial.Serial(
                    port,
                    GPS_NMEA_BAUD,
                    timeout=0.5,
                    )
            except Exception as ex:
                with gps_lock:
                    gps_state.update({
                        "status": "PORT_OPEN_FAILED",
                        "connected": False,
                        "port": port,
                        "message": str(ex),
                        })
                continue

            try:
                with gps_lock:
                    gps_state.update({
                        "status": "PROBING",
                        "connected": True,
                        "receiving_nmea": False,
                        "port": port,
                        "message": f"Checking {port}",
                        })

                probe_deadline = time.monotonic() + 2.0
                last_valid_monotonic = None
                while not stop_event.is_set():
                    raw_line = gps_serial.readline()
                    receive_system_time_ns = time.time_ns()
                    receive_monotonic_ns = time.monotonic_ns()
                    if not raw_line:
                        if last_valid_monotonic is None:
                            if time.monotonic() >= probe_deadline:
                                break
                        elif time.monotonic() - last_valid_monotonic > 5.0:
                            break
                        continue
                    try:
                        sentence = raw_line.decode("ascii", errors="ignore").strip()
                    except Exception:
                        continue
                    if parse_nmea_sentence(
                            sentence,
                            receive_system_time_ns,
                            receive_monotonic_ns,
                    ):
                        found_stream = True
                        last_valid_monotonic = time.monotonic()
                        with gps_lock:
                            gps_state["port"] = port
                            gps_state["connected"] = True
            except Exception as ex:
                with gps_lock:
                    gps_state.update({
                        "status": "SERIAL_ERROR",
                        "message": str(ex),
                        })
            finally:
                gps_serial.close()
                with gps_lock:
                    gps_state["connected"] = False

            if found_stream:
                break

        if not found_stream:
            with gps_lock:
                gps_state.update({
                    "status": "NMEA_NOT_FOUND",
                    "receiving_nmea": False,
                    "message": "No NMEA sentences detected",
                    })
        stop_event.wait(1.0)


def get_nearest_gps_sample(image_system_time_ns):
    """Returns a recent GPS position closest to the image's AGX time."""
    if not isinstance(image_system_time_ns, int):
        return None
    with gps_lock:
        if not gps_samples:
            return None
        sample = min(
            gps_samples,
            key=lambda item: abs(
                item["receive_system_time_ns"] - image_system_time_ns
                ),
            )
        age_ns = abs(sample["receive_system_time_ns"] - image_system_time_ns)
        if age_ns > GPS_POSITION_MAX_AGE_NS:
            return None
        return dict(sample)


def arm_recording_on_next_pps():
    """Arms AGX saving on the next PPS after the user presses T."""
    if not utc_source_ready():
        return False
    with pps_condition:
        session_id = agx_pps_state["active_session_id"]
        sequence = agx_pps_state["sequence"]
        if session_id is None or sequence is None:
            return False
        if recording_control["state"] == "RECORDING":
            return True
        recording_control.update({
            "state": "WAIT_NEXT_PPS",
            "target_session_id": session_id,
            "target_pps_sequence": sequence + 1,
            })
        return True


def should_save_frame(metadata):
    """Allows saving only after the user-selected PPS boundary."""
    with pps_lock:
        state = recording_control["state"]
        target_session_id = recording_control["target_session_id"]
        target_pps_sequence = recording_control["target_pps_sequence"]
    return (
        state == "RECORDING"
        and metadata.get("agx_trigger_matched") is True
        and metadata.get("utc_timestamp_valid") is True
        and isinstance(metadata.get("image_utc_ns"), int)
        and metadata.get("session_id") == target_session_id
        and isinstance(metadata.get("pps_sequence"), int)
        and isinstance(target_pps_sequence, int)
        and metadata["pps_sequence"] >= target_pps_sequence
        )


def runtime_status_snapshot():
    """Returns thread-safe status strings for the live AGX view."""
    with pps_lock:
        sync_snapshot = {
            **sync_status,
            "acknowledged_cameras": set(sync_status["acknowledged_cameras"]),
            "missing_cameras": set(sync_status["missing_cameras"]),
            }
        recording_snapshot = dict(recording_control)
        pps_sequence = agx_pps_state["sequence"]
    with gps_lock:
        gps_snapshot = dict(gps_state)
    return sync_snapshot, recording_snapshot, pps_sequence, gps_snapshot


def pps_listener(pps_pin, stop_event):
    """
    Records AGX PPS edges and maintains the shared session sequence table.

    The first PPS after a successful camera handshake becomes sequence zero.
    Each later PPS gets a UTC label from RMC/ZDA or the synchronized AGX system
    clock, depending on UTC_SOURCE.
    """
    global agx_pps_state

    while not stop_event.is_set():
        edge = GPIO.wait_for_edge(pps_pin, GPIO.RISING, timeout=1000)

        if edge is None:
            continue

        pps_monotonic_ns = time.monotonic_ns()
        system_time_sample_ns = time.time_ns()
        system_sample_monotonic_ns = time.monotonic_ns()
        pps_system_time_ns = (
            system_time_sample_ns
            - (system_sample_monotonic_ns - pps_monotonic_ns)
            )
        with gps_lock:
            system_ntp_synchronized = (
                gps_state["system_ntp_synchronized"] is True
                )
        system_utc_valid = (
            UTC_SOURCE == "SYSTEM_NTP"
            and system_ntp_synchronized
            )
        paired_system_utc_ns = (
            pps_system_time_ns if system_utc_valid else None
            )

        with pps_condition:
            armed_session_id = agx_pps_state["armed_session_id"]
            previous_pps_monotonic_ns = agx_pps_state["monotonic_ns"]

            if previous_pps_monotonic_ns is None:
                pps_interval_ns = None
                sequence_step = 1
            else:
                pps_interval_ns = pps_monotonic_ns - previous_pps_monotonic_ns
                sequence_step = max(
                    1,
                    int((pps_interval_ns + 500000000) // 1000000000),
                    )

            if armed_session_id is not None:
                agx_pps_state["active_session_id"] = armed_session_id
                agx_pps_state["armed_session_id"] = None
                agx_pps_state["sequence"] = 0
                sync_status["state"] = "SYNCED"
                sync_status["session_id"] = armed_session_id
            elif agx_pps_state["active_session_id"] is not None:
                agx_pps_state["sequence"] += sequence_step

            agx_pps_state["edge_count"] += 1
            agx_pps_state["monotonic_ns"] = pps_monotonic_ns
            agx_pps_state["system_time_ns"] = pps_system_time_ns
            agx_pps_state["interval_ns"] = pps_interval_ns
            session_id = agx_pps_state["active_session_id"]
            sequence = agx_pps_state["sequence"]

            if (
                recording_control["state"] == "WAIT_NEXT_PPS"
                and session_id == recording_control["target_session_id"]
                and isinstance(sequence, int)
                and sequence >= recording_control["target_pps_sequence"]
            ):
                recording_control["state"] = "RECORDING"
                print(
                    "[recording] Started on "
                    f"session={session_id}, PPS={sequence}"
                    )

            if session_id is not None:
                agx_pps_table[(session_id, sequence)] = {
                    "session_id": session_id,
                    "pps_sequence": sequence,
                    "agx_pps_monotonic_ns": pps_monotonic_ns,
                    "agx_pps_system_time_ns": pps_system_time_ns,
                    "agx_pps_interval_ns": pps_interval_ns,
                    "agx_missed_pps_count": sequence_step - 1,
                    "pps_utc_ns": paired_system_utc_ns,
                    "utc_pair_valid": system_utc_valid,
                    "utc_source": UTC_SOURCE,
                    }

            pps_condition.notify_all()

        if session_id is not None and system_utc_valid:
            with gps_lock:
                gps_state["utc_pair_status"] = "PAIRED"
                gps_state["utc_pair_count"] += 1
                gps_state["last_paired_session_id"] = session_id
                gps_state["last_paired_pps_sequence"] = sequence
                gps_state["last_paired_pps_utc_ns"] = paired_system_utc_ns

        if session_id is None:
            print("[pps] AGX pulse received while waiting for a session")
        else:
            print(f"[pps] AGX session={session_id}, sequence={sequence}")


def create_session_id(attempt):
    """Creates a unique collection session identifier for one handshake attempt."""
    utc_now = datetime.now(timezone.utc)
    return f"run_{utc_now.strftime('%Y%m%d_%H%M%S_%f')}_{attempt}"


def synchronize_pps_session(stop_event):
    """
    Arms all three Nano devices and the AGX for the same next physical PPS.

    The AGX sends START_ON_NEXT_PPS immediately after one PPS. All expected
    ARMED replies must arrive before the following PPS; otherwise a new session
    identifier is generated and the procedure retries after the next pulse.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(('', CONTROL_ACK_PORT))
    sock.settimeout(0.05)
    attempt = 0
    with pps_lock:
        observed_edge_count = agx_pps_state["edge_count"]
        sync_status.update({
            "state": "WAITING_FOR_PPS",
            "session_id": None,
            "attempt": 0,
            "acknowledged_cameras": set(),
            "missing_cameras": set(EXPECTED_CAMERAS),
            })

    try:
        while not stop_event.is_set():
            with pps_condition:
                pps_condition.wait_for(
                    lambda: (
                        agx_pps_state["edge_count"] > observed_edge_count
                        or stop_event.is_set()
                        ),
                    timeout=2.0,
                    )
                if stop_event.is_set():
                    return None

                command_edge_count = agx_pps_state["edge_count"]
                if command_edge_count <= observed_edge_count:
                    continue
                observed_edge_count = command_edge_count

            # Allow all Nano GPIO callbacks to finish the same PPS before arming.
            if stop_event.wait(CONTROL_PPS_GUARD_SECONDS):
                return None
            with pps_lock:
                if agx_pps_state["edge_count"] != command_edge_count:
                    continue

            attempt += 1
            session_id = create_session_id(attempt)
            with pps_lock:
                sync_status.update({
                    "state": "HANDSHAKING",
                    "session_id": session_id,
                    "attempt": attempt,
                    "acknowledged_cameras": set(),
                    "missing_cameras": set(EXPECTED_CAMERAS),
                    })
            command = {
                "command": "START_ON_NEXT_PPS",
                "session_id": session_id,
                }
            command_bytes = json.dumps(command).encode('utf-8')
            acknowledged_cameras = set()
            deadline = time.monotonic() + CONTROL_DEADLINE_SECONDS
            next_send_time = 0.0

            while time.monotonic() < deadline and not stop_event.is_set():
                with pps_lock:
                    if agx_pps_state["edge_count"] != command_edge_count:
                        break

                if time.monotonic() >= next_send_time:
                    sock.sendto(
                        command_bytes,
                        (CONTROL_BROADCAST_IP, CONTROL_PORT),
                        )
                    next_send_time = time.monotonic() + CONTROL_RETRY_SECONDS

                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue

                try:
                    acknowledgement = json.loads(data.decode('utf-8'))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue

                if (
                    acknowledgement.get("message") == "ARMED"
                    and acknowledgement.get("session_id") == session_id
                    and acknowledgement.get("camera_id") in EXPECTED_CAMERAS
                ):
                    acknowledged_cameras.add(acknowledgement["camera_id"])
                    with pps_lock:
                        sync_status["acknowledged_cameras"] = set(
                            acknowledged_cameras
                            )
                        sync_status["missing_cameras"] = (
                            EXPECTED_CAMERAS - acknowledged_cameras
                            )

                if acknowledged_cameras == EXPECTED_CAMERAS:
                    with pps_condition:
                        if agx_pps_state["edge_count"] != command_edge_count:
                            break
                        agx_pps_state["armed_session_id"] = session_id
                        sync_status["state"] = "ARMED_WAIT_NEXT_PPS"
                        sync_status["acknowledged_cameras"] = set(
                            acknowledged_cameras
                            )
                        sync_status["missing_cameras"] = set()
                    print(f"[control] All cameras ARMED for session {session_id}")
                    return session_id

            missing = sorted(EXPECTED_CAMERAS - acknowledged_cameras)
            with pps_lock:
                sync_status["state"] = "RETRYING"
                sync_status["acknowledged_cameras"] = set(
                    acknowledged_cameras
                    )
                sync_status["missing_cameras"] = set(missing)
            print(f"[control] Session attempt failed; missing={missing}. Retrying.")
    finally:
        sock.close()

    return None


def match_frame_to_trigger(metadata, label):
    """
    Matches Nano frame metadata to the AGX record for the same PPS trigger.

    Saving is allowed only when RMC/ZDA has assigned a valid UTC second to the
    matching PPS. The image UTC is calculated with integer nanosecond math.
    """
    metadata = dict(metadata)
    session_id = metadata.get("session_id")
    pps_sequence = metadata.get("pps_sequence")
    delta_from_pps_ns = metadata.get("delta_from_pps_ns")
    camera_id_matches = metadata.get("camera_id") == label

    trigger_key = (session_id, pps_sequence)
    with pps_condition:
        pps_condition.wait_for(
            lambda: (
                (
                    trigger_key in agx_pps_table
                    and agx_pps_table[trigger_key].get("utc_pair_valid") is True
                    and isinstance(
                        agx_pps_table[trigger_key].get("pps_utc_ns"),
                        int,
                        )
                )
                or stop_event.is_set()
                ),
            timeout=NMEA_PPS_PAIR_WAIT_SECONDS,
            )
        trigger_record = agx_pps_table.get(trigger_key)
        if trigger_record is not None:
            trigger_record = dict(trigger_record)

    trigger_matched = (
        metadata.get("timestamp_valid") is True
        and camera_id_matches
        and trigger_record is not None
        and isinstance(delta_from_pps_ns, int)
        and trigger_record.get("utc_pair_valid") is True
        and isinstance(trigger_record.get("pps_utc_ns"), int)
        )
    metadata["agx_trigger_matched"] = trigger_matched
    metadata["utc_timestamp_valid"] = trigger_matched

    if trigger_matched:
        metadata["pps_utc_ns"] = trigger_record["pps_utc_ns"]
        metadata["agx_pps_interval_ns"] = trigger_record["agx_pps_interval_ns"]
        metadata["agx_missed_pps_count"] = trigger_record["agx_missed_pps_count"]
        metadata["image_utc_ns"] = (
            trigger_record["pps_utc_ns"]
            + delta_from_pps_ns
            )
    else:
        metadata["pps_utc_ns"] = None
        metadata["image_utc_ns"] = None

    return metadata


def _decimal_degrees_to_exif(value):
    """Converts signed decimal degrees into EXIF degree/minute/second rationals."""
    absolute = abs(float(value))
    degrees = int(absolute)
    minutes_float = (absolute - degrees) * 60.0
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60.0
    return (
        (degrees, 1),
        (minutes, 1),
        (int(round(seconds * 1_000_000)), 1_000_000),
        )


def merge_frame_metadata_exif(exif_bytes, metadata):
    """
    Writes only the PPS-derived UTC image time into EXIF.

    DateTimeOriginal carries the UTC whole second, SubSecTimeOriginal carries
    nine fractional digits, and UserComment retains the exact integer utc_ns.
    GPS position tags are deliberately removed.
    """
    try:
        exif_dict = piexif.load(exif_bytes)
    except Exception:
        exif_dict = {
            "0th": {},
            "Exif": {},
            "GPS": {},
            "1st": {},
            "thumbnail": None,
            }

    image_utc_ns = metadata.get("image_utc_ns")
    if not isinstance(image_utc_ns, int):
        return piexif.dump(exif_dict)

    utc_seconds, utc_subsecond_ns = divmod(image_utc_ns, 1_000_000_000)
    image_utc_datetime = datetime.fromtimestamp(
        utc_seconds,
        tz=timezone.utc,
        )
    exif_datetime = image_utc_datetime.strftime("%Y:%m:%d %H:%M:%S").encode(
        "ascii"
        )
    subsecond = f"{utc_subsecond_ns:09d}".encode("ascii")

    exif_dict.setdefault("0th", {})
    exif_dict.setdefault("Exif", {})
    exif_dict["0th"][piexif.ImageIFD.DateTime] = exif_datetime
    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_datetime
    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = exif_datetime
    exif_dict["Exif"][piexif.ExifIFD.SubSecTimeOriginal] = subsecond
    offset_tag = getattr(piexif.ExifIFD, "OffsetTimeOriginal", None)
    if offset_tag is not None:
        exif_dict["Exif"][offset_tag] = b"+00:00"

    exif_dict["GPS"] = {}
    utc_comment = json.dumps(
        {
            "time_standard": "UTC",
            "utc_ns": image_utc_ns,
        },
        separators=(',', ':'),
        ).encode('utf-8')
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = (
        b"ASCII\x00\x00\x00" + utc_comment
        )
    return piexif.dump(exif_dict)

# ==============
# initilization
# ==============
def init():
    """Starts the camera sender script on each configured Nano over SSH."""
    # goes through all camera configurations and will SSH each camera's computer
    # the path in remote_cmd can be modified to target different variations of the blackfly script
    # blkfly_md can run at 200 fps
    # blkfly_md435 can run at 435 fps
    # this requires that an SSH key exists on the target computer that is generated by the host machine
    # without this a password is required which cannot be done with this code
    for host, config in CAMERA_CONFIGS.items():
        remote_cmd = (
            f"cd /home/{config['ssh_user']}/Alex/Blackfly && "
            f"nohup python3 blkfly_rawbytes.py > /tmp/{config['label']}_blkfly_rawbytes.log 2>&1 &"
        )
        try:
            subprocess.Popen(["ssh", f"{config['ssh_user']}@{host}", remote_cmd])
            print(f"Started remote sender on {host}")
        except OSError as e:
            print(f"Failed to start remote sender on {host}: {e}")

# ==============
# receiver
# ==============
def receiver(server_sock, label):
    """
    Receives raw frames, EXIF data, and PPS timing metadata from one Nano.

    The TCP protocol contains three length-prefixed sections in this order:
    JSON metadata, raw Mono8 image bytes, and EXIF bytes.
    """
    global l_rec, m_rec, r_rec, L_tot_fps, M_tot_fps, R_tot_fps, R_send, R_capt, M_capt, M_send, L_capt, L_send
    frame_count = 0
    last_time = time.time()
    
    # when the code runs it will attempt to connect to the camera computers
    while not stop_event.is_set():
        print(f"{label}: waiting for connection...")
        conn, addr = accept_client(server_sock, label)
        
        # after connecting it will begin receiving images from the cameras
        # first a 4 byte header will come to give the length of the incoming frame
        # once received the header will be decoded into the frame length and used to inform recv_exact of how long the frame is
        # once the frame is fully received the raw bytes will be put into a queue to be processed elsewhere
        # the frame count is also kept along with the elapsed time to help calculate the frame rate received
        # and the bytes will be put into their respective latest_left/mid/right global variables for displaying
        try:
            while not stop_event.is_set():
                header = recv_exact(conn, 4)
                if header is None:
                    print(f"{label}: disconnected while reading metadata_len")
                    break
                metadata_len = struct.unpack("!I", header)[0]

                header = recv_exact(conn, 4)
                if header is None:
                    print(f"{label}: disconnected while reading img_len")
                    break
                img_len = struct.unpack("!I", header)[0]

                header = recv_exact(conn, 4)
                if header is None:
                    print(f"{label}: disconnected while reading exif_len")
                    break
                exif_len = struct.unpack("!I", header)[0]

                metadata_bytes = recv_exact(conn, metadata_len)
                img_bytes = recv_exact(conn, img_len)
                exif_bytes = recv_exact(conn, exif_len)
                agx_receive_system_time_ns = time.time_ns()
                agx_receive_monotonic_ns = time.monotonic_ns()
                if None in (metadata_bytes, img_bytes, exif_bytes):
                    break

                try:
                    metadata = json.loads(metadata_bytes.decode('utf-8'))
                except (UnicodeDecodeError, json.JSONDecodeError) as ex:
                    print(f"{label}: invalid frame metadata: {ex}")
                    continue
                metadata["agx_receive_system_time_ns"] = (
                    agx_receive_system_time_ns
                    )
                metadata["agx_receive_monotonic_ns"] = (
                    agx_receive_monotonic_ns
                    )

                payload = [img_bytes, exif_bytes, metadata]

                # ONLY STORE RAW PACKET
                q = raw_q[label]
                if q.full():
                    try:
                        q.get_nowait()
                    except:
                        pass
                
                q.put_nowait(payload)
                frame_count += 1
                now = time.time()
                elapsed = now-last_time
                if elapsed >= 1:
                    with stats_lock:
                        if label == 'left':
                            l_rec = frame_count/elapsed
                            L_tot_fps.append(l_rec)
                        elif label == 'mid':
                            m_rec = frame_count/elapsed
                            M_tot_fps.append(m_rec)
                        else:
                            r_rec = frame_count/elapsed
                            R_tot_fps.append(r_rec)
                    frame_count = 0
                    last_time = now
        except Exception as e:
            print(f"{label} receiver error: {e}")
        finally:
            conn.close()


# ===============
# decoder
# ===============
def decode_raw_frame(img_bytes, metadata):
    """
    Validates and decodes one metadata-described raw camera frame.

    The current camera pipeline supports Mono8. Width and height come from the
    transmitted frame metadata rather than being assumed by the AGX.
    """
    if not isinstance(metadata, dict):
        raise ValueError("frame metadata must be a JSON object")

    width = metadata.get("width")
    height = metadata.get("height")
    pixel_format = metadata.get("pixel_format")

    if type(width) is not int or type(height) is not int:
        raise ValueError("frame width and height must be integers")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame dimensions: {width}x{height}")
    if pixel_format != "Mono8":
        raise ValueError(f"unsupported pixel format: {pixel_format!r}")

    expected_size = width * height
    actual_size = len(img_bytes)
    if actual_size != expected_size:
        raise ValueError(
            f"Mono8 payload size mismatch: expected {expected_size} bytes "
            f"for {width}x{height}, received {actual_size}"
            )

    return np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width))


def decoder(label):
    """
    Converts raw Mono8 bytes into an image and matches its PPS on the AGX.

    Frame dimensions and pixel format are validated from the transmitted
    metadata before the image is published to the preview or save pipeline.
    The trigger match result is embedded back into EXIF before the frame is
    forwarded to the saver.
    """
    global latest_left, latest_mid, latest_right
    while not stop_event.is_set():
        try:
            payload = raw_q[label].get(timeout=0.1)
            img_bytes, exif_bytes, metadata = payload
        except queue.Empty:
            continue
        try:
            frame = decode_raw_frame(img_bytes, metadata)
        except (TypeError, ValueError) as ex:
            print(f"{label}: invalid raw frame: {ex}")
            continue

        # Publish only frames that passed the metadata and payload validation.
        with frame_lock:
            if label == "left":
                latest_left = frame
            elif label == "mid":
                latest_mid = frame
            else:
                latest_right = frame

        # During stage one the receiver continues updating the live preview,
        # while this worker drains and discards its save-path copy immediately.
        with pps_lock:
            current_recording_state = recording_control["state"]
        if current_recording_state == "PREVIEW":
            continue

        # Once T has armed recording, retain the existing matching and EXIF
        # pipeline but gate the final save on the selected PPS boundary.
        metadata = match_frame_to_trigger(metadata, label)
        exif_bytes = merge_frame_metadata_exif(exif_bytes, metadata)
        if should_save_frame(metadata) and not save_q[label].full():
            save_q[label].put([frame, exif_bytes, metadata])

# ==================
# dummy gps sender 
# ==================
def gps_dummy_sender():
    """Broadcasts changing test coordinates for development without a GPS."""
    # optional function to send dummy gps coordinates to the cameras to test live embedding of the coordinates into the images
    # requires the gps receiver to be turned on in more recent versions of the blkfly script (double check this before using)
    # Configuration matches blkfly2.py gps_listener
    TARGET_PORT = 5005
    BROADCAST_IP = '192.168.1.255' # Broadcast to the camera subnet
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    print(f"Starting GPS dummy broadcast on port {TARGET_PORT}")
    
    # Dummy starting values
    lat = 39.7589  
    lon = -84.1916
    alt = 225.0
    
    try:
        while not stop_event.is_set():
            gps_payload = {
                "left":  {"lat": lat, "lon": lon, "alt": alt},
                "mid":   {"lat": lat, "lon": lon, "alt": alt},
                "right": {"lat": lat, "lon": lon, "alt": alt}
            }
            
            try:
                message = json.dumps(gps_payload).encode('utf-8')
                sock.sendto(message, (BROADCAST_IP, TARGET_PORT))
                lat += 0.00001 # just continuously adds to the lat and lon coordinate to ensure predictable variability
                lon += 0.00001
            except Exception as e:
                print(f"GPS Sender Error: {e}")
                
            time.sleep(1)
    finally:
        sock.close()
        print("GPS sender exiting cleanly")
    
# =================
# saver
# =================
def saver():
    """
    Saves the three decoded camera streams with their matched PPS metadata.

    The metadata has already been embedded in EXIF by decoder, so the saved
    JPEG retains the session, trigger sequence, relative time, and match result.
    """
    # the counters can keep track of how many frames have been saved, this can be done either independently
    # (when the right camera saves only the right counter goes up) or this can be done dependently 
    # (when the right camera save all counters go up)
    # each camera saves the images to a different folder
    counters = {"left": 1, "mid": 1, "right": 1}

    paths = {
        "left": "road_test_2/left_cam",
        "mid": "road_test_2/mid_cam",
        "right": "road_test_2/right_cam",
    }
    # if the directories don't exist on your machine os.makedirs() will create the required directories
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    # once the directories exist this function will pull the decoded frames from each save_q
    # then pull the count from the corresponding label's counter and create a file path using the count as the frame ID
    # the function then writes the image into the file path and updates the counters (currently working dependent on one another)
    while not stop_event.is_set():
        made_progress = False

        for label in ["left", "mid", "right"]:
            try:
                payload = save_q[label].get(timeout=0.05)
            except queue.Empty:
                continue
            frame, exif_bytes, metadata = payload
            made_progress = True
            i = counters[label]
            file_path = os.path.join(paths[label], f"frame_{i}.jpg")

            try:
                img = Image.fromarray(frame)

                img.save(
                    file_path,
                    format="JPEG",
                    exif=exif_bytes
                )
                counters['left'] += 1
                counters['mid'] += 1
                counters['right'] += 1
            except Exception as e:
                print(f"saver error {label}: {e}")

        if not made_progress:
            time.sleep(0.02)

        
# =======================
# de-initilization
# =======================
def de_init():
    """Stops the remote camera sender processes over SSH during shutdown."""
    # will SSH into each camera's computer and pass the "pgrep -f" command
    # this identifies all PIDs where the desired script is running
    # then using the PIDs it will SSH into the computer again and pass the "kill -2" command twice to those PIDs
    # this acts like sending a CTRL+C twice
    # the first "kill -2" causes it to exit the aquisition loops and the second tells it to exit the main function and terminate
    
    for host, config in CAMERA_CONFIGS.items():
        try:
            result = subprocess.run(
                ["ssh", f"{config['ssh_user']}@{host}", "pgrep -f blkfly_rawbytes.py"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as e:
            print(f"Failed to query remote PIDs on {host}: {e}")
            continue

        pids = result.stdout.strip().split()
        if not pids:
            continue

        for _ in range(2):
            try:
                subprocess.run(
                    ["ssh", f"{config['ssh_user']}@{host}", "kill -2 " + " ".join(pids)],
                    timeout=10,
                )
            except Exception as e:
                print(f"Failed to stop remote script on {host}: {e}")
            time.sleep(1)
        print(f"Killed remote script PIDs on {host}: {', '.join(pids)}")
    
    
def main():
    """
    Starts AGX PPS synchronization, three camera receivers, saving, and display.

    The AGX listens to the same PPS as the Nano devices and runs the UDP
    START_ON_NEXT_PPS handshake in parallel with the existing camera services.
    """
    global l_sock, m_sock, r_sock, STAT_PORT1, STAT_PORT2, STAT_PORT3, L_tot_fps, M_tot_fps, R_tot_fps, L_capt, L_send, M_capt, M_send, R_capt, R_send

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PPS_PIN, GPIO.IN)
    
    # creates the tcp servers for each camera using different ports
    l_sock = create_tcp_server(5001)
    m_sock = create_tcp_server(5002)
    r_sock = create_tcp_server(5000)
    
    # creates a receiving thread for each camera using the created TCP servers
    l_rec_thread = threading.Thread(target=receiver, args=(l_sock, "left"), daemon=True)
    m_rec_thread = threading.Thread(target=receiver, args=(m_sock, "mid"), daemon=True)
    r_rec_thread = threading.Thread(target=receiver, args=(r_sock, "right"), daemon=True)
    
    # creates a decode thread for each camera
    l_dec_thread = threading.Thread(target=decoder, args=("left",), daemon=True)
    m_dec_thread = threading.Thread(target=decoder, args=("mid",), daemon=True)
    r_dec_thread = threading.Thread(target=decoder, args=("right",), daemon=True)
    
    # creates a save thread to handle all of three cameras at once
    save_thread = threading.Thread(target=saver, daemon = True)
    
    if UTC_SOURCE == "SYSTEM_NTP":
        gps_thread = threading.Thread(
            target=system_ntp_monitor,
            args=(stop_event,),
            name="System-NTP-Monitor",
            daemon=True,
            )
    else:
        gps_thread = threading.Thread(
            target=nmea_listener,
            args=(stop_event,),
            name="GPS-NMEA-Listener",
            daemon=True,
            )

    # creates the PPS listener and shared-session synchronization threads
    pps_thread = threading.Thread(
        target=pps_listener,
        args=(PPS_PIN, stop_event),
        name="AGX-PPS-Listener",
        daemon=True,
        )
    sync_thread = threading.Thread(
        target=synchronize_pps_session,
        args=(stop_event,),
        name="PPS-Session-Sync",
        daemon=True,
        )
    
    #creates a stats thread for each camera
    m_stats_thread = threading.Thread(target=stat_thread, args=(STAT_PORT2, "mid"), daemon=True)
    l_stats_thread = threading.Thread(target=stat_thread, args=(STAT_PORT1, "left"), daemon=True)
    r_stats_thread = threading.Thread(target=stat_thread, args=(STAT_PORT3, "right"), daemon=True)
    
    # starts the receiving threads
    l_rec_thread.start()
    m_rec_thread.start()
    r_rec_thread.start()
    
    # starts the decoding threads
    l_dec_thread.start()
    m_dec_thread.start()
    r_dec_thread.start()
    
    # starts the stats threads
    m_stats_thread.start()
    l_stats_thread.start()
    r_stats_thread.start()

    # starts listening before the remote cameras are launched
    pps_thread.start()

    # runs the initilization function to start the cameras
    init()
    
    # starts the save thread 
    save_thread.start()
    
    # starts the selected UTC source before the PPS session handshake
    gps_thread.start()

    # arms all four devices so the same next PPS becomes sequence zero
    sync_thread.start()
    global latest_left, latest_mid, latest_right, l_capt, l_send, l_enc, l_stream, l_save, l_exif, l_time, m_capt, m_send, m_enc, m_stream, m_save, m_exif, m_time, r_capt, r_enc, r_send, r_stream, r_save, r_exif, r_time, l_rec, m_rec, r_rec
    
    # initilizes pygame and sets the screen size
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Live Stream")
        
    clock = pygame.time.Clock()
    running = True
    
    # creates a window to display the live camera views and statistics while the script runs
    try:
        while running and not stop_event.is_set():
            # if the pygame window is shut down the script will terminate
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    stop_event.set()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                    if arm_recording_on_next_pps():
                        print(
                            "[recording] T pressed; discarding frames until "
                            "the next PPS"
                            )
                    else:
                        print(
                            "[recording] T ignored; PPS handshake or UTC "
                            "source is not ready"
                            )
            
            # using frame lock the left/mid/right bytes are taken directly from the latest_left/mid/right
            # left/mid/right_img is set to None so that if a camera stops sending the image will go black instead of freezing
            with frame_lock:
                left_img = None
                mid_img = None
                right_img = None
                left_bytes = latest_left
                mid_bytes = latest_mid
                right_bytes = latest_right
                
            # if left/mid/right_bytes are all None then the loop runs again after a short waiting time
            '''if left_bytes is None and mid_bytes is None and right_bytes is None:
                print('all cameras failing to send')
                clock.tick(2)
                continue
            
            # if any of left/mid/right_bytes is not None then the bytes are decoded into a grayscale .jpg similar to the save thread
            # optional print available to tell which camera is failing
            if left_bytes is not None:
                npdata = np.frombuffer(left_bytes, dtype=np.uint8)
                left_img = cv2.imdecode(npdata, cv2.IMREAD_GRAYSCALE)
            else:
                #print('left camera failing')
                pass
            if mid_bytes is not None:
                npdata = np.frombuffer(mid_bytes, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", npdata, [cv2.IMWRITE_JPEG_QUALITY, 85])
                mid_img = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            else:
                print('mid camera failing')
                pass
            if right_bytes is not None:
                npdata = np.frombuffer(right_bytes, dtype=np.uint8)
                right_img = cv2.imdecode(npdata, cv2.IMREAD_GRAYSCALE)
            else:
                #print('right camera failing')
                pass'''
            
            left_img = left_bytes
            mid_img = mid_bytes
            right_img = right_bytes
            
            # initilizes the text to be displayed on screen"
            left_text = 'left camera'
            l_capture = f"capture: {l_capt} FPS"
            l_encode = f"encode: {l_enc} FPS"
            l_sent = f"send: {l_send} FPS"
            l_receive = f"receiced: {l_rec} FPS"
            l_streamq = f"stream_q: {l_stream} frames"
            l_saveq = f"save_q: {l_save} frames"
            #l_timeq = f"time_q: {l_time} timestamps"
            #l_exifq = f"exif_q: {l_exif} arrays"
            mid_text = 'mid camera'
            m_capture = f"capture: {m_capt} FPS"
            m_encode = f"encode: {m_enc} FPS"
            m_sent = f"send: {m_send} FPS"
            m_receive = f"received: {m_rec} FPS"
            m_streamq = f"stream_q: {m_stream} frames"
            m_saveq = f"save_q: {m_save} frames"
            #m_timeq = f"time_q: {m_time} timestamps"
            #m_exifq = f"exif_q: {m_exif} arrays"
            right_text = 'right camera'
            r_capture = f"capture: {r_capt} FPS"
            r_encode = f"encode: {r_enc} FPS"
            r_sent = f"send: {r_send} FPS"
            r_receive = f"received: {r_rec} FPS"
            r_streamq = f"stream_q: {r_stream} frames"
            r_saveq = f"save_q: {r_save} frames"
            #r_timeq = f"time_q: {r_time} timestamps"
            #r_exifq = f"exif_q: {r_exif} arrays"

            (
                sync_snapshot,
                recording_snapshot,
                current_pps_sequence,
                gps_snapshot,
                ) = runtime_status_snapshot()
            acknowledged_text = ",".join(
                sorted(sync_snapshot["acknowledged_cameras"])
                ) or "none"
            missing_text = ",".join(
                sorted(sync_snapshot["missing_cameras"])
                ) or "none"
            handshake_text = (
                f"Handshake: {sync_snapshot['state']} | "
                f"armed={acknowledged_text} | missing={missing_text}"
                )
            recording_text = (
                f"Mode: {recording_snapshot['state']} | "
                f"target PPS={recording_snapshot['target_pps_sequence']}"
                )
            pps_text = f"AGX PPS sequence: {current_pps_sequence}"
            gps_text = (
                f"UTC[{gps_snapshot['utc_source']}]: "
                f"{gps_snapshot['status']} | "
                f"port={gps_snapshot['port']} | "
                f"connected={gps_snapshot['connected']} | "
                f"UTC pair={gps_snapshot['utc_pair_status']} | "
                f"pairs={gps_snapshot['utc_pair_count']}"
                )
            instruction_text = (
                "Press T after Handshake=SYNCED and UTC pair=PAIRED"
                )
            
            # sets the text color to white, background color to black, font to 24 pt, and fills the screen with the background color
            text_color = (255,255,255)
            bg_color = (0,0,0)
            font = pygame.font.SysFont(None, 24)
            screen.fill(bg_color)
            
            # each image that is available will have a surface created for it and screen.blit will paste the image
            # to the screen at specified coordinates. These coordinates correspond to the top left of the image
            if left_img is not None:
                left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
                left_surface = pygame.surfarray.make_surface(np.swapaxes(left_img, 0, 1))
                screen.blit(left_surface, (0,0))
            if mid_img is not None:
                mid_img = cv2.cvtColor(mid_img, cv2.COLOR_GRAY2RGB)
                mid_surface = pygame.surfarray.make_surface(np.swapaxes(mid_img, 0, 1))
                screen.blit(mid_surface, (640,0))
            if right_img is not None:
                right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)
                right_surface = pygame.surfarray.make_surface(np.swapaxes(right_img, 0, 1))
                screen.blit(right_surface, (1280,0))
                
            # creates surfaces for each line of text to be displayed
            l_text_surface = font.render(left_text, True, text_color)
            lcs = font.render(l_capture, True, text_color)
            les = font.render(l_encode, True, text_color)
            lss = font.render(l_sent, True, text_color)
            lrs = font.render(l_receive, True, text_color)
            lsts = font.render(l_streamq, True, text_color)
            lsas = font.render(l_saveq, True, text_color)
            #lts = font.render(l_timeq, True, text_color)
            #lexs = font.render(l_exifq, True, text_color)
            m_text_surface = font.render(mid_text, True, text_color)
            mcs = font.render(m_capture, True, text_color)
            mes = font.render(m_encode, True, text_color)
            mss = font.render(m_sent, True, text_color)
            mrs = font.render(m_receive, True, text_color)
            msts = font.render(m_streamq, True, text_color)
            msas = font.render(m_saveq, True, text_color)
            #mts = font.render(m_timeq, True, text_color)
            #mexs = font.render(m_exifq, True, text_color)
            r_text_surface = font.render(right_text, True, text_color)
            rcs = font.render(r_capture, True, text_color)
            res = font.render(r_encode, True, text_color)
            rss = font.render(r_sent, True, text_color)
            rrs = font.render(r_receive, True, text_color)
            rsts = font.render(r_streamq, True, text_color)
            rsas = font.render(r_saveq, True, text_color)
            #rts = font.render(r_timeq, True, text_color)
            #rexs = font.render(r_exifq, True, text_color)
            handshake_surface = font.render(
                handshake_text,
                True,
                text_color,
                )
            recording_surface = font.render(
                recording_text,
                True,
                text_color,
                )
            pps_surface = font.render(pps_text, True, text_color)
            gps_surface = font.render(gps_text, True, text_color)
            instruction_surface = font.render(
                instruction_text,
                True,
                text_color,
                )
            
            # creates the rectangles for each text surface to be displayed within giving the coordinates of the center of each
            left_rect = l_text_surface.get_rect(center=(360,552))
            lcr = lcs.get_rect(center=(360,600))
            ler = les.get_rect(center=(360,650))
            lsr = lss.get_rect(center=(360,700))
            lrr = lrs.get_rect(center=(360,750))
            lstr = lsts.get_rect(center=(360,800))
            lsar = lsas.get_rect(center=(360,850))
            #ltr = lts.get_rect(center=(360,900))
            #lexr = lexs.get_rect(center=(360,950))
            mid_rect = m_text_surface.get_rect(center = (1080, 552))
            mcr = mcs.get_rect(center=(1080,600))
            mer = mes.get_rect(center=(1080,650))
            msr = mss.get_rect(center=(1080,700))
            mrr = mrs.get_rect(center=(1080,750))
            mstr = msts.get_rect(center=(1080,800))
            msar = msas.get_rect(center=(1080,850))
            #mtr = mts.get_rect(center=(1080,900))
            #mexr = mexs.get_rect(center=(1080,950))
            right_rect = r_text_surface.get_rect(center = (1800, 552))
            rcr = rcs.get_rect(center=(1800,600))
            rer = res.get_rect(center=(1800,650))
            rsr = rss.get_rect(center=(1800,700))
            rrr = rrs.get_rect(center=(1800,750))
            rstr = rsts.get_rect(center=(1800,800))
            rsar = rsas.get_rect(center=(1800,850))
            #rtr = rts.get_rect(center=(1800,900))
            #rexr = rexs.get_rect(center=(1800,950))
            handshake_rect = handshake_surface.get_rect(center=(960,900))
            recording_rect = recording_surface.get_rect(center=(960,930))
            pps_rect = pps_surface.get_rect(center=(960,960))
            gps_rect = gps_surface.get_rect(center=(960,990))
            instruction_rect = instruction_surface.get_rect(center=(960,1020))
            
            # screen.blit will paste each text surface to the screen on their rectangle
            screen.blit(l_text_surface, left_rect)
            screen.blit(lcs,lcr)
            screen.blit(les,ler)
            screen.blit(lss,lsr)
            screen.blit(lrs, lrr)
            screen.blit(lsts, lstr)
            screen.blit(lsas,lsar)
            #screen.blit(lts, ltr)
            #screen.blit(lexs,lexr)
            screen.blit(m_text_surface, mid_rect)
            screen.blit(mcs,mcr)
            screen.blit(mes,mer)
            screen.blit(mss,msr)
            screen.blit(mrs, mrr)
            screen.blit(msts,mstr)
            screen.blit(msas,msar)
            #screen.blit(mts,mtr)
            #screen.blit(mexs,mexr)
            screen.blit(r_text_surface, right_rect)
            screen.blit(rcs,rcr)
            screen.blit(res,rer)
            screen.blit(rss,rsr)
            screen.blit(rrs, rrr)
            screen.blit(rsts,rstr)
            screen.blit(rsas, rsar)
            #screen.blit(rts,rtr)
            #screen.blit(rexs,rexr)
            screen.blit(handshake_surface, handshake_rect)
            screen.blit(recording_surface, recording_rect)
            screen.blit(pps_surface, pps_rect)
            screen.blit(gps_surface, gps_rect)
            screen.blit(instruction_surface, instruction_rect)
            pygame.display.flip()
            clock.tick(20)
            
    # if a CTRL+C command is sent the script will begin shutting down
    except KeyboardInterrupt:
        stop_event.set()
        
    # also if the display loop crashes the script will exit
    except Exception as e:
        stop_event.set()
        print(f"Display loop error: {e}")
    finally:
        # sets stop_event to cleanly exit the threads, closes the pygame window, closes the sockets,
        # uses de_init to shut down remote scripts, ensures that threads are exited then terminates
        stop_event.set()
        pygame.quit()
        l_sock.close()
        m_sock.close()
        r_sock.close()
        time.sleep(1)
        de_init()
        l_rec_thread.join(timeout=5)
        m_rec_thread.join(timeout=5)
        r_rec_thread.join(timeout=5)
        save_thread.join(timeout=5)
        gps_thread.join(timeout=5)
        pps_thread.join(timeout=2)
        sync_thread.join(timeout=2)
        m_stats_thread.join(timeout=5)
        l_stats_thread.join(timeout=5)
        r_stats_thread.join(timeout=5)
        GPIO.cleanup(PPS_PIN)
        L_tot_fps = [float(x) for x in L_tot_fps]
        L_capt = [float(x) for x in L_capt]
        L_send = [float(x) for x in L_send]
        M_tot_fps= [float(x) for x in M_tot_fps]
        M_capt = [float(x) for x in M_capt]
        M_send = [float(x) for x in M_send]
        R_tot_fps = [float(x) for x in R_tot_fps]
        R_capt = [float(x) for x in R_capt]
        R_send = [float(x) for x in R_send]
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(range(len(L_tot_fps)), L_tot_fps, linestyle='-', alpha=.5)
        ax[0].plot(range(len(L_capt)-7), L_capt[7:], linestyle=':', alpha=1)
        ax[0].plot(range(len(L_send)-7), L_send[7:], linestyle='--', alpha=.5)
        ax[0].legend(['recieved', 'captured', 'sent'])
        ax[0].set_xlabel('time [s]')
        ax[0].set_ylabel('frames per second')
        ax[0].set_title('left camera frame rate vs. time')
        ax[1].plot(range(len(R_tot_fps)), R_tot_fps, linestyle='-', alpha=.5)
        ax[1].plot(range(len(R_capt)-7), R_capt[7:], linestyle=':', alpha=1)
        ax[1].plot(range(len(R_send)-7), R_send[7:], linestyle='--', alpha=.5)
        ax[1].legend(['recieved', 'captured', 'sent'])
        ax[1].set_xlabel('time [s]')
        ax[1].set_ylabel('frames per second')
        ax[1].set_title('right camera frame rate vs. time')
        ax[2].plot(range(len(M_tot_fps)), M_tot_fps, linestyle='-', alpha=.5)
        ax[2].plot(range(len(M_capt)-6), M_capt[6:], linestyle=':', alpha=1)
        ax[2].plot(range(len(M_send)-6), M_send[6:], linestyle='--', alpha=.5)
        ax[2].legend(['recieved', 'captured', 'sent'])
        ax[2].set_xlabel('time [s]')
        ax[2].set_ylabel('frames per second')
        ax[2].set_title('middle camera frame rate vs. time')
        plt.show()
        print('main exiting')
        

if __name__ == "__main__":
    main()
