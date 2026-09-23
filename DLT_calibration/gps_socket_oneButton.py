"""Single-receiver GPS point collector. Close the window or use Ctrl+C to stop."""
import csv
import math
from pathlib import Path
import socket
import threading
import time
from datetime import datetime, timezone
import tkinter as tk

# Laptop joins ESP_XBee_2; NTRIP Master joins the phone hotspot.
HOST = "192.168.4.1"
PORT = 5000
CSV_FILENAME = Path(__file__).resolve().with_name("GPS_points.csv")
MAX_POINT_AGE_SECONDS = 3.0
RECONNECT_SECONDS = 2.0
REQUIRE_RTK_FIXED = False  # Also adjustable in the GUI.
FIELDNAMES = ["save_timestamp", "pc_timestamp", "utc_time", "latitude",
              "longitude", "altitude_m", "fix_quality", "satellites"]
FIX_NAMES = {0: "INVALID", 1: "Standalone GNSS", 2: "Differential GNSS",
             3: "PPS", 4: "RTK FIXED", 5: "RTK FLOAT",
             6: "Estimated / dead reckoning", 7: "Manual", 8: "Simulation"}
VALID_FIXES = {1, 2, 3, 4, 5}
latest_gps = None
latest_received = None
connected = False
connection_status = "Connecting..."
gps_lock = threading.Lock()
stop_event = threading.Event()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def nmea_to_decimal(coord, direction):
    if direction not in ("N", "S", "E", "W"):
        raise ValueError("Invalid direction")
    digits = 2 if direction in ("N", "S") else 3
    if len(coord.split(".")[0]) != digits + 2:
        raise ValueError("Invalid coordinate width")
    degrees, minutes = int(coord[:digits]), float(coord[digits:])
    limit = 90 if digits == 2 else 180
    if not math.isfinite(minutes) or not 0 <= minutes < 60:
        raise ValueError("Invalid minutes")
    if not 0 <= degrees <= limit or (degrees == limit and minutes != 0):
        raise ValueError("Coordinate out of range")
    value = degrees + minutes / 60
    return -value if direction in ("S", "W") else value


def parse_gga(sentence):
    """Ignore other sentences; reject damaged GGA; expose invalid fixes."""
    if not sentence.startswith("$"):
        return None
    identifier = sentence.split(",", 1)[0]
    if len(identifier) != 6 or identifier[3:] != "GGA":
        return None
    body, separator, checksum = sentence[1:].partition("*")
    if not separator or len(checksum) != 2:
        raise ValueError("Missing or malformed checksum")
    computed = 0
    for character in body:
        computed ^= ord(character)
    if computed != int(checksum, 16):
        raise ValueError("Checksum mismatch")
    parts = body.split(",")
    if len(parts) < 11:
        raise ValueError("Incomplete GGA")
    quality = int(parts[6] or 0)
    if quality not in VALID_FIXES or not parts[2] or not parts[4]:
        return {"fix_quality": quality, "latitude": None}
    if parts[3] not in ("N", "S") or parts[5] not in ("E", "W"):
        raise ValueError("Invalid latitude/longitude directions")
    altitude = float(parts[9]) if parts[9] else None
    if altitude is not None and (not math.isfinite(altitude) or parts[10] != "M"):
        raise ValueError("Invalid altitude or units")
    return {"utc_time": parts[1],
            "latitude": nmea_to_decimal(parts[2], parts[3]),
            "longitude": nmea_to_decimal(parts[4], parts[5]),
            "altitude_m": altitude, "fix_quality": quality,
            "satellites": int(parts[7] or 0)}


def set_connection(is_connected, status):
    global connected, connection_status, latest_gps, latest_received
    with gps_lock:
        connected, connection_status = is_connected, status
        if not is_connected:
            latest_gps = latest_received = None


def receive_gps():
    """Receive without transmitting; reconnect and skip damaged sentences."""
    global latest_gps, latest_received
    while not stop_event.is_set():
        set_connection(False, f"Connecting to {HOST}:{PORT}...")
        try:
            with socket.create_connection((HOST, PORT), timeout=3) as stream:
                stream.settimeout(1)
                set_connection(True, f"Connected to {HOST}:{PORT}")
                buffer = b""
                while not stop_event.is_set():
                    try:
                        data = stream.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        raise ConnectionError("GPS connection closed")
                    buffer += data
                    while b"\n" in buffer:
                        raw_line, buffer = buffer.split(b"\n", 1)
                        try:
                            point = parse_gga(raw_line.decode("ascii").strip())
                        except (ValueError, UnicodeError):
                            continue
                        if point is None:
                            continue
                        point["pc_timestamp"] = utc_now()
                        with gps_lock:
                            latest_gps = point
                            latest_received = time.monotonic()
                    if len(buffer) > 65536:
                        buffer = b""
        except OSError as error:
            set_connection(False, f"Disconnected: {error}. Retrying...")
        if not stop_event.is_set():
            stop_event.wait(RECONNECT_SECONDS)
    set_connection(False, "Stopped")


def get_snapshot():
    with gps_lock:
        return (latest_gps.copy() if latest_gps else None,
                latest_received, connected, connection_status)


def save_block_reason(point, received, is_connected, require_fixed):
    if not is_connected:
        return "GPS disconnected; waiting to reconnect."
    if point is None or received is None:
        return "Waiting for a GPS position."
    if time.monotonic() - received > MAX_POINT_AGE_SECONDS:
        return "GPS data is stale; waiting for a fresh position."
    if point.get("latitude") is None or point["fix_quality"] not in VALID_FIXES:
        return "No valid measured GPS fix."
    if require_fixed and point["fix_quality"] != 4:
        return "Waiting for RTK FIXED."
    return ""


def append_point(point):
    """Keep the original CSV columns and write exactly one row."""
    with open(CSV_FILENAME, "a+", newline="", encoding="utf-8") as csvfile:
        csvfile.seek(0)
        header = next(csv.reader(csvfile), None)
        if header is not None and header != FIELDNAMES:
            raise ValueError("Existing CSV columns differ; rename the file before saving.")
        csvfile.seek(0, 2)
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if header is None:
            writer.writeheader()
        writer.writerow(point)


def save_gps_point():
    point, received, is_connected, _ = get_snapshot()
    reason = save_block_reason(point, received, is_connected, require_fixed.get())
    if reason:
        status_label.config(text=reason)
        return
    point["save_timestamp"] = utc_now()
    try:
        append_point(point)
    except (OSError, ValueError, csv.Error) as error:
        status_label.config(text=f"Could not save: {error}")
        return
    status_label.config(text=f"Saved at {point['save_timestamp']}\n"
                             f"{point['latitude']:.8f}, {point['longitude']:.8f}")


def update_display():
    point, received, is_connected, status = get_snapshot()
    reason = save_block_reason(point, received, is_connected, require_fixed.get())
    lines = [status]
    if point is not None:
        quality = point["fix_quality"]
        lines.append(f"Fix: {FIX_NAMES.get(quality, 'Unknown')} ({quality})")
        if point.get("latitude") is not None:
            altitude = point["altitude_m"]
            altitude_text = f"{altitude:.3f} m" if altitude is not None else "Unavailable"
            lines.extend([f"Latitude: {point['latitude']:.8f}",
                          f"Longitude: {point['longitude']:.8f}",
                          f"Altitude: {altitude_text}",
                          f"Satellites: {point['satellites']}"])
        lines.append(f"Data age: {time.monotonic() - received:.1f} seconds")
    lines.append(reason or "Ready to save latest received point")
    gps_display.config(text="\n".join(lines))
    save_button.config(state=tk.DISABLED if reason else tk.NORMAL)
    if not stop_event.is_set():
        root.after(250, update_display)


def close_program():
    stop_event.set()
    root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("GPS Point Collector")
    root.geometry("560x570")
    tk.Label(root, text="GPS Point Collector", font=("Arial", 20, "bold")).pack(pady=15)
    gps_display = tk.Label(root, text="Connecting...", font=("Arial", 12),
                           justify="left", wraplength=520)
    gps_display.pack(padx=15, pady=10)
    require_fixed = tk.BooleanVar(value=REQUIRE_RTK_FIXED)
    tk.Checkbutton(root, text="Require RTK FIXED before saving",
                   variable=require_fixed).pack(pady=5)
    save_button = tk.Button(root, text="SAVE GPS POINT", font=("Arial", 18, "bold"),
                            height=2, command=save_gps_point, state=tk.DISABLED)
    save_button.pack(pady=15)
    status_label = tk.Label(root, text="No point saved yet.", justify="left", wraplength=520)
    status_label.pack(padx=15, pady=5)
    tk.Label(root, text=f"CSV: {CSV_FILENAME}", wraplength=520).pack(padx=15, pady=10)
    root.protocol("WM_DELETE_WINDOW", close_program)
    gps_thread = threading.Thread(target=receive_gps, daemon=True)
    gps_thread.start()
    try:
        update_display()
        root.mainloop()
    except KeyboardInterrupt:
        close_program()
    finally:
        stop_event.set()
        gps_thread.join(timeout=4)
