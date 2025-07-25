import pynmea2
import serial
import pandas as pd

def log_gps_error(lat_err, lon_err, alt_err):
    """
    Log GPS accuracy errors to a csv file.
    Args:
        lat_err (float): Latitude error in meters.
        lon_err (float): Longitude error in meters.
        alt_err (float): Altitude error in meters.
    Output:
        Appends a new line to 'gps_accuracy_log.csv' with the current timestamp and errors.
    """
    if lat_err is None or lon_err is None or alt_err is None:
        return
    data = {
        'timestamp': pd.Timestamp.now(),
        'lat_error': lat_err,
        'lon_error': lon_err,
        'alt_error': alt_err
    }
    df = pd.DataFrame([data])
    df.to_csv('gps_accuracy_log.csv', mode='a', header=not pd.io.common.file_exists('gps_accuracy_log.csv'), index=False)
    return


def read_gps_line(ser):
    """
    Parse GPS data and extract:
    - Position (lat, lon, alt) from GGA
    - Accuracy (HDOP, VDOP, Lat/Lon/Alt errors) from GST/GSA
    Returns (lat, lon, alt, hdop, vdop, lat_err, lon_err, alt_err) if available.
    """
    line = ser.readline().decode('ascii', errors='ignore').strip()
    if not line.startswith('$'):
        return None

    try:
        msg = pynmea2.parse(line)

        # --- Position Data (GGA) ---
        if isinstance(msg, pynmea2.types.talker.GGA):
            print(f"[GGA] Time: {msg.timestamp} | Lat: {msg.latitude}° | Lon: {msg.longitude}° | Alt: {msg.altitude} {msg.altitude_units} | HDOP: {msg.horizontal_dil}")
            return {
                'type': 'GGA',
                'lat': float(msg.latitude),
                'lon': float(msg.longitude),
                'alt': float(msg.altitude) if msg.altitude else None,
                'hdop': float(msg.horizontal_dil) if msg.horizontal_dil else None,
            }

        # --- Accuracy Data (GST) ---
        elif isinstance(msg, pynmea2.types.talker.GST):
            # GST fields: [timestamp, rms, semi-major, semi-minor, orientation, lat_err, lon_err, alt_err]
            # Indices:      0         1    2          3          4           5        6        7
            time_err = msg.timestamp if msg.timestamp else None
            lat_err = float(msg.data[5]) if msg.data[5] else None
            lon_err = float(msg.data[6]) if msg.data[6] else None
            alt_err = float(msg.data[7]) if msg.data[7] else None
            print(f"[GST] Lat Error: {lat_err} m | Lon Error: {lon_err} m | Alt Error: {alt_err} m | time: {time_err} m")
            log_gps_error(time_err, lat_err, lon_err, alt_err)
            return {
                'type': 'GST',
                'lat_err': lat_err,
                'lon_err': lon_err,
                'alt_err': alt_err,
            }

        # --- Heading (HDT) ---
        elif isinstance(msg, pynmea2.types.talker.HDT):
            print(f"[HDT] Heading: {msg.heading}°")
            return None

    except pynmea2.ParseError as e:
        print(f"Parse error: {e} | Line: {line}")
        return None

# Example usage:
ser = serial.Serial('COM10', baudrate=115200, timeout=1)
while True:
    data = read_gps_line(ser)
    if data:
        print("Logged:", data)  # Or save to file/database