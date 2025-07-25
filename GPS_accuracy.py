import pynmea2
import serial

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
            lat_err = float(msg.data[5]) if msg.data[5] else None
            lon_err = float(msg.data[6]) if msg.data[6] else None
            alt_err = float(msg.data[7]) if msg.data[7] else None
            rms = float(msg.data[1]) if msg.data[1] else None
            print(f"[GST] Lat Error: {lat_err} m | Lon Error: {lon_err} m | Alt Error: {alt_err} m | RMS: {rms} m")
            return {
                'type': 'GST',
                'lat_err': lat_err,
                'lon_err': lon_err,
                'alt_err': alt_err,
                'rms': rms,
            }

        # --- Dilution of Precision (GSA) ---
        elif isinstance(msg, pynmea2.types.talker.GSA):
            print(f"[GSA] HDOP: {msg.hdop} | VDOP: {msg.vdop} | PDOP: {msg.pdop}")
            return {
                'type': 'GSA',
                'hdop': float(msg.hdop) if msg.hdop else None,
                'vdop': float(msg.vdop) if msg.vdop else None,
                'pdop': float(msg.pdop) if msg.pdop else None,
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