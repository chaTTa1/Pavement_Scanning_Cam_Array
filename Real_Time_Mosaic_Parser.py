import serial
import pynmea2

ser = serial.Serial('COM3', 115200, timeout=1)

while True:
    line = ser.readline().decode('ascii', errors='ignore').strip()
    if not line.startswith('$'):
        continue
    try:
        msg = pynmea2.parse(line)

        if isinstance(msg, pynmea2.types.talker.GGA):
            print(f"[GGA] TOW: {msg.timestamp} | Lat: {msg.latitude}° | Lon: {msg.longitude}° | Alt: {msg.altitude} {msg.altitude_units}")

        elif isinstance(msg, pynmea2.types.talker.HDT):
            print(f"[HDT] Heading: {msg.heading}°")

        elif isinstance(msg, pynmea2.types.talker.RMC):
            print(f"[RMC] Time: {msg.timestamp}, Date: {msg.datestamp}")

    except pynmea2.ParseError:
        continue