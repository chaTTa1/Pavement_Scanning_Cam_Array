import serial
from pysbf2 import SBFReader, SBF_PROTOCOL

ser = serial.Serial('COM3', 115200, timeout=1)

reader = SBFReader(
    ser,
    protfilter=SBF_PROTOCOL,
    quitonerror=1,
    validate=1
)
print(reader)
while True:
    raw, msg = reader.read()
    if not msg:
        print("No message received or error in reading.")
        print(f"Raw data: {raw}")
        continue

    if msg.name == "PVTGeodetic":
        tow = msg.fields["TOW"]
        lat = msg.fields["Latitude"]
        lon = msg.fields["Longitude"]
        alt = msg.fields["Height"]
        print(f"TOW: {tow} ms | Lat: {lat:.6f} | Lon: {lon:.6f} | Alt: {alt:.2f} m")

    elif msg.name == "Heading":
        heading = msg.fields["Heading"]
        print(f"Heading: {heading:.2f}°")
    else:
        print("Fail")
        print(f"Unknown message: {msg.name}")