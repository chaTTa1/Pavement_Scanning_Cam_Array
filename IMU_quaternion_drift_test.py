import serial
import struct

SERIAL_PORT = 'COM5'
BAUDRATE = 115200

CMD_QUATERNION = 32

def parse_packet(cmd_id, payload):
    if cmd_id == CMD_QUATERNION:
        timestamp, q1, q2, q3, q4 = struct.unpack('<Iffff', payload)
        return timestamp, {'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4}
    else:
        return None, {}

with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1) as ser:
    print("Timestamp (us), q1, q2, q3, q4")
    while True:
        # Read packet header
        if ser.read() != b'\xAA' or ser.read() != b'\x55':
            continue
        length = ser.read(1)
        if not length:
            continue
        length = length[0]

        payload = ser.read(length)
        crc = ser.read(2)

        if len(payload) < 4:
            continue
        header = struct.unpack('<I', payload[:4])[0]
        cmd_id = header & 0x7F
        timestamp, data = parse_packet(cmd_id, payload[4:])
        if timestamp is None:
            continue

        # Print quaternion if present
        if all(k in data for k in ('q1', 'q2', 'q3', 'q4')):
            print(f"{timestamp}, {data['q1']:.6f}, {data['q2']:.6f}, {data['q3']:.6f}, {data['q4']:.6f}")