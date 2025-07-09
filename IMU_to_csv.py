import serial
import struct
import csv
import time
import datetime

# Set up serial port and baud rate
SERIAL_PORT = 'COMX'  # Replace with your actual COM port
BAUDRATE = 115200

# Output CSV file
CSV_FILE = 'IMU_stonemill_test.csv'

# Define CMD IDs
CMD_QUATERNION = 32
CMD_RPY = 35
CMD_RAW = 41

# Define all possible columns for all types
CSV_COLUMNS = [
    'timestamp',  # IMU microseconds
    'gps_time',   # Absolute GPS time in seconds since GPS epoch
    'q1', 'q2', 'q3', 'q4',           # quaternion
    'roll', 'pitch', 'yaw',           # rpy
    'accel_x', 'accel_y', 'accel_z',  # raw
    'gyro_x', 'gyro_y', 'gyro_z',     # raw
    'mag_x', 'mag_y', 'mag_z'         # raw
]

def utc_to_gps_week_and_tow(dt):
    gps_epoch = datetime.datetime(1980, 1, 6, 0, 0, 0, tzinfo=datetime.timezone.utc)
    delta = dt - gps_epoch
    gps_seconds = delta.total_seconds()
    week = int(gps_seconds // 604800)
    tow = gps_seconds % 604800
    return week, tow

def parse_packet(cmd_id, payload):
    if cmd_id == CMD_QUATERNION:
        timestamp, q1, q2, q3, q4 = struct.unpack('<Iffff', payload)
        return timestamp, {'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4}
    elif cmd_id == CMD_RPY:
        timestamp, roll, pitch, yaw = struct.unpack('<Ifff', payload)
        return timestamp, {'roll': roll, 'pitch': pitch, 'yaw': yaw}
    elif cmd_id == CMD_RAW:
        vals = struct.unpack('<Ifff fff fff', payload)
        return vals[0], {
            'accel_x': vals[1], 'accel_y': vals[2], 'accel_z': vals[3],
            'gyro_x': vals[4], 'gyro_y': vals[5], 'gyro_z': vals[6],
            'mag_x': vals[7], 'mag_y': vals[8], 'mag_z': vals[9]
        }
    else:
        return None, {}

def main():
    # Get GPS week and TOW at script start
    now = datetime.datetime.now(datetime.timezone.utc)
    gps_week, gps_tow = utc_to_gps_week_and_tow(now)
    print(f"IMU logging started at GPS Week: {gps_week}, TOW: {gps_tow:.2f}")

    with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1) as ser, \
         open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        current_timestamp = None
        row_data = {}
        imu_start_timestamp = None

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

            # Set IMU start timestamp on first packet
            if imu_start_timestamp is None:
                imu_start_timestamp = timestamp

            # Calculate gps_time as TOW for this row
            imu_time_sec = (timestamp - imu_start_timestamp) / 1e6
            gps_tow_this = (gps_tow + imu_time_sec + 18) % 604800  # add leap second offset

            # If new timestamp, write previous row (if any)
            if current_timestamp is not None and timestamp != current_timestamp:
                row_data['timestamp'] = current_timestamp
                row_data['gps_time'] = prev_gps_tow
                writer.writerow(row_data)
                row_data = {}

            # Merge new data
            current_timestamp = timestamp
            prev_gps_tow = gps_tow_this
            row_data.update(data)

if __name__ == '__main__':
    try:
        print(f"Logging to {CSV_FILE}...")
        main()
    except KeyboardInterrupt:
        print("\nLogging stopped.")
