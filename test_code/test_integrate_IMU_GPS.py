# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:06:36 2026

@author: Desktop
"""

    
#%%

import serial
import struct
import numpy as np

def parse_imu_packet(ser):
    while True:
        # 1. Sync with Header
        b1 = ser.read(1)
        if b1 != b'\xAA': continue
        b2 = ser.read(1)
        if b2 != b'\x55': continue
            
        # 2. Read Length
        length_byte = ser.read(1)
        if not length_byte: return None, {}
        length = length_byte[0]
        
        # 3. Read Payload
        payload = ser.read(length)
        if len(payload) < length: return None, {}
            
        # 4. Check Command ID
        header_info = struct.unpack('<I', payload[:4])[0]
        cmd_id = header_info & 0x7F
        
        if cmd_id == 41:
            data_to_unpack = payload[4:]
            data_len = len(data_to_unpack)
            
            # Determine how many floats we can actually unpack
            # 4 bytes for Timestamp (I) + 4 bytes for each Float (f)
            # If we have at least 28 bytes: Timestamp + 3 Accel + 3 Gyro
            if data_len >= 28:
                # Dynamically build the format string based on available data
                # We subtract 4 for the 'I', then divide the rest by 4 for 'f's
                num_floats = (data_len - 4) // 4
                fmt = '<I' + ('f' * num_floats)
                
                # Only slice exactly what the format string needs
                required_bytes = 4 + (num_floats * 4)
                vals = struct.unpack(fmt, data_to_unpack[:required_bytes])
                
                return vals[0], {
                    'accel_x': vals[1], 'accel_y': vals[2], 'accel_z': vals[3],
                    'gyro_x': vals[4], 'gyro_y': vals[5], 'gyro_z': vals[6]
                }
        return None, {}

# --- REST OF YOUR CODE (Setup and While Loop) ---
port_name = 'COM12' 
baud_rate = 115200

try:
    imu_ser = serial.Serial(port_name, baud_rate, timeout=0.1)
    print(f"--- Monitoring Raw Data on {port_name} ---")

    while True:
        imu_timestamp, imu_data = parse_imu_packet(imu_ser)
        if imu_data:
            gyro = np.array([imu_data.get('gyro_x'), imu_data.get('gyro_y'), imu_data.get('gyro_z')])
            accel = np.array([imu_data.get('accel_x'), imu_data.get('accel_y'), imu_data.get('accel_z')])
            print(f"TS: {imu_timestamp} | Accel: {accel} | Gyro: {gyro}")
            
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    if 'imu_ser' in locals(): imu_ser.close()


#%%

import serial
import struct
import numpy as np

def parse_imu_packet(ser):
    while True:
        # 1. Sync with Header
        b1 = ser.read(1)
        if b1 != b'\xAA': continue
        b2 = ser.read(1)
        if b2 != b'\x55': continue
            
        # 2. Read Length
        length_byte = ser.read(1)
        if not length_byte: return None, None
        length = length_byte[0]
        
        # 3. Read Payload
        payload = ser.read(length)
        if len(payload) < length: return None, None
            
        # 4. Check Command ID (41 = Raw Data)
        header_info = struct.unpack('<I', payload[:4])[0]
        cmd_id = header_info & 0x7F
        
        if cmd_id == 41:
            data_to_unpack = payload[4:]
            data_len = len(data_to_unpack)
            
            # Ensure we have at least Timestamp + Accel + Gyro (28 bytes)
            if data_len >= 28:
                num_floats = (data_len - 4) // 4
                fmt = '<I' + ('f' * num_floats)
                required_bytes = 4 + (num_floats * 4)
                
                vals = struct.unpack(fmt, data_to_unpack[:required_bytes])
                
                # vals[0] = Timestamp, vals[1-3] = Accel, vals[4-6] = Gyro
                return vals[0], vals
        return None, None

# --- Main Execution ---
PORT = 'COM12' 
BAUD = 115200

try:
    imu_ser = serial.Serial(PORT, BAUD, timeout=0.1)
    
    # Print the Header exactly like the GUI text file
    print("rawData = [ ")
    print("%   File created by Python Script.")
    print("%   NAME:  timeStamp,   gyroX,   gyroY,   gyroZ,     accX,   accY,   accZ ;")
    print("%   UNIT:   Second  ,   rad/s,   rad/s,   rad/s,       g ,     g ,     g ;")

    while True:
        ts_raw, all_vals = parse_imu_packet(imu_ser)
        
        if all_vals:
            # Match the GUI units and formatting:
            # 1. Timestamp to seconds
            timestamp = ts_raw / 1000.0
            
            # 2. Extract Accel and Gyro (Indices match the <Ifffffff format)
            # GUI Order is: Time, GyroX, GyroY, GyroZ, AccelX, AccelY, AccelZ
            accX, accY, accZ = all_vals[1], all_vals[2], all_vals[3]
            gyrX, gyrY, gyrZ = all_vals[4], all_vals[5], all_vals[6]

            # 3. Formatted Print String
            # {value:10.6f} means 10 characters total width, 6 decimals
            line = (f"          {timestamp:10.6f}, "
                    f"{gyrX:16.9f}, {gyrY:16.9f}, {gyrZ:16.9f}, "
                    f"{accX:16.9f}, {accY:16.9f}, {accZ:16.9f} ;")
            print(line)
            
except KeyboardInterrupt:
    print("];") # Close the MATLAB array format
finally:
    if 'imu_ser' in locals():
        imu_ser.close()