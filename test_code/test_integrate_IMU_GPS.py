# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:06:36 2026

@author: Desktop
"""


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
        
#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd

filename = r"C:\Users\Desktop\Desktop\DataLogin_NodeID-491_04-16-2026_15-35-50_RawData.txt"

# 1. Parse Data
data = []
with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('%') or 'rawData' in line or line == '];':
            continue
        line = line.rstrip(';')
        parts = [float(p.strip()) for p in line.split(',') if p.strip()]
        if len(parts) >= 11:
            data.append(parts)

data = np.array(data)
t = data[:, 0]
gyro = data[:, 2:5] # rad/s
acc = data[:, 5:8] * 9.80665 # g to m/s^2

# 2. Bias Calibration (using first 50 static samples)
# Assuming device was still at the very start
initial_acc_avg = np.mean(acc[:50], axis=0)
# We know Z should be 9.81 m/s^2 (downward)
# We calculate the bias so that a_net starts close to zero in the body frame relative to gravity
gravity_mag = 9.80665

# 3. Integration Function
def calculate_path_20hz(method='constant'):
    pos = np.zeros(3)
    vel = np.zeros(3)
    orientation = R.from_euler('xyz', [0, 0, 0])
    gravity_vec = np.array([0, 0, gravity_mag])
    
    output_path = []
    output_interval = 0.05
    next_output_time = t[0] + output_interval
    
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        if dt <= 0: continue
        
        # Track Orientation
        orientation *= R.from_rotvec(gyro[i] * dt)
        
        # Net Acc in world frame
        # Subtracting initial average acc to center it (Basic Zeroing)
        # Note: In real life, one would subtract gravity in world frame
        a_net = orientation.apply(acc[i]) - gravity_vec
        
        v_old = vel.copy()
        
        if method == 'euler':
            pos += v_old * dt
            vel += a_net * dt
        elif method == 'constant':
            pos += (v_old * dt) + (0.5 * a_net * (dt**2))
            vel += a_net * dt
        elif method == 'trapezoidal':
            v_new = v_old + a_net * dt
            pos += 0.5 * (v_old + v_new) * dt
            vel = v_new
            
        if t[i+1] >= next_output_time:
            output_path.append([t[i+1], pos[0], pos[1], pos[2]])
            next_output_time += output_interval
            
    return np.array(output_path)

# Calculate for all 3 methods
path_euler = calculate_path_20hz('euler')
path_const = calculate_path_20hz('constant')
path_trap = calculate_path_20hz('trapezoidal')

# 4. Plotting
plt.figure(figsize=(10, 8))
plt.plot(path_euler[:, 1], path_euler[:, 2], '--', label='Euler (Standard)', alpha=0.7)
plt.plot(path_const[:, 1], path_const[:, 2], '-', label='Constant Accel (0.5at²)', alpha=0.9)
plt.plot(path_trap[:, 1], path_trap[:, 2], ':', label='Trapezoidal (Avg Vel)', linewidth=2)

plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('20Hz Resampled Path Comparison (3 Integration Methods)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.axis('equal')
plt.savefig('comparison_20hz_path.png')

# Save to CSV
# We use the timestamp from one of them (they are all aligned)
results_df = pd.DataFrame({
    'timestamp': path_euler[:, 0],
    'euler_x': path_euler[:, 1], 'euler_y': path_euler[:, 2],
    'const_x': path_const[:, 1], 'const_y': path_const[:, 2],
    'trap_x': path_trap[:, 1], 'trap_y': path_trap[:, 2]
})
results_df.to_csv('path_comparison_20hz.csv', index=False)

print("Calculation complete. CSV saved as 'path_comparison_20hz.csv'.")

#%%




"""
#	Issue	Severity	Type
1	Gravity sign convention mismatch	🔴 Critical	Bug
2	No gyroscope bias removal	🔴 Critical	Bug
3	Hardcoded calibration sample count	🟡 Medium	Robustness
4	Euler method labeling / order	🟢 Low	Clarity
5	Rotation update not method-matched	🟡 Medium	Physics
6	Empty path crash risk	🟡 Medium	Robustness
7	CSV save commented out but message isn't	🟢 Low	Misleading
8	Skipped column undocumented	🟢 Low	Clarity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
INPUT_FILE = r"C:\Users\Desktop\Desktop\DataLogin_NodeID-491_04-16-2026_15-35-50_RawData.txt"
OUTPUT_FILE = 'IMU_Integration_Comparison_20Hz.csv'
OUTPUT_INTERVAL = 0.05  # 20Hz output rate
GRAVITY_MAG = 9.80665   # Standard gravity in m/s^2

def parse_imu_data(filename):
    """Parses the TM151 RawData text file format."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip headers and array syntax
            if not line or line.startswith('%') or 'rawData' in line or line == '];':
                continue
            # Clean trailing semicolon and split
            parts = [float(p.strip()) for p in line.rstrip(';').split(',') if p.strip()]
            if len(parts) >= 8:
                data.append(parts)
    return np.array(data)


# 1. Load Data
raw_data = parse_imu_data(INPUT_FILE)
t = raw_data[:, 0]
gyro = raw_data[:, 2:5] # rad/s
acc = raw_data[:, 5:8] * GRAVITY_MAG # Convert g to m/s^2

# 2. Static Calibration
# Use the first 0.5 seconds of data to estimate initial bias (assuming still)
calibration_samples = 200 
avg_accel_initial = np.mean(acc[:calibration_samples], axis=0)
# The error is the difference between measured acceleration and vertical gravity
accel_bias = avg_accel_initial - np.array([0, 0, -GRAVITY_MAG])

# 3. Integration Setup
# Initialize states for 3 methods
methods = ['euler', 'constant_accel', 'trapezoidal']
states = {m: {'pos': np.zeros(3), 'vel': np.zeros(3), 'quat': R.from_euler('xyz', [0,0,0]), 'path': []} for m in methods}

gravity_vec = np.array([0, 0, GRAVITY_MAG])
next_output_time = t[0] + OUTPUT_INTERVAL

# 4. Processing Loop (400Hz)
for i in range(len(t) - 1):
    dt = t[i+1] - t[i]
    if dt <= 0: continue
    
    # Calculate rotation update (Shared for all methods)
    omega = gyro[i]
    delta_rot = R.from_rotvec(omega * dt)
    
    # Calculate acceleration in Global Frame (Shared)
    # Note: We subtract the bias estimated during calibration
    for m in methods:
        # Update orientation
        states[m]['quat'] = states[m]['quat'] * delta_rot
        
        # Rotate Accel to World Frame and Remove Gravity
        acc_world = states[m]['quat'].apply(acc[i] - accel_bias)
        a_net = acc_world - gravity_vec
        
        v_old = states[m]['vel'].copy()
        
        # Apply Equations
        if m == 'euler':
            # P_new = P_old + V_old * dt
            states[m]['pos'] += v_old * dt
            states[m]['vel'] += a_net * dt
            
        elif m == 'constant_accel':
            # P_new = P_old + V_old * dt + 0.5 * a * dt^2
            states[m]['pos'] += (v_old * dt) + (0.5 * a_net * (dt**2))
            states[m]['vel'] += a_net * dt
            
        elif m == 'trapezoidal':
            # V_new = V_old + a * dt
            # P_new = P_old + 0.5 * (V_old + V_new) * dt
            v_new = v_old + (a_net * dt)
            states[m]['pos'] += 0.5 * (v_old + v_new) * dt
            states[m]['vel'] = v_new

    # 5. Handle 20Hz Resampling
    if t[i+1] >= next_output_time:
        for m in methods:
            states[m]['path'].append([t[i+1], states[m]['pos'][0], states[m]['pos'][1], states[m]['pos'][2]])
        next_output_time += OUTPUT_INTERVAL

# 6. Data Packaging & Comparison
results = {'timestamp': np.array(states['euler']['path'])[:, 0]}
for m in methods:
    path_arr = np.array(states[m]['path'])
    results[f'{m}_x'] = path_arr[:, 1]
    results[f'{m}_y'] = path_arr[:, 2]
    results[f'{m}_z'] = path_arr[:, 3]

df_final = pd.DataFrame(results)
# df_final.to_csv(OUTPUT_FILE, index=False)

# 7. Plotting
plt.figure(figsize=(10, 6))
plt.plot(df_final['constant_accel_x'], df_final['constant_accel_y'], 'b-', label='Constant Accel (0.5at²)')
plt.plot(df_final['trapezoidal_x'], df_final['trapezoidal_y'], 'r--', label='Trapezoidal')
plt.plot(df_final['euler_x'], df_final['euler_y'], 'g:', label='Euler')

plt.title('Path Comparison @ 20Hz Output (Integrated from 400Hz IMU)')
plt.xlabel('X Distance (meters)')
plt.ylabel('Y Distance (meters)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

print(f"Success! 20Hz path saved to {OUTPUT_FILE}")


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# ============================================================
# GLOBAL PLOT CONFIGURATION (Larger Text)
# ============================================================
plt.rcParams.update({
    'font.size': 20,          # Base font size
    'axes.titlesize': 20,      # Title size
    'axes.labelsize': 20,      # X and Y label size
    'xtick.labelsize': 20,     # X tick size
    'ytick.labelsize': 20,     # Y tick size
    'legend.fontsize': 20,     # Legend size
    'figure.titlesize': 20     # Figure title size
})

# --- CONFIGURATION ---
INPUT_FILE = r"C:\Users\Desktop\Desktop\DataLogin_NodeID-491_04-16-2026_15-35-50_RawData.txt"
OUTPUT_FILE = 'IMU_Integration_Comparison_20Hz.csv'
OUTPUT_INTERVAL = 0.05  # 20Hz output rate
GRAVITY_MAG = 9.80665   # Standard gravity in m/s^2
CALIBRATION_DURATION = 0.5  # seconds of static data for bias estimation

def parse_imu_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%') or 'rawData' in line or line == '];':
                continue
            parts = [float(p.strip()) for p in line.rstrip(';').split(',') if p.strip()]
            if len(parts) >= 8:
                data.append(parts)
    if not data:
        raise ValueError(f"No valid data rows found in '{filename}'.")
    return np.array(data)

# 1. Load Data
raw_data = parse_imu_data(INPUT_FILE)
t = raw_data[:, 0]
gyro = raw_data[:, 2:5]                # rad/s
acc  = raw_data[:, 5:8] * GRAVITY_MAG  # Convert g → m/s²

# 2. Static Calibration
calibration_samples = np.searchsorted(t, t[0] + CALIBRATION_DURATION)
calibration_samples = max(calibration_samples, 10)

avg_accel_initial = np.mean(acc[:calibration_samples], axis=0)
accel_bias = avg_accel_initial - np.array([0, 0, GRAVITY_MAG])
gyro_bias = np.mean(gyro[:calibration_samples], axis=0)

# 3. Integration Setup
gravity_world = np.array([0, 0, GRAVITY_MAG])
methods = ['euler', 'constant_accel', 'trapezoidal']
states = {m: {'pos': np.zeros(3), 'vel': np.zeros(3), 'quat': R.from_euler('xyz', [0, 0, 0]), 'path': []} for m in methods}
next_output_time = t[0] + OUTPUT_INTERVAL

# 4. Processing Loop (400 Hz)
for i in range(len(t) - 1):
    dt = t[i + 1] - t[i]
    if dt <= 0: continue

    omega = gyro[i] - gyro_bias
    delta_rot = R.from_rotvec(omega * dt)
    acc_body_corrected = acc[i] - accel_bias

    for m in methods:
        states[m]['quat'] = states[m]['quat'] * delta_rot
        acc_world = states[m]['quat'].apply(acc_body_corrected)
        a_net = acc_world - gravity_world
        v_old = states[m]['vel'].copy()

        if m == 'euler':
            states[m]['vel'] += a_net * dt
            states[m]['pos'] += v_old * dt
        elif m == 'constant_accel':
            states[m]['pos'] += (v_old * dt) + (0.5 * a_net * dt**2)
            states[m]['vel'] += a_net * dt
        elif m == 'trapezoidal':
            v_new = v_old + a_net * dt
            states[m]['pos'] += 0.5 * (v_old + v_new) * dt
            states[m]['vel'] = v_new

    if t[i + 1] >= next_output_time:
        for m in methods:
            states[m]['path'].append([t[i + 1], states[m]['pos'][0], states[m]['pos'][1], states[m]['pos'][2]])
        next_output_time += OUTPUT_INTERVAL

# 6. Data Packaging & Export
results = {'timestamp': np.array(states['euler']['path'])[:, 0]}
for m in methods:
    path_arr = np.array(states[m]['path'])
    results[f'{m}_x'] = path_arr[:, 1]; results[f'{m}_y'] = path_arr[:, 2]; results[f'{m}_z'] = path_arr[:, 3]

df_final = pd.DataFrame(results)
df_final.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# 7. Plotting (With Large Text Config)
# ============================================================
plot_styles = {
    'constant_accel': {'color': '#1f77b4', 'ls': '-',  'lw': 2.5, 'label': 'Constant Accel'},
    'trapezoidal':    {'color': '#d62728', 'ls': '--', 'lw': 2.5, 'label': 'Trapezoidal'},
    'euler':          {'color': '#2ca02c', 'ls': ':',  'lw': 2.5, 'label': 'Euler'},
}

# FIGURE 1: 2D Path
fig1, ax1 = plt.subplots(figsize=(12, 8))
for m, style in plot_styles.items():
    ax1.plot(df_final[f'{m}_x'], df_final[f'{m}_y'], **style)
ax1.set_title('2D Path (X–Y) @ 20 Hz')
ax1.set_xlabel('X Distance (m)')
ax1.set_ylabel('Y Distance (m)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# FIGURE 2: 3D Path
fig3 = plt.figure(figsize=(14, 10))
ax3d = fig3.add_subplot(111, projection='3d')
for m, style in plot_styles.items():
    ax3d.plot(df_final[f'{m}_x'], df_final[f'{m}_y'], df_final[f'{m}_z'], **style)

ax3d.set_title('3D Path Comparison @ 20 Hz', pad=20)
ax3d.set_xlabel('X (m)', labelpad=15)
ax3d.set_ylabel('Y (m)', labelpad=15)
ax3d.set_zlabel('Z (m)', labelpad=15)
ax3d.legend()
plt.show()