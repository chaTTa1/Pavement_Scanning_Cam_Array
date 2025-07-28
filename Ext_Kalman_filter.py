import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from scipy.spatial.transform import Rotation as R
import serial
import pynmea2
import struct
import time
from geopy.distance import geodesic
from geopy.point import Point
import pandas as pd
import os

def quaternion_propagate(q, omega, dt):
    """
    Propagate quaternion with angular velocity omega over dt
    """
    r = R.from_quat(q)
    delta_r = R.from_rotvec(omega * dt)
    new_r = r * delta_r
    return new_r.as_quat()


def fx(x, dt, u):
    """
    State transition function
    x: state vector [pos, vel, quat, gyro_bias, acc_bias]
    u: control input [gyro_raw (3,), acc_raw (3,)]
    dt: time step in seconds
    """
    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    b_g = x[10:13]
    b_a = x[13:16]

    # Split control input into gyro and accel
    gyro = u[:3] - b_g
    acc = u[3:6] - b_a

    # Normalize quaternion to prevent numerical instability
    quat = quat / np.linalg.norm(quat)
    r = R.from_quat(quat)
    acc_world = r.apply(acc)
    acc_world[2] -= 9.81  # remove gravity

    new_pos = pos + vel * dt + 0.5 * acc_world * dt**2
    new_vel = vel + acc_world * dt
    new_quat = quaternion_propagate(quat, gyro, dt)

    # Bias random walk (small process noise)
    new_b_g = b_g
    new_b_a = b_a

    x_new = np.zeros(16)
    x_new[0:3] = new_pos
    x_new[3:6] = new_vel
    x_new[6:10] = new_quat
    x_new[10:13] = new_b_g
    x_new[13:16] = new_b_a
    return x_new


def hx(x):
    """Measurement function - GPS measures position only"""
    return x[0:3]

def log_position_and_error(timestamp, gps_pos, gps_error=None):
    """
    Log position data and errors for later verification
    
    Args:
        timestamp: Timestamp of the measurement
        gps_pos: GPS position as (lat, lon, alt)
        gps_error: Optional tuple of (lat_err, lon_err, alt_err)
    """
    data = {
        'timestamp': pd.Timestamp.now(),
        'lat': gps_pos[0],
        'lon': gps_pos[1],
        'alt': gps_pos[2],
    }
    
    # Add error data if available
    if gps_error is not None:
        data.update({
            'lat_err': gps_error[0],
            'lon_err': gps_error[1],
            'alt_err': gps_error[2],
        })
    
    df = pd.DataFrame([data])
    df.to_csv(
        'position_log.csv',
        mode='a',
        header=not os.path.exists('position_log.csv'),
        index=False
    )

def H_jacobian(x):
    """Jacobian of measurement function"""
    H = np.zeros((3, 16))
    H[:, :3] = np.eye(3)
    return H


class SensorFusionEKF:
    def __init__(self):
        # Initialize EKF
        self.ekf = ExtendedKalmanFilter(dim_x=16, dim_z=3)
        
        # Initial state - all zeros except quaternion (identity)
        self.ekf.x = np.zeros(16)
        self.ekf.x[6] = 1.0  # Identity quaternion
        
        # Initial covariance - high uncertainty
        self.ekf.P = np.eye(16) * 100
        
        # Process noise - tune these values based on your sensors
        self.ekf.Q = np.eye(16) * 0.1
        self.ekf.Q[0:3, 0:3] = np.eye(3) * 0.01  # Position
        self.ekf.Q[3:6, 3:6] = np.eye(3) * 0.1   # Velocity
        self.ekf.Q[6:10, 6:10] = np.eye(4) * 0.01  # Orientation
        self.ekf.Q[10:13, 10:13] = np.eye(3) * 0.001  # Gyro bias
        self.ekf.Q[13:16, 13:16] = np.eye(3) * 0.01  # Accel bias
        
        # Measurement noise - GPS covariance
        self.ekf.R = np.eye(3) * 1.0  # 1 meter variance
        
        self.reference_gps = None
        self.last_imu_time = None
        self.dt = 0.01  # Default time step
        
    def set_reference_gps(self, lat, lon, alt):
        """Set the reference GPS position for ENU conversion"""
        if alt is None:
            alt = 0.0  # Default altitude if None
        self.reference_gps = (lat, lon, alt)
        
    def update_imu(self, gyro, accel, timestamp):
        """Update EKF with new IMU data"""
        if self.last_imu_time is None:
            self.last_imu_time = timestamp
            return
            
        dt = (timestamp - self.last_imu_time) / 1e6  # Convert microseconds to seconds
        self.last_imu_time = timestamp
        self.dt = dt
        
        # Combine gyro and accel into a single control vector (6 elements)
        u = np.concatenate([gyro, accel])
        
        # Predict step
        self.ekf.fx = lambda x: fx(x, self.dt, u)
        self.ekf.predict()
        
    def update_gps(self, lat, lon, alt):
        """Update EKF with new GPS data"""
        if self.reference_gps is None:
            return
            
        # Handle None altitude
        if alt is None:
            alt = self.reference_gps[2]  # Use reference altitude
            
        # Convert to ENU coordinates
        try:
            # Calculate ENU coordinates with bounds checking
            e = geodesic((self.reference_gps[0], self.reference_gps[1]), 
                        (self.reference_gps[0], lon)).meters
            e *= (1 if lon >= self.reference_gps[1] else -1)
            
            n = geodesic((self.reference_gps[0], self.reference_gps[1]), 
                        (lat, self.reference_gps[1])).meters
            n *= (1 if lat >= self.reference_gps[0] else -1)
            
            # Validate ENU coordinates
            if abs(e) > 1000 or abs(n) > 1000:  # 1km threshold
                print(f"Warning: Large ENU coordinates (e={e:.1f}, n={n:.1f}), skipping update")
                return
                
            u = alt - self.reference_gps[2] if alt is not None else 0
            
            z = np.array([e, n, u])
            self.ekf.update(z, HJacobian=H_jacobian, Hx=hx)
            
        except Exception as e:
            print(f"GPS update error: {e}")
            # Reset position covariance if we get consistent errors
            self.ekf.P[0:3, 0:3] = np.eye(3) * 100  # Reset position uncertainty
        
    def get_position(self):
        """Get current position in ENU coordinates"""
        return self.ekf.x[0:3]
        
    def get_gps_position(self):
        """Convert current position back to GPS coordinates"""
        if self.reference_gps is None:
            return None
            
        x, y, z = self.ekf.x[0:3]
        try:
            flat_distance = np.sqrt(x**2 + y**2)
            bearing = np.degrees(np.arctan2(x, y)) % 360  # East-North
            ref_point = Point(self.reference_gps[0], self.reference_gps[1])
            dest = geodesic(meters=flat_distance).destination(ref_point, bearing)
            alt_out = self.reference_gps[2] + z
            
            return dest.latitude, dest.longitude, alt_out
        except Exception as e:
            print(f"Position conversion error: {e}")
            return None


# --- Serial port setup ---
IMU_PORT = 'COM3'
IMU_BAUD = 115200
GPS_PORT = 'COM10'
GPS_BAUD = 115200

def parse_packet(cmd_id, payload):
    """Parse IMU packet"""
    if cmd_id == 32:  # CMD_QUATERNION
        timestamp, q1, q2, q3, q4 = struct.unpack('<Iffff', payload)
        return timestamp, {'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4}
    elif cmd_id == 35:  # CMD_RPY
        timestamp, roll, pitch, yaw = struct.unpack('<Ifff', payload)
        return timestamp, {'roll': roll, 'pitch': pitch, 'yaw': yaw}
    elif cmd_id == 41:  # CMD_RAW
        vals = struct.unpack('<Ifff fff fff', payload)
        return vals[0], {
            'accel_x': vals[1], 'accel_y': vals[2], 'accel_z': vals[3],
            'gyro_x': vals[4], 'gyro_y': vals[5], 'gyro_z': vals[6],
            'mag_x': vals[7], 'mag_y': vals[8], 'mag_z': vals[9]
        }
    return None, {}

def read_imu_packet(ser):
    """Read IMU packet from serial port"""
    while True:
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
        return timestamp, data

def read_gps_line(ser):
    line = ser.readline().decode('ascii', errors='ignore').strip()
    if not line.startswith('$'):
        return None

    try:
        msg = pynmea2.parse(line)

        if isinstance(msg, pynmea2.types.talker.GGA):
            print(f"[GGA] Time: {msg.timestamp} | Lat: {msg.latitude}° | Lon: {msg.longitude}° | Alt: {msg.altitude} {msg.altitude_units} | HDOP: {msg.horizontal_dil}")
            return {
                'type': 'GGA',
                'lat': float(msg.latitude),
                'lon': float(msg.longitude),
                'alt': float(msg.altitude) if msg.altitude else None,
                'hdop': float(msg.horizontal_dil) if msg.horizontal_dil else None,
            }

        elif isinstance(msg, pynmea2.types.talker.GST):
            time_err = msg.timestamp if msg.timestamp else None
            lat_err = float(msg.data[5]) if msg.data[5] else None
            lon_err = float(msg.data[6]) if msg.data[6] else None
            alt_err = float(msg.data[7]) if msg.data[7] else None
            print(f"[GST] Lat Error: {lat_err} m | Lon Error: {lon_err} m | Alt Error: {alt_err} m | time: {time_err}")
            return {
                'type': 'GST',
                'lat_err': lat_err,
                'lon_err': lon_err,
                'alt_err': alt_err,
            }

        elif isinstance(msg, pynmea2.types.talker.HDT):
            print(f"[HDT] Heading: {msg.heading}°")
            return None

    except pynmea2.ParseError as e:
        print(f"Parse error: {e}")

def main():
    # Initialize sensor fusion
    ekf_fusion = SensorFusionEKF()
    
    # Open serial ports
    try:
        imu_ser = serial.Serial(IMU_PORT, IMU_BAUD, timeout=0.01)
        gps_ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=0.01)
    except serial.SerialException as e:
        print(f"Failed to open serial ports: {e}")
        return
        
    # Wait for first valid GPS fix to set reference
    print("Waiting for GPS fix...")
    start_time = time.time()
    while ekf_fusion.reference_gps is None and time.time() - start_time < 30:
        gps_fix = read_gps_line(gps_ser)
        if gps_fix and gps_fix.get('type') == 'GGA':
            lat = gps_fix.get('lat')
            lon = gps_fix.get('lon')
            alt = gps_fix.get('alt')
            if lat and lon and lat != 0.0 and lon != 0.0:
                ekf_fusion.set_reference_gps(lat, lon, alt)
                print(f"Reference GPS set: {ekf_fusion.reference_gps}")

    
    if ekf_fusion.reference_gps is None:
        print("Failed to get GPS fix within timeout period")
        return
    
    # Main loop
    print("Starting sensor fusion...")
    try:
        while True:
            try:
                # Read IMU
                imu_timestamp, imu_data = read_imu_packet(imu_ser)
                if imu_data is None:
                    continue
                    
                gyro = np.array([
                    imu_data.get('gyro_x', 0),
                    imu_data.get('gyro_y', 0),
                    imu_data.get('gyro_z', 0)
                ])
                accel = np.array([
                    imu_data.get('accel_x', 0),
                    imu_data.get('accel_y', 0),
                    imu_data.get('accel_z', 0)
                ])
                ekf_fusion.update_imu(gyro, accel, imu_timestamp)
                
                # Read GPS
                gps_fix = read_gps_line(gps_ser)
                if gps_fix is not None:
                    ekf_fusion.update_gps(gps_fix['lat'], gps_fix['lon'], gps_fix['alt'])
                    
                # Output current position
                gps_pos = ekf_fusion.get_gps_position()
                if gps_pos is not None:
                    print(f"EKF GPS: lat={gps_pos[0]:.8f}, lon={gps_pos[1]:.8f}, alt={gps_pos[2]:.2f}")
                    log_position_and_error(pd.Timestamp.now(), gps_pos, gps_error=(
                        gps_fix.get('lat_err'),
                        gps_fix.get('lon_err'),
                        gps_fix.get('alt_err')
                    ) if gps_fix else None)
                    
                # Sleep to maintain loop rate
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        imu_ser.close()
        gps_ser.close()
        print("Serial ports closed")

if __name__ == "__main__":
    main()