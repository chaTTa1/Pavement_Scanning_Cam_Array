import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from scipy.spatial.transform import Rotation as R
import serial
import pynmea2
import struct
import time
from geopy.distance import geodesic
from geopy.point import Point



def correct_yaw_from_mag(quat, mag_vector, correction_strength=0.01):
    """
    Corrects yaw based on magnetometer heading.

    quat: current orientation quaternion (x, y, z, w)
    mag_vector: 3D magnetometer vector (x, y, z)
    correction_strength: small gain to apply correction
    """
    # Normalize magnetometer vector
    mag_vector = mag_vector / np.linalg.norm(mag_vector)

    # Estimated yaw from quaternion
    r = R.from_quat(quat)
    yaw_imu = r.as_euler('zyx')[0]  # in radians

    # Heading from magnetometer (horizontal projection)
    mag_heading = np.arctan2(mag_vector[1], mag_vector[0])  # yaw from X/Y

    # Compute error and apply small corrective yaw
    yaw_error = mag_heading - yaw_imu
    yaw_correction = R.from_euler('z', correction_strength * yaw_error)
    corrected = yaw_correction * r

    return corrected.as_quat()


def quaternion_propagate(q, omega, dt):
    """
    Propagate quaternion with angular velocity omega over dt
    """
    r = R.from_quat(q)
    delta_r = R.from_rotvec(omega * dt)
    new_r = r * delta_r
    return new_r.as_quat()


def fx(x, u, dt):
    """
    State transition function
    x: state vector [pos, vel, quat, gyro_bias, acc_bias]
    u: control input [gyro_raw, acc_raw]
    """
    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    b_g = x[10:13]
    b_a = x[13:16]

    gyro = u[0] - b_g
    acc = u[1] - b_a

    quat = quat / np.linalg.norm(quat)
    r = R.from_quat(quat)
    acc_world = r.apply(acc)
    acc_world[2] -= 9.81  # remove gravity

    new_pos = pos + vel * dt + 0.5 * acc_world * dt**2
    new_vel = vel + acc_world * dt
    new_quat = quaternion_propagate(quat, gyro, dt)

    # Bias random walk (allow filter to adapt)
    new_b_g = b_g  # Optionally: + np.random.normal(0, 0.001, 3) * dt
    new_b_a = b_a  # Optionally: + np.random.normal(0, 0.01, 3) * dt

    x_new = np.zeros(16)
    x_new[0:3] = new_pos
    x_new[3:6] = new_vel
    x_new[6:10] = new_quat
    x_new[10:13] = new_b_g
    x_new[13:16] = new_b_a
    return x_new


def hx(x):
    return x[0:3]  # GPS measures position only


def H_jacobian(x):
    H = np.zeros((3, 16))
    H[:, :3] = np.eye(3)
    return H


# Initialize EKF
ekf = ExtendedKalmanFilter(dim_x=16, dim_z=3)
ekf.x = np.zeros(16)
ekf.x[6] = 1.0  # Initialize quaternion to identity
ekf.P = np.eye(16) * 100
ekf.Q = np.eye(16) * 10.0
ekf.Q[10:13, 10:13] = np.eye(3) * 0.01  # gyro bias
ekf.Q[13:16, 13:16] = np.eye(3) * 2.0  # or even higher
ekf.R = np.diag([0.01**2, 0.01**2, 0.02**2])  # RTK GPS accuracy


# --- Serial port setup ---
IMU_PORT = 'COM3'
IMU_BAUD = 115200
GPS_PORT = 'COM10'
GPS_BAUD = 115200

imu_ser = serial.Serial(IMU_PORT, IMU_BAUD, timeout=0.01)
gps_ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=0.01)

last_imu_time = None

# --- IMU packet parsing (copied from IMU_to_csv.py) ---
CMD_QUATERNION = 32
CMD_RPY = 35
CMD_RAW = 41

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

def read_imu_packet(ser):
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
    """
    Parse GPS data exactly as in Real_Time_Mosaic_Parser.py.
    Returns (lat, lon, alt) if GGA sentence, else None.
    """
    line = ser.readline().decode('ascii', errors='ignore').strip()
    if not line.startswith('$'):
        return None
    try:
        msg = pynmea2.parse(line)

        if isinstance(msg, pynmea2.types.talker.GGA):
            # For debugging, you can uncomment the next line:
            # print(f"[GGA] TOW: {msg.timestamp} | Lat: {msg.latitude}° | Lon: {msg.longitude}° | Alt: {msg.altitude} {msg.altitude_units}")
            return float(msg.latitude), float(msg.longitude), float(msg.altitude)

        elif isinstance(msg, pynmea2.types.talker.HDT):
            # print(f"[HDT] Heading: {msg.heading}°")
            return None

        elif isinstance(msg, pynmea2.types.talker.RMC):
            # print(f"[RMC] Time: {msg.timestamp}, Date: {msg.datestamp}")
            return None

    except pynmea2.ParseError:
        return None

reference_gps = None
reference_gps_samples = []
reference_gps_time = None
REFERENCE_GPS_SAMPLE_COUNT = 10
REFERENCE_GPS_STATIONARY_SECONDS = 2.0
REFERENCE_GPS_POSITION_TOL = 0.00001  # about 1 meter

# --- Main fusion loop ---
while True:
    loop_start = time.perf_counter()

    # --- Read IMU ---
    imu_packet = read_imu_packet(imu_ser)
    if imu_packet is None:
        continue
    imu_timestamp, imu_data = imu_packet

    # Compose IMU input for EKF (match field names from IMU_to_csv.py)
    gyro = np.array([
        imu_data.get('gyro_x', 0),
        imu_data.get('gyro_y', 0),
        imu_data.get('gyro_z', 0)
    ])
    acc = np.array([
        imu_data.get('accel_x', 0),
        imu_data.get('accel_y', 0),
        imu_data.get('accel_z', 0)
    ])
    u = [gyro, acc]

    # Time step
    if last_imu_time is None:
        last_imu_time = imu_timestamp
        continue
    dt = (imu_timestamp - last_imu_time) / 1e6  # microseconds to seconds
    last_imu_time = imu_timestamp

    # EKF predict
    predict_start = time.perf_counter()
    ekf.x = fx(ekf.x, u, dt)  # <--- update state using your fx function
    predict_end = time.perf_counter()

    # --- Try GPS update (non-blocking) ---
    gps_fix = read_gps_line(gps_ser)
    update_time = None
    if gps_fix is not None:
        lat, lon, alt = gps_fix

        # Reference GPS logic
        if reference_gps is None:
            now = time.time()
            if reference_gps_time is None:
                reference_gps_time = now
                reference_gps_samples = [(lat, lon, alt)]
            else:
                reference_gps_samples.append((lat, lon, alt))
                print(f"Collecting reference samples: {len(reference_gps_samples)}")
                # Check if stationary
                if len(reference_gps_samples) >= REFERENCE_GPS_SAMPLE_COUNT:
                    lats = [s[0] for s in reference_gps_samples]
                    lons = [s[1] for s in reference_gps_samples]
                    alts = [s[2] for s in reference_gps_samples]
                    if (max(lats) - min(lats) < REFERENCE_GPS_POSITION_TOL and
                        max(lons) - min(lons) < REFERENCE_GPS_POSITION_TOL and
                        max(alts) - min(alts) < 2):  # 2 meters altitude tolerance
                        if now - reference_gps_time >= REFERENCE_GPS_STATIONARY_SECONDS:
                            # Use mean as reference
                            reference_gps = (
                                sum(lats) / len(lats),
                                sum(lons) / len(lons),
                                sum(alts) / len(alts)
                            )
                            print(f"Reference GPS set: {reference_gps}")
                    else:
                        # Movement detected, reset
                        reference_gps_time = now
                        reference_gps_samples = [(lat, lon, alt)]
            continue  # Don't update EKF until reference is set

        # Convert lat/lon/alt to local ENU offset from reference
        ref_point = Point(reference_gps[0], reference_gps[1])
        # Approximate ENU (East, North, Up)
        e = geodesic((reference_gps[0], reference_gps[1]), (reference_gps[0], lon)).meters * (1 if lon >= reference_gps[1] else -1)
        n = geodesic((reference_gps[0], reference_gps[1]), (lat, reference_gps[1])).meters * (1 if lat >= reference_gps[0] else -1)
        u_ = alt - reference_gps[2]
        z = np.array([e, n, u_])
        update_start = time.perf_counter()
        ekf.update(z=z, HJacobian=H_jacobian, Hx=hx)
        update_end = time.perf_counter()
        update_time = update_end - update_start

        # Debugging: Print innovation and Kalman Gain
        print("Innovation:", z - hx(ekf.x))
        print("Kalman Gain (diag):", np.diag(ekf.K))
        print("GPS update applied:", z)

        print("Reference GPS:", reference_gps)
        print("ENU from GPS:", e, n, u_)

    # --- Output current state as GPS ---
    if reference_gps is not None:
        x, y, z_ = ekf.x[0:3]  # EKF position in meters (ENU)
        flat_distance = np.sqrt(x**2 + y**2)
        bearing = np.degrees(np.arctan2(x, y)) % 360  # East-North
        ref_point = Point(reference_gps[0], reference_gps[1])
        dest = geodesic(meters=flat_distance).destination(ref_point, bearing)
        alt_out = reference_gps[2] + z_
        print(f"EKF GPS: lat={dest.latitude:.8f}, lon={dest.longitude:.8f}, alt={alt_out:.2f}")

    # Remove timing printouts
    # loop_end = time.perf_counter()
    # loop_time = loop_end - loop_start
    # predict_time = predict_end - predict_start
    # if update_time is not None:
    #     print(f"Loop: {loop_time1000:.2f} ms | Predict: {predict_time1000:.2f} ms | Update: {update_time*1000:.2f} ms")
    # else:
    #     print(f"Loop: {loop_time1000:.2f} ms | Predict: {predict_time1000:.2f} ms")

    # Sleep to avoid busy loop (optional)
    time.sleep(0.001)