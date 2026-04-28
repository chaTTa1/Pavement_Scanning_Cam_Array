# -*- coding: utf-8 -*-
"""
2D GPS + IMU EKF baseline equivalent to the Janudis GPS/IMU project.

State:
    x = [utm_easting_m, utm_northing_m, yaw_rad]

Control input:
    u = [forward_velocity_mps, yaw_rate_rad_s]

Measurement:
    z = [utm_easting_m, utm_northing_m] from GPS when available

This is intentionally a 2D vehicle-kinematic EKF, matching the referenced
project's function. It is not a raw 3D strapdown GNSS/INS filter.
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass

import numpy as np


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
UTM_K0 = 0.9996


def normalize_angle(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def parse_float(value):
    if value is None:
        return math.nan
    value = str(value).strip()
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def parse_timestamp_seconds(value, first_timestamp=None, scale=None):
    """Convert timestamps to seconds relative to the first timestamp."""
    t_raw = parse_float(value)
    if math.isnan(t_raw):
        return math.nan, first_timestamp, scale

    if first_timestamp is None:
        first_timestamp = t_raw
        if scale is None:
            # Heuristic: Janudis data says ns in README, but the C++ code
            # divides by 1e6. Keep a robust auto mode for both.
            if abs(t_raw) > 1e17:
                scale = 1e9
            elif abs(t_raw) > 1e14:
                scale = 1e6
            elif abs(t_raw) > 1e11:
                scale = 1e3
            else:
                scale = 1.0

    return ((t_raw - first_timestamp) / scale,
            first_timestamp, scale)


def latlon_to_utm(lat_deg, lon_deg, force_zone=None):
    """
    Convert WGS84 latitude/longitude to UTM easting/northing.

    Returns:
        easting_m, northing_m, zone_number, hemisphere
    """
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    if force_zone is None:
        zone = int((lon_deg + 180.0) / 6.0) + 1
    else:
        zone = int(force_zone)

    lon_origin_deg = (zone - 1) * 6.0 - 180.0 + 3.0
    lon_origin_rad = math.radians(lon_origin_deg)

    e2 = WGS84_E2
    ep2 = e2 / (1.0 - e2)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    tan_lat = math.tan(lat_rad)

    n = WGS84_A / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = ep2 * cos_lat * cos_lat
    a = cos_lat * (lon_rad - lon_origin_rad)

    m = WGS84_A * (
        (1.0 - e2 / 4.0 - 3.0 * e2 ** 2 / 64.0
         - 5.0 * e2 ** 3 / 256.0) * lat_rad
        - (3.0 * e2 / 8.0 + 3.0 * e2 ** 2 / 32.0
           + 45.0 * e2 ** 3 / 1024.0) * math.sin(2.0 * lat_rad)
        + (15.0 * e2 ** 2 / 256.0
           + 45.0 * e2 ** 3 / 1024.0) * math.sin(4.0 * lat_rad)
        - (35.0 * e2 ** 3 / 3072.0) * math.sin(6.0 * lat_rad)
    )

    easting = UTM_K0 * n * (
        a + (1.0 - t + c) * a ** 3 / 6.0
        + (5.0 - 18.0 * t + t ** 2 + 72.0 * c
           - 58.0 * ep2) * a ** 5 / 120.0
    ) + 500000.0

    northing = UTM_K0 * (
        m + n * tan_lat * (
            a ** 2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c ** 2)
            * a ** 4 / 24.0
            + (61.0 - 58.0 * t + t ** 2 + 600.0 * c
               - 330.0 * ep2) * a ** 6 / 720.0
        )
    )

    hemisphere = "N" if lat_deg >= 0 else "S"
    if lat_deg < 0:
        northing += 10000000.0

    return easting, northing, zone, hemisphere


@dataclass
class EKFDebug:
    pred_state: np.ndarray
    pred_P_diag: np.ndarray
    innovation: np.ndarray
    innovation_norm: float
    S_diag: np.ndarray
    K: np.ndarray
    updated: bool


class VehicleGpsImuEKF:
    def __init__(self, xy_obs_noise_std=5.0,
                 yaw_rate_noise_std=0.02,
                 forward_velocity_noise_std=0.3,
                 initial_yaw_std=math.pi):
        self.x = np.zeros(3)
        self.P = np.diag([
            xy_obs_noise_std ** 2,
            xy_obs_noise_std ** 2,
            initial_yaw_std ** 2,
        ])
        self.Q_gps = np.diag([
            xy_obs_noise_std ** 2,
            xy_obs_noise_std ** 2,
        ])
        self.R_control = np.diag([
            forward_velocity_noise_std ** 2,
            yaw_rate_noise_std ** 2,
        ])
        self.initialized = False

    def initialize(self, easting, northing, yaw_rad=0.0):
        self.x[:] = [easting, northing, normalize_angle(yaw_rad)]
        self.initialized = True

    @staticmethod
    def motion_model(x, v, omega, dt):
        px, py, yaw = x
        if abs(omega) > 1e-8:
            yaw_new = yaw + omega * dt
            px += (v / omega) * (math.sin(yaw_new) - math.sin(yaw))
            py += (v / omega) * (-math.cos(yaw_new) + math.cos(yaw))
            yaw = yaw_new
        else:
            px += v * math.cos(yaw) * dt
            py += v * math.sin(yaw) * dt
        return np.array([px, py, normalize_angle(yaw)], dtype=float)

    @staticmethod
    def jacobian_F(x, v, omega, dt):
        _, _, yaw = x
        F = np.eye(3)
        if abs(omega) > 1e-8:
            yaw_new = yaw + omega * dt
            F[0, 2] = (v / omega) * (
                math.cos(yaw_new) - math.cos(yaw))
            F[1, 2] = (v / omega) * (
                math.sin(yaw_new) - math.sin(yaw))
        else:
            F[0, 2] = -v * math.sin(yaw) * dt
            F[1, 2] = v * math.cos(yaw) * dt
        return F

    @staticmethod
    def jacobian_control(x, v, omega, dt):
        _, _, yaw = x
        V = np.zeros((3, 2))
        if abs(omega) > 1e-8:
            yaw_new = yaw + omega * dt
            V[0, 0] = (math.sin(yaw_new) - math.sin(yaw)) / omega
            V[1, 0] = (-math.cos(yaw_new) + math.cos(yaw)) / omega
            V[2, 0] = 0.0

            V[0, 1] = (
                v * (omega * dt * math.cos(yaw_new)
                     - math.sin(yaw_new) + math.sin(yaw))
                / (omega ** 2)
            )
            V[1, 1] = (
                v * (omega * dt * math.sin(yaw_new)
                     + math.cos(yaw_new) - math.cos(yaw))
                / (omega ** 2)
            )
            V[2, 1] = dt
        else:
            V[0, 0] = math.cos(yaw) * dt
            V[1, 0] = math.sin(yaw) * dt
            V[2, 0] = 0.0
            V[0, 1] = -0.5 * v * math.sin(yaw) * dt ** 2
            V[1, 1] = 0.5 * v * math.cos(yaw) * dt ** 2
            V[2, 1] = dt
        return V

    def propagate(self, forward_velocity_mps, yaw_rate_rad_s, dt):
        dt = max(float(dt), 0.0)
        v = float(forward_velocity_mps)
        omega = float(yaw_rate_rad_s)

        F = self.jacobian_F(self.x, v, omega, dt)
        V = self.jacobian_control(self.x, v, omega, dt)
        self.x = self.motion_model(self.x, v, omega, dt)
        self.P = F @ self.P @ F.T + V @ self.R_control @ V.T
        self.P = 0.5 * (self.P + self.P.T)
        return F, V

    def update_gps(self, easting, northing):
        z = np.array([easting, northing], dtype=float)
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        hx = self.x[:2].copy()
        y = z - hx
        S = H @ self.P @ H.T + self.Q_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = normalize_angle(self.x[2])
        I = np.eye(3)
        # Match the Janudis style while keeping symmetry after the update.
        self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return y, S, K


def get_by_index(row, idx):
    if idx is None or idx < 0 or idx >= len(row):
        return ""
    return row[idx]


def read_input_rows(path, args):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            yield line_number, header, row


def run_filter(args):
    ekf = VehicleGpsImuEKF(
        xy_obs_noise_std=args.xy_obs_noise_std,
        yaw_rate_noise_std=args.yaw_rate_noise_std,
        forward_velocity_noise_std=args.forward_velocity_noise_std,
        initial_yaw_std=args.initial_yaw_std,
    )

    output_rows = []
    debug_rows = []

    first_timestamp = None
    timestamp_scale = args.timestamp_scale
    last_t = None
    last_speed = 0.0
    last_yaw_rate = 0.0
    last_yaw = args.initial_yaw_rad
    force_zone = args.utm_zone
    utm_zone = None
    utm_hemisphere = None

    for step, (line_number, header, row) in enumerate(
            read_input_rows(args.input_csv, args)):
        t, first_timestamp, timestamp_scale = parse_timestamp_seconds(
            get_by_index(row, args.timestamp_col),
            first_timestamp,
            timestamp_scale,
        )
        if math.isnan(t):
            continue

        dt = 0.0 if last_t is None else t - last_t
        if dt < 0.0:
            dt = 0.0

        lat = parse_float(get_by_index(row, args.lat_col))
        lon = parse_float(get_by_index(row, args.lon_col))
        yaw_value = parse_float(get_by_index(row, args.yaw_col))
        speed = parse_float(get_by_index(row, args.speed_col))
        yaw_rate = parse_float(get_by_index(row, args.yaw_rate_col))

        gps_valid = not math.isnan(lat) and not math.isnan(lon)
        if not math.isnan(speed):
            last_speed = speed
        if not math.isnan(yaw_rate):
            last_yaw_rate = yaw_rate
        if not math.isnan(yaw_value):
            last_yaw = (math.radians(yaw_value)
                        if args.yaw_degrees else yaw_value)

        z_easting = math.nan
        z_northing = math.nan
        if gps_valid:
            z_easting, z_northing, zone, hemisphere = latlon_to_utm(
                lat, lon, force_zone=force_zone)
            if force_zone is None:
                force_zone = zone
            utm_zone = zone
            utm_hemisphere = hemisphere

        if not ekf.initialized:
            if not gps_valid:
                continue
            ekf.initialize(z_easting, z_northing, last_yaw)
            last_t = t
            output_rows.append([
                t, lat, lon, z_easting, z_northing, last_yaw,
                ekf.x[0], ekf.x[1], ekf.x[2],
            ])
            debug_rows.append(make_debug_row(
                step, line_number, t, 0.0, "init", gps_valid,
                lat, lon, z_easting, z_northing,
                last_speed, last_yaw_rate, ekf.x, ekf.P,
                np.zeros(3), ekf.P, np.zeros(2), 0.0,
                np.zeros(2), np.zeros((3, 2)), utm_zone,
                utm_hemisphere))
            continue

        pred_state = ekf.x.copy()
        pred_P = ekf.P.copy()
        ekf.propagate(last_speed, last_yaw_rate, dt)

        event = "predict"
        innovation = np.zeros(2)
        S = np.zeros((2, 2))
        K = np.zeros((3, 2))
        innovation_norm = 0.0

        if gps_valid:
            event = "predict+gps_update"
            innovation, S, K = ekf.update_gps(z_easting, z_northing)
            innovation_norm = float(np.linalg.norm(innovation))

        output_rows.append([
            t, lat if gps_valid else "",
            lon if gps_valid else "",
            z_easting if gps_valid else "",
            z_northing if gps_valid else "",
            last_yaw,
            ekf.x[0], ekf.x[1], ekf.x[2],
        ])
        debug_rows.append(make_debug_row(
            step, line_number, t, dt, event, gps_valid, lat, lon,
            z_easting, z_northing, last_speed, last_yaw_rate,
            ekf.x, ekf.P, pred_state, pred_P, innovation,
            innovation_norm, S.diagonal() if S.size else np.zeros(2),
            K, utm_zone, utm_hemisphere))

        last_t = t

    write_outputs(args.output_csv, args.debug_csv, output_rows, debug_rows)


def make_debug_row(step, line_number, timestamp_s, dt_s, event, gps_valid,
                   lat, lon, z_easting, z_northing, speed, yaw_rate,
                   state, P, pred_state, pred_P, innovation,
                   innovation_norm, S_diag, K, utm_zone, utm_hemisphere):
    return {
        "step": step,
        "line_number": line_number,
        "timestamp_s": f"{timestamp_s:.9f}",
        "dt_s": f"{dt_s:.9f}",
        "event": event,
        "gps_valid": int(gps_valid),
        "lat": "" if math.isnan(lat) else f"{lat:.12f}",
        "lon": "" if math.isnan(lon) else f"{lon:.12f}",
        "utm_easting_meas": (
            "" if math.isnan(z_easting) else f"{z_easting:.6f}"),
        "utm_northing_meas": (
            "" if math.isnan(z_northing) else f"{z_northing:.6f}"),
        "forward_velocity_mps": f"{speed:.6f}",
        "yaw_rate_rad_s": f"{yaw_rate:.9f}",
        "pred_x": f"{pred_state[0]:.6f}",
        "pred_y": f"{pred_state[1]:.6f}",
        "pred_yaw_rad": f"{pred_state[2]:.9f}",
        "state_x": f"{state[0]:.6f}",
        "state_y": f"{state[1]:.6f}",
        "state_yaw_rad": f"{state[2]:.9f}",
        "state_yaw_deg": f"{math.degrees(state[2]):.6f}",
        "innovation_x_m": f"{innovation[0]:.6f}",
        "innovation_y_m": f"{innovation[1]:.6f}",
        "innovation_norm_m": f"{innovation_norm:.6f}",
        "P_x": f"{P[0, 0]:.9f}",
        "P_y": f"{P[1, 1]:.9f}",
        "P_yaw": f"{P[2, 2]:.9f}",
        "pred_P_x": f"{pred_P[0, 0]:.9f}",
        "pred_P_y": f"{pred_P[1, 1]:.9f}",
        "pred_P_yaw": f"{pred_P[2, 2]:.9f}",
        "S_x": f"{S_diag[0]:.9f}",
        "S_y": f"{S_diag[1]:.9f}",
        "K_x_x": f"{K[0, 0]:.9f}",
        "K_x_y": f"{K[0, 1]:.9f}",
        "K_y_x": f"{K[1, 0]:.9f}",
        "K_y_y": f"{K[1, 1]:.9f}",
        "K_yaw_x": f"{K[2, 0]:.9f}",
        "K_yaw_y": f"{K[2, 1]:.9f}",
        "utm_zone": "" if utm_zone is None else utm_zone,
        "utm_hemisphere": "" if utm_hemisphere is None else utm_hemisphere,
    }


def write_outputs(output_csv, debug_csv, output_rows, debug_rows):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_s", "gps_lat", "gps_lon",
            "gps_utm_easting", "gps_utm_northing", "input_yaw_rad",
            "state_x", "state_y", "state_yaw_rad",
        ])
        writer.writerows(output_rows)

    os.makedirs(os.path.dirname(debug_csv) or ".", exist_ok=True)
    with open(debug_csv, "w", newline="") as f:
        if debug_rows:
            writer = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
            writer.writeheader()
            writer.writerows(debug_rows)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="2D GPS+IMU EKF baseline with debug logging")
    parser.add_argument("input_csv",
                        help="Input CSV similar to localization_log2.csv")
    parser.add_argument("--output-csv",
                        default="output_utm_python.csv")
    parser.add_argument("--debug-csv",
                        default="debug_ekf_python.csv")

    parser.add_argument("--timestamp-col", type=int, default=0)
    parser.add_argument("--lat-col", type=int, default=1)
    parser.add_argument("--lon-col", type=int, default=2)
    parser.add_argument("--speed-col", type=int, default=4)
    parser.add_argument("--yaw-rate-col", type=int, default=5)
    parser.add_argument("--yaw-col", type=int, default=-1,
                        help="Optional yaw column. -1 disables it.")

    parser.add_argument("--timestamp-scale", type=float, default=None,
                        help=("Timestamp units per second. Auto if omitted; "
                              "use 1e9 for ns, 1e6 for us, 1e3 for ms."))
    parser.add_argument("--yaw-degrees", action="store_true",
                        help="Interpret --yaw-col as degrees.")
    parser.add_argument("--initial-yaw-rad", type=float, default=0.0)
    parser.add_argument("--utm-zone", type=int, default=None,
                        help="Force UTM zone. Defaults to first valid GPS.")

    parser.add_argument("--xy-obs-noise-std", type=float, default=5.0)
    parser.add_argument("--yaw-rate-noise-std", type=float, default=0.02)
    parser.add_argument("--forward-velocity-noise-std",
                        type=float, default=0.3)
    parser.add_argument("--initial-yaw-std", type=float, default=math.pi)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_filter(args)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.debug_csv}")


if __name__ == "__main__":
    main()
