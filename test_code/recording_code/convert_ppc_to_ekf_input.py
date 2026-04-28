"""Convert PPC Tokyo dataset runs into EKF_only.py input CSVs.

The PPC Tokyo folders contain:
  - imu.csv: 100 Hz IMU samples in vehicle body axes
  - reference.csv: Applanix reference trajectory/attitude at 5 Hz

EKF_only.py expects the CSV layout produced by GPS_IMU_data_recording.py:
imu_raw.csv, imu_gravity.csv, imu_quat.csv, and gps_raw.csv. This converter
creates those files so the existing EKF can be validated on a PPC run.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


G_MPS2 = 9.80665


IMU_COLUMNS = [
    "wall_time",
    "imu_timestamp_us",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
]


GRAVITY_COLUMNS = [
    "wall_time",
    "imu_timestamp_us",
    "gravity_x_g",
    "gravity_y_g",
    "gravity_z_g",
]


QUAT_COLUMNS = [
    "wall_time",
    "imu_timestamp_us",
    "q1",
    "q2",
    "q3",
    "q4",
]


GPS_COLUMNS = [
    "wall_time",
    "gps_utc",
    "msg_type",
    "lat",
    "lon",
    "alt",
    "fix_quality",
    "num_sats",
    "hdop",
    "lat_err_m",
    "lon_err_m",
    "alt_err_m",
    "heading_deg",
    "speed_knots",
    "speed_kmh",
    "course_true_deg",
    "rmc_status",
]


def clean_row(row: Dict[str, str]) -> Dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [clean_row(row) for row in csv.DictReader(handle)]


def f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def in_window(tow: float, start_tow: float, end_tow: Optional[float]) -> bool:
    return tow >= start_tow and (end_tow is None or tow <= end_tow)


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def quat_xyzw_from_matrix(m: List[List[float]]) -> Tuple[float, float, float, float]:
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return qx / norm, qy / norm, qz / norm, qw / norm


def body_to_enu_quat_xyzw(
    roll_deg: float,
    pitch_deg: float,
    heading_deg: float,
) -> Tuple[float, float, float, float]:
    """Return scipy-style [x, y, z, w] quaternion for PPC body axes to ENU.

    PPC body axes are X forward, Y right, Z down. Heading is clockwise from
    north. The intermediate navigation frame is NED, then converted to ENU.
    """
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(heading_deg)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]

    ned_from_body = matmul(matmul(rz, ry), rx)
    enu_from_ned = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    enu_from_body = matmul(enu_from_ned, ned_from_body)
    return quat_xyzw_from_matrix(enu_from_body)


def gravity_body_g(
    roll_deg: float,
    pitch_deg: float,
) -> Tuple[float, float, float]:
    """Approximate body-frame gravity vector in g for PPC body axes.

    This matches the PPC convention where a stationary level IMU reports about
    +1 g on body Z because Z points downward.
    """
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    return (
        math.sin(pitch),
        math.sin(roll) * math.cos(pitch),
        math.cos(roll) * math.cos(pitch),
    )


def gps_course_deg(ve: float, vn: float) -> float:
    return math.degrees(math.atan2(ve, vn)) % 360.0


def write_imu_raw(
    imu_rows: List[Dict[str, str]],
    output_dir: Path,
    start_tow: float,
    end_tow: Optional[float],
) -> int:
    count = 0
    with (output_dir / "imu_raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMU_COLUMNS)
        writer.writeheader()
        for row in imu_rows:
            tow = f(row, "GPS TOW (s)")
            if not in_window(tow, start_tow, end_tow):
                continue
            rel_t = tow - start_tow
            writer.writerow(
                {
                    "wall_time": f"{rel_t:.6f}",
                    "imu_timestamp_us": str(int(round(rel_t * 1_000_000.0))),
                    "gyro_x_rad_s": f"{math.radians(f(row, 'Ang Rate X (deg/s)')):.12g}",
                    "gyro_y_rad_s": f"{math.radians(f(row, 'Ang Rate Y (deg/s)')):.12g}",
                    "gyro_z_rad_s": f"{math.radians(f(row, 'Ang Rate Z (deg/s)')):.12g}",
                    "accel_x_g": f"{f(row, 'Acc X (m/s^2)') / G_MPS2:.12g}",
                    "accel_y_g": f"{f(row, 'Acc Y (m/s^2)') / G_MPS2:.12g}",
                    "accel_z_g": f"{f(row, 'Acc Z (m/s^2)') / G_MPS2:.12g}",
                }
            )
            count += 1
    return count


def write_reference_derived_files(
    reference_rows: List[Dict[str, str]],
    output_dir: Path,
    start_tow: float,
    end_tow: Optional[float],
    gps_sigma_m: float,
    heading_sigma_deg: float,
) -> Tuple[int, int, int]:
    gravity_count = 0
    quat_count = 0
    gps_count = 0

    with (output_dir / "imu_gravity.csv").open("w", newline="") as gravity_handle, \
            (output_dir / "imu_quat.csv").open("w", newline="") as quat_handle, \
            (output_dir / "gps_raw.csv").open("w", newline="") as gps_handle:
        gravity_writer = csv.DictWriter(gravity_handle, fieldnames=GRAVITY_COLUMNS)
        quat_writer = csv.DictWriter(quat_handle, fieldnames=QUAT_COLUMNS)
        gps_writer = csv.DictWriter(gps_handle, fieldnames=GPS_COLUMNS)
        gravity_writer.writeheader()
        quat_writer.writeheader()
        gps_writer.writeheader()

        for row in reference_rows:
            tow = f(row, "GPS TOW (s)")
            if not in_window(tow, start_tow, end_tow):
                continue

            rel_t = tow - start_tow
            timestamp_us = str(int(round(rel_t * 1_000_000.0)))
            roll = f(row, "Roll (deg)")
            pitch = f(row, "Pitch (deg)")
            heading = f(row, "Heading (deg)")
            gx, gy, gz = gravity_body_g(roll, pitch)
            qx, qy, qz, qw = body_to_enu_quat_xyzw(roll, pitch, heading)

            gravity_writer.writerow(
                {
                    "wall_time": f"{rel_t:.6f}",
                    "imu_timestamp_us": timestamp_us,
                    "gravity_x_g": f"{gx:.12g}",
                    "gravity_y_g": f"{gy:.12g}",
                    "gravity_z_g": f"{gz:.12g}",
                }
            )
            gravity_count += 1

            quat_writer.writerow(
                {
                    "wall_time": f"{rel_t:.6f}",
                    "imu_timestamp_us": timestamp_us,
                    "q1": f"{qw:.12g}",
                    "q2": f"{qx:.12g}",
                    "q3": f"{qy:.12g}",
                    "q4": f"{qz:.12g}",
                }
            )
            quat_count += 1

            lat = f(row, "Latitude (deg)")
            lon = f(row, "Longitude (deg)")
            alt = f(row, "Ellipsoid Height (m)")
            ve = f(row, "East Velocity (m/s)")
            vn = f(row, "North Velocity (m/s)")
            speed_ms = math.sqrt(ve * ve + vn * vn)
            course = gps_course_deg(ve, vn)
            base = {
                "wall_time": f"{rel_t:.6f}",
                "gps_utc": "",
                "lat": "",
                "lon": "",
                "alt": "",
                "fix_quality": "",
                "num_sats": "",
                "hdop": "",
                "lat_err_m": "",
                "lon_err_m": "",
                "alt_err_m": "",
                "heading_deg": "",
                "speed_knots": "",
                "speed_kmh": "",
                "course_true_deg": "",
                "rmc_status": "",
            }

            gps_writer.writerow(
                {
                    **base,
                    "msg_type": "GGA",
                    "lat": f"{lat:.10f}",
                    "lon": f"{lon:.10f}",
                    "alt": f"{alt:.4f}",
                    "fix_quality": "4",
                    "lat_err_m": f"{gps_sigma_m:.4f}",
                    "lon_err_m": f"{gps_sigma_m:.4f}",
                    "alt_err_m": f"{gps_sigma_m * 1.5:.4f}",
                }
            )
            gps_writer.writerow(
                {
                    **base,
                    "msg_type": "RMC",
                    "speed_knots": f"{speed_ms / 0.514444:.8f}",
                    "course_true_deg": f"{course:.8f}",
                    "rmc_status": "A",
                }
            )
            gps_writer.writerow(
                {
                    **base,
                    "msg_type": "HDT",
                    "heading_deg": f"{heading:.8f}",
                    "lat_err_m": "",
                    "lon_err_m": "",
                    "alt_err_m": "",
                }
            )
            gps_count += 3

    # The heading noise is configured in EKF_only.py. Keep this argument visible
    # in the CLI so runs record the intended assumption in command history.
    _ = heading_sigma_deg
    return gravity_count, quat_count, gps_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PPC Tokyo run into recorder-format EKF CSV files.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="PPC run folder containing imu.csv and reference.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Default: <run_dir>/ekf_input.")
    parser.add_argument(
        "--start-tow",
        type=float,
        default=None,
        help="GPS TOW start time. Default: first IMU sample.")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional duration in seconds for a quick validation slice.")
    parser.add_argument(
        "--gps-sigma-m",
        type=float,
        default=0.05,
        help="1-sigma position noise assigned to reference-derived GGA rows.")
    parser.add_argument(
        "--heading-sigma-deg",
        type=float,
        default=1.0,
        help="Document the heading noise assumption used by EKF_only.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "ekf_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    imu_path = run_dir / "imu.csv"
    reference_path = run_dir / "reference.csv"
    if not imu_path.exists() or not reference_path.exists():
        raise FileNotFoundError("run_dir must contain imu.csv and reference.csv")

    imu_rows = read_rows(imu_path)
    reference_rows = read_rows(reference_path)
    if not imu_rows or not reference_rows:
        raise RuntimeError("imu.csv and reference.csv must not be empty")

    start_tow = args.start_tow
    if start_tow is None:
        start_tow = f(imu_rows[0], "GPS TOW (s)")
    end_tow = None
    if args.duration_s is not None:
        end_tow = start_tow + args.duration_s

    imu_count = write_imu_raw(imu_rows, output_dir, start_tow, end_tow)
    gravity_count, quat_count, gps_count = write_reference_derived_files(
        reference_rows,
        output_dir,
        start_tow,
        end_tow,
        args.gps_sigma_m,
        args.heading_sigma_deg,
    )

    print(f"Converted PPC run: {run_dir}")
    print(f"Output folder:     {output_dir}")
    print(f"Time window:       {start_tow:.2f} to {end_tow if end_tow else 'end'} GPS TOW")
    print(f"imu_raw.csv:       {imu_count} rows")
    print(f"imu_gravity.csv:   {gravity_count} rows")
    print(f"imu_quat.csv:      {quat_count} rows")
    print(f"gps_raw.csv:       {gps_count} rows")


if __name__ == "__main__":
    main()
