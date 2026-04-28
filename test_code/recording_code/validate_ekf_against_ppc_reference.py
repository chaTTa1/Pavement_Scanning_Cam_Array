"""Compare EKF_only.py output with a PPC Tokyo reference trajectory."""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
SPYDER_RUN_DIR = SCRIPT_DIR / "tokyo" / "run1"
SPYDER_EKF_OUTPUT_DIR = None  # None = auto-detect from run folder

M_PER_DEG_LAT = 111_132.92
M_PER_DEG_LON_EQ = 111_319.49


def m_per_deg_lon(lat_deg: float) -> float:
    return M_PER_DEG_LON_EQ * math.cos(math.radians(lat_deg))


def clean_row(row: Dict[str, str]) -> Dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [clean_row(row) for row in csv.DictReader(handle)]


def f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def interp_reference(
    ref_rows: List[Dict[str, str]],
    ref_tows: List[float],
    tow: float,
) -> Optional[Dict[str, float]]:
    idx = bisect_right(ref_tows, tow)
    if idx == 0 or idx >= len(ref_rows):
        return None

    left = ref_rows[idx - 1]
    right = ref_rows[idx]
    t0 = ref_tows[idx - 1]
    t1 = ref_tows[idx]
    if t1 <= t0:
        return None
    a = (tow - t0) / (t1 - t0)

    def lerp(key: str) -> float:
        return f(left, key) + a * (f(right, key) - f(left, key))

    heading0 = f(left, "Heading (deg)")
    heading_delta = angle_diff_deg(f(right, "Heading (deg)"), heading0)

    return {
        "lat": lerp("Latitude (deg)"),
        "lon": lerp("Longitude (deg)"),
        "alt": lerp("Ellipsoid Height (m)"),
        "ve": lerp("East Velocity (m/s)"),
        "vn": lerp("North Velocity (m/s)"),
        "vu": lerp("Up Velocity (m/s)"),
        "heading": (heading0 + a * heading_delta) % 360.0,
    }


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def rms(values: List[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else math.nan


def default_output_dir(run_dir: Path) -> Path:
    candidates = [
        run_dir / "ekf_output_30s",
        run_dir / "ekf_output_full",
        run_dir / "ekf_output",
        SCRIPT_DIR / "ekf_output",
    ]
    for candidate in candidates:
        if (candidate / "position_ekf.csv").exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate EKF_only.py position_ekf.csv against PPC reference.csv.")
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=SPYDER_RUN_DIR,
        help="PPC run folder containing reference.csv.")
    parser.add_argument(
        "ekf_output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Folder containing position_ekf.csv from EKF_only.py.")
    parser.add_argument(
        "--start-tow",
        type=float,
        default=None,
        help="GPS TOW corresponding to EKF wall_time=0. Default: first imu.csv row.")
    parser.add_argument(
        "--sample-step",
        type=int,
        default=10,
        help="Use every Nth EKF row to keep validation fast. Default: 10.")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring non-validator command-line args: {unknown}")
    if args.ekf_output_dir is None:
        args.ekf_output_dir = (
            Path(SPYDER_EKF_OUTPUT_DIR)
            if SPYDER_EKF_OUTPUT_DIR
            else default_output_dir(args.run_dir)
        )
    return args


def main() -> None:
    args = parse_args()
    reference_path = args.run_dir / "reference.csv"
    ekf_path = args.ekf_output_dir / "position_ekf.csv"
    imu_path = args.run_dir / "imu.csv"

    print("PPC/EKF validation")
    print(f"Run dir:        {args.run_dir}")
    print(f"EKF output dir: {args.ekf_output_dir}")

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    if not ekf_path.exists():
        raise FileNotFoundError(f"EKF output file not found: {ekf_path}")

    ref_rows = read_rows(reference_path)
    ekf_rows = read_rows(ekf_path)
    if args.start_tow is None:
        if not imu_path.exists():
            raise FileNotFoundError("Provide --start-tow when imu.csv is unavailable.")
        start_tow = f(read_rows(imu_path)[0], "GPS TOW (s)")
    else:
        start_tow = args.start_tow

    ref_tows = [f(row, "GPS TOW (s)") for row in ref_rows]
    ref_lat0 = f(ref_rows[0], "Latitude (deg)")

    horizontal_errors = []
    vertical_errors = []
    velocity_errors = []
    heading_errors = []

    for index, row in enumerate(ekf_rows):
        if index % max(args.sample_step, 1) != 0:
            continue

        tow = start_tow + f(row, "wall_time")
        ref = interp_reference(ref_rows, ref_tows, tow)
        if ref is None:
            continue

        de = (f(row, "lon") - ref["lon"]) * m_per_deg_lon(ref_lat0)
        dn = (f(row, "lat") - ref["lat"]) * M_PER_DEG_LAT
        du = f(row, "alt") - ref["alt"]
        horizontal_errors.append(math.sqrt(de * de + dn * dn))
        vertical_errors.append(abs(du))

        dve = f(row, "ve") - ref["ve"]
        dvn = f(row, "vn") - ref["vn"]
        velocity_errors.append(math.sqrt(dve * dve + dvn * dvn))

        if "yaw" in row and row["yaw"]:
            heading_errors.append(abs(angle_diff_deg(f(row, "yaw"), ref["heading"])))

    if not horizontal_errors:
        raise RuntimeError("No overlapping EKF/reference samples were found.")

    print(f"Compared samples:       {len(horizontal_errors)}")
    print(f"Horizontal RMS error:   {rms(horizontal_errors):.3f} m")
    print(f"Horizontal mean error:  {sum(horizontal_errors) / len(horizontal_errors):.3f} m")
    print(f"Horizontal 95% error:   {percentile(horizontal_errors, 95):.3f} m")
    print(f"Horizontal max error:   {max(horizontal_errors):.3f} m")
    print(f"Vertical RMS error:     {rms(vertical_errors):.3f} m")
    print(f"Velocity RMS error:     {rms(velocity_errors):.3f} m/s")
    if heading_errors:
        print(f"Heading RMS error:      {rms(heading_errors):.3f} deg")


if __name__ == "__main__":
    main()
