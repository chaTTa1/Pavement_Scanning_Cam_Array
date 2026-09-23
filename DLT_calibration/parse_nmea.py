# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:49:16 2026

@author: alexc
"""

"""Parse NMEA logs into the CSV format expected by coordinate_calculation.py.

Produces three CSV files:
    gps_main.csv               (from the main-antenna NMEA log)
    gps_imu_antenna.csv        (from the auxiliary-antenna NMEA log)
    camera_antenna_point.csv   (from the target-survey NMEA log[s])

The calibration script requires ELLIPSOID height. NMEA GGA sentences report
MSL altitude in field 9 and the geoid separation in field 11, so this parser
adds the two to produce ellipsoid height in meters.

Usage examples
--------------
Parse the two static logs, keeping every valid fix:

    python parse_nmea_to_csv.py static \
        --input GPS_MAIN_LOG_0000.nmea --output gps_main.csv
    python parse_nmea_to_csv.py static \
        --input gps_imu_antenna.nmea --output gps_imu_antenna.csv

Parse the target log(s), auto-averaging each stationary dwell into one row:

    python parse_nmea_to_csv.py targets \
        --input DLT_target_test1.nmea DLT_target_test1_0000.nmea \
        --output camera_antenna_point.csv

After running the targets command, open camera_antenna_point.csv and fill in
the image_name column (or replace it with camera_1_image / camera_2_image /
camera_3_image columns) so each row points to the correct calibration image.
"""

import argparse
import csv
import math
from pathlib import Path


def parse_nmea_latitude(field, hemisphere):
    """Convert NMEA ddmm.mmmm plus N/S hemisphere to signed decimal degrees."""
    if field == "" or hemisphere == "":
        return None

    value = float(field)
    degrees = int(value // 100)
    minutes = value - degrees * 100
    decimal = degrees + minutes / 60.0

    if hemisphere.upper() == "S":
        decimal = -decimal
    elif hemisphere.upper() != "N":
        return None

    return decimal


def parse_nmea_longitude(field, hemisphere):
    """Convert NMEA dddmm.mmmm plus E/W hemisphere to signed decimal degrees."""
    if field == "" or hemisphere == "":
        return None

    value = float(field)
    degrees = int(value // 100)
    minutes = value - degrees * 100
    decimal = degrees + minutes / 60.0

    if hemisphere.upper() == "W":
        decimal = -decimal
    elif hemisphere.upper() != "E":
        return None

    return decimal


def parse_nmea_time_to_seconds(field):
    """Convert an NMEA hhmmss.sss timestamp to seconds since UTC midnight."""
    if field == "":
        return None

    if "." in field:
        integer_part, fractional_part = field.split(".", 1)
    else:
        integer_part, fractional_part = field, "0"

    if len(integer_part) < 6:
        return None

    hours = int(integer_part[0:2])
    minutes = int(integer_part[2:4])
    seconds = int(integer_part[4:6])
    fractional = float("0." + fractional_part) if fractional_part else 0.0
    return hours * 3600 + minutes * 60 + seconds + fractional


def validate_nmea_checksum(sentence):
    """Return True if the NMEA checksum is present and valid, else False."""
    if "*" not in sentence:
        # Some loggers strip the checksum; accept these but flag with False.
        return False

    body, checksum_text = sentence.rsplit("*", 1)
    body = body.lstrip("$")
    checksum_text = checksum_text.strip()

    if len(checksum_text) < 2:
        return False

    try:
        expected = int(checksum_text[:2], 16)
    except ValueError:
        return False

    computed = 0

    for character in body:
        computed ^= ord(character)

    return computed == expected


def iterate_gga_fixes(nmea_paths, min_fix_quality, require_checksum):
    """Yield dicts of parsed GGA fixes from one or more NMEA files."""
    for nmea_path in nmea_paths:
        nmea_path = Path(nmea_path)

        with nmea_path.open("r", encoding="ascii", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()

                if not line.startswith("$"):
                    continue

                # Accept GPGGA, GNGGA, GLGGA, GBGGA, etc.
                if "GGA" not in line[:7]:
                    continue

                if require_checksum and not validate_nmea_checksum(line):
                    continue

                body = line.split("*", 1)[0]
                fields = body.split(",")

                if len(fields) < 12:
                    continue

                try:
                    fix_quality = int(fields[6]) if fields[6] != "" else 0
                except ValueError:
                    continue

                if fix_quality < min_fix_quality:
                    continue

                latitude_deg = parse_nmea_latitude(fields[2], fields[3])
                longitude_deg = parse_nmea_longitude(fields[4], fields[5])

                if latitude_deg is None or longitude_deg is None:
                    continue

                try:
                    msl_altitude_m = float(fields[9]) if fields[9] != "" else None
                except ValueError:
                    msl_altitude_m = None

                try:
                    geoid_separation_m = (
                        float(fields[11]) if fields[11] != "" else 0.0
                    )
                except ValueError:
                    geoid_separation_m = 0.0

                if msl_altitude_m is None:
                    continue

                ellipsoid_height_m = msl_altitude_m + geoid_separation_m
                seconds_of_day = parse_nmea_time_to_seconds(fields[1])

                if not all(
                    math.isfinite(value)
                    for value in [
                        latitude_deg,
                        longitude_deg,
                        ellipsoid_height_m,
                    ]
                ):
                    continue

                yield {
                    "source_file": nmea_path.name,
                    "line_number": line_number,
                    "seconds_of_day": seconds_of_day,
                    "latitude_deg": latitude_deg,
                    "longitude_deg": longitude_deg,
                    "msl_altitude_m": msl_altitude_m,
                    "geoid_separation_m": geoid_separation_m,
                    "ellipsoid_height_m": ellipsoid_height_m,
                    "fix_quality": fix_quality,
                }


def unwrap_seconds_of_day(fixes):
    """Add 86400 whenever the timestamp rolls backward past midnight."""
    previous_seconds = None
    offset = 0.0

    for fix in fixes:
        seconds = fix["seconds_of_day"]

        if seconds is None:
            fix["timestamp_s"] = None
            continue

        if previous_seconds is not None and seconds + offset < previous_seconds - 3600:
            offset += 86400.0

        unwrapped = seconds + offset
        fix["timestamp_s"] = unwrapped
        previous_seconds = unwrapped


def write_static_csv(fixes, output_path, include_timestamp):
    """Write one CSV row per parsed GGA fix."""
    columns = ["latitude_deg", "longitude_deg", "ellipsoid_height_m"]

    if include_timestamp:
        columns.append("timestamp_s")

    columns.extend(["fix_quality", "source_file"])

    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)

        for fix in fixes:
            row = [
                f"{fix['latitude_deg']:.9f}",
                f"{fix['longitude_deg']:.9f}",
                f"{fix['ellipsoid_height_m']:.4f}",
            ]

            if include_timestamp:
                row.append(
                    ""
                    if fix.get("timestamp_s") is None
                    else f"{fix['timestamp_s']:.3f}"
                )

            row.extend([fix["fix_quality"], fix["source_file"]])
            writer.writerow(row)


def haversine_distance_meters(latitude_a, longitude_a, latitude_b, longitude_b):
    """Great-circle distance for stationary-segment detection."""
    earth_radius_m = 6371000.0
    lat_a = math.radians(latitude_a)
    lat_b = math.radians(latitude_b)
    delta_lat = math.radians(latitude_b - latitude_a)
    delta_lon = math.radians(longitude_b - longitude_a)
    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_m * c


def detect_stationary_segments(
    fixes,
    stationary_radius_m,
    minimum_dwell_seconds,
    gap_seconds,
):
    """Group consecutive fixes that stay within a small radius into segments."""
    segments = []
    current = []

    def flush():
        if not current:
            return

        duration = 0.0

        if (
            current[0].get("timestamp_s") is not None
            and current[-1].get("timestamp_s") is not None
        ):
            duration = current[-1]["timestamp_s"] - current[0]["timestamp_s"]

        if duration >= minimum_dwell_seconds and len(current) >= 2:
            segments.append(list(current))

    for fix in fixes:
        if not current:
            current.append(fix)
            continue

        anchor = current[0]
        distance = haversine_distance_meters(
            anchor["latitude_deg"],
            anchor["longitude_deg"],
            fix["latitude_deg"],
            fix["longitude_deg"],
        )
        time_gap = None

        if (
            fix.get("timestamp_s") is not None
            and current[-1].get("timestamp_s") is not None
        ):
            time_gap = fix["timestamp_s"] - current[-1]["timestamp_s"]

        if distance > stationary_radius_m or (
            time_gap is not None and time_gap > gap_seconds
        ):
            flush()
            current = [fix]
        else:
            current.append(fix)

    flush()
    return segments


def summarize_segment(segment):
    """Average lat/lon/height across a stationary segment."""
    count = len(segment)
    mean_latitude = sum(fix["latitude_deg"] for fix in segment) / count
    mean_longitude = sum(fix["longitude_deg"] for fix in segment) / count
    mean_ellipsoid_height = (
        sum(fix["ellipsoid_height_m"] for fix in segment) / count
    )
    start_timestamp = segment[0].get("timestamp_s")
    end_timestamp = segment[-1].get("timestamp_s")

    if start_timestamp is not None and end_timestamp is not None:
        duration = end_timestamp - start_timestamp
    else:
        duration = None

    return {
        "latitude_deg": mean_latitude,
        "longitude_deg": mean_longitude,
        "ellipsoid_height_m": mean_ellipsoid_height,
        "sample_count": count,
        "start_timestamp_s": start_timestamp,
        "end_timestamp_s": end_timestamp,
        "duration_s": duration,
    }


def write_target_csv(segments, output_path):
    """Write one CSV row per detected target dwell segment."""
    columns = [
        "latitude_deg",
        "longitude_deg",
        "ellipsoid_height_m",
        "image_name",
        "sample_count",
        "start_timestamp_s",
        "end_timestamp_s",
        "duration_s",
    ]

    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)

        for segment_index, segment in enumerate(segments, start=1):
            summary = summarize_segment(segment)
            writer.writerow(
                [
                    f"{summary['latitude_deg']:.9f}",
                    f"{summary['longitude_deg']:.9f}",
                    f"{summary['ellipsoid_height_m']:.4f}",
                    f"target_{segment_index:03d}.jpg",
                    summary["sample_count"],
                    ""
                    if summary["start_timestamp_s"] is None
                    else f"{summary['start_timestamp_s']:.3f}",
                    ""
                    if summary["end_timestamp_s"] is None
                    else f"{summary['end_timestamp_s']:.3f}",
                    ""
                    if summary["duration_s"] is None
                    else f"{summary['duration_s']:.3f}",
                ]
            )


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    static_parser = subparsers.add_parser(
        "static",
        help="Write one CSV row per valid GGA fix.",
    )
    static_parser.add_argument("--input", nargs="+", required=True)
    static_parser.add_argument("--output", required=True)
    static_parser.add_argument("--min-fix-quality", type=int, default=4)
    static_parser.add_argument("--require-checksum", action="store_true")
    static_parser.add_argument("--no-timestamp", action="store_true")

    target_parser = subparsers.add_parser(
        "targets",
        help="Detect stationary dwell segments and average each into a row.",
    )
    target_parser.add_argument("--input", nargs="+", required=True)
    target_parser.add_argument("--output", required=True)
    target_parser.add_argument("--min-fix-quality", type=int, default=4)
    target_parser.add_argument("--require-checksum", action="store_true")
    target_parser.add_argument("--stationary-radius-m", type=float, default=0.15)
    target_parser.add_argument("--minimum-dwell-seconds", type=float, default=15.0)
    target_parser.add_argument("--gap-seconds", type=float, default=5.0)

    return parser


def main():
    parser = build_argument_parser()
    arguments = parser.parse_args()

    all_fixes = list(
        iterate_gga_fixes(
            arguments.input,
            min_fix_quality=arguments.min_fix_quality,
            require_checksum=arguments.require_checksum,
        )
    )

    if not all_fixes:
        raise SystemExit(
            "No GGA fixes at or above the requested fix quality were found. "
            "Lower --min-fix-quality or check the input files."
        )

    unwrap_seconds_of_day(all_fixes)

    if arguments.command == "static":
        write_static_csv(
            all_fixes,
            arguments.output,
            include_timestamp=not arguments.no_timestamp,
        )
        print(f"Wrote {len(all_fixes)} rows to {arguments.output}")
        return

    if arguments.command == "targets":
        segments = detect_stationary_segments(
            all_fixes,
            stationary_radius_m=arguments.stationary_radius_m,
            minimum_dwell_seconds=arguments.minimum_dwell_seconds,
            gap_seconds=arguments.gap_seconds,
        )

        if not segments:
            raise SystemExit(
                "No stationary dwell segments were detected. Loosen "
                "--stationary-radius-m or --minimum-dwell-seconds."
            )

        write_target_csv(segments, arguments.output)
        print(
            f"Detected {len(segments)} target segments from "
            f"{len(all_fixes)} fixes; wrote {arguments.output}"
        )
        print(
            "Now open the CSV and replace each target_XXX.jpg placeholder "
            "with the actual image filename for that dwell (or split into "
            "camera_1_image / camera_2_image / camera_3_image columns)."
        )
        return


if __name__ == "__main__":
    main()
