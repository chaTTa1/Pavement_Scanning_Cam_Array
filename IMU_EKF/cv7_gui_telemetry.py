"""Non-blocking text telemetry for the CV7 Pangolin viewer.

The recorder remains the source of truth and writes every event to CSV.  This
module only publishes a low-rate, best-effort snapshot for visualization.  A
lost UDP datagram therefore cannot create a gap in the recorded sensor data.
"""

from __future__ import annotations

import math
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


PROTOCOL_HEADER = "CV7GUI/1"
EARTH_RADIUS_M = 6378137.0

FILTER_STATE_TEXT = {
    0: "Startup",
    1: "Initialization",
    2: "Vertical Gyro",
    3: "AHRS",
    4: "Full Navigation",
}

FILTER_CONDITION_TEXT = {
    0: "Unknown",
    1: "Stable",
    2: "Converging",
    3: "Unstable or Recovering",
}

FIX_QUALITY_TEXT = {
    0: "Invalid",
    1: "Autonomous GNSS",
    2: "Differential GNSS",
    3: "PPS Fix",
    4: "RTK Fixed",
    5: "RTK Float",
    6: "Estimated or Dead Reckoning",
    7: "Manual Input",
    8: "Simulation",
}

FILTER_WARNING_BITS = {
    2: "Roll or pitch warning",
    3: "Heading warning",
    4: "Position warning",
    5: "Velocity warning",
    6: "IMU bias warning",
    7: "GNSS clock warning",
    8: "Antenna lever arm warning",
    9: "Mounting transform warning",
    10: "Time synchronization warning",
    12: "Filter solution error",
    13: "Filter solution error",
    14: "Filter solution error",
    15: "Filter solution error",
}


def _float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> Optional[int]:
    number = _float(value)
    return int(number) if number is not None else None


def _valid_position(latitude: Optional[float], longitude: Optional[float]) -> bool:
    if latitude is None or longitude is None:
        return False
    if abs(latitude) < 1e-12 and abs(longitude) < 1e-12:
        return False
    return -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0


def _sanitize_text(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("=", ":")


def encode_snapshot(snapshot: Dict[str, Any]) -> bytes:
    """Encode one dependency-free UTF-8 key/value UDP datagram."""
    lines = [PROTOCOL_HEADER]
    for key in sorted(snapshot):
        value = snapshot[key]
        if value is None:
            text = ""
        elif isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, float):
            text = f"{value:.12g}" if math.isfinite(value) else ""
        else:
            text = _sanitize_text(value)
        lines.append(f"{key}={text}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def decode_snapshot(payload: bytes) -> Dict[str, str]:
    """Decode the text protocol; primarily used by tests and diagnostics."""
    text = payload.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if not lines or lines[0] != PROTOCOL_HEADER:
        raise ValueError("not a CV7 GUI telemetry packet")
    result: Dict[str, str] = {}
    for line in lines[1:]:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


def filter_state_text(value: Any) -> str:
    state = _int(value)
    if state is None:
        return "Waiting for filter state"
    return FILTER_STATE_TEXT.get(state, "Unrecognized filter state")


def filter_condition_text(flags_value: Any) -> str:
    flags = _int(flags_value)
    if flags is None:
        return "Waiting for filter condition"
    return FILTER_CONDITION_TEXT.get(flags & 0x03, "Unknown")


def filter_warnings_text(flags_value: Any) -> str:
    flags = _int(flags_value)
    if flags is None:
        return "Waiting for filter status"
    warnings = [
        label for bit, label in FILTER_WARNING_BITS.items() if flags & (1 << bit)
    ]
    # Bits 12-15 all describe solution errors; avoid repeating the same text.
    warnings = list(dict.fromkeys(warnings))
    return "; ".join(warnings) if warnings else "No warnings"


def fix_quality_text(value: Any) -> str:
    quality = _int(value)
    if quality is None:
        return "Waiting for GGA fix"
    return FIX_QUALITY_TEXT.get(quality, "Unrecognized GNSS fix")


def aiding_status_text(indicators: Iterable[int]) -> str:
    values = list(indicators)
    if not values:
        return "Waiting for aiding status"
    if any(value & 0x10 for value in values):
        return "Configuration error"
    if any(value & 0x08 for value in values):
        return "Sample time warning"
    if any(value & 0x20 for value in values):
        return "Measurement limit exceeded"
    if all((value & 0x03) == 0x03 for value in values):
        return "Enabled and Used"
    if any(value & 0x02 for value in values) and any(
        (value & 0x03) != 0x03 for value in values
    ):
        return "Partially used"
    if any(value & 0x02 for value in values):
        return "Used"
    if any(value & 0x01 for value in values):
        return "Enabled, not used"
    return "Disabled"


class TelemetryState:
    """Aggregate native-rate recorder events into a low-rate GUI snapshot."""

    def __init__(
        self,
        cv7_port: str,
        gps_port: Optional[str],
        output_path: Path,
        start_monotonic: Optional[float] = None,
    ) -> None:
        self.cv7_port = cv7_port
        self.gps_port = gps_port or "Disabled"
        self.output_path = str(output_path)
        self.start_monotonic = start_monotonic or time.monotonic()

        self.sequence = 0
        self.row_count = 0
        self.source_counts: Dict[str, int] = {}
        self.previous_rate_counts: Dict[str, int] = {}
        self.rates: Dict[str, float] = {}
        self.last_rate_time = self.start_monotonic

        self.last_cv7_time: Optional[float] = None
        self.last_imu_time: Optional[float] = None
        self.last_ekf_time: Optional[float] = None
        self.last_gps_time: Optional[float] = None
        self.last_gga_time: Optional[float] = None
        self.last_velocity_time: Optional[float] = None
        self.last_hdt_sentence_time: Optional[float] = None

        self.filter_state: Optional[int] = None
        self.filter_flags: Optional[int] = None
        self.aiding_indicators: Tuple[int, ...] = ()

        self.fix_quality: Optional[int] = None
        self.satellites: Optional[int] = None
        self.hdop: Optional[float] = None
        self.gps_heading_deg: Optional[float] = None
        self.gps_time_offset_s: Optional[float] = None

        self.latitude_gps: Optional[float] = None
        self.longitude_gps: Optional[float] = None
        self.latitude_ekf: Optional[float] = None
        self.longitude_ekf: Optional[float] = None
        self.origin: Optional[Tuple[float, float]] = None
        self.gps_east_m: Optional[float] = None
        self.gps_north_m: Optional[float] = None
        self.ekf_east_m: Optional[float] = None
        self.ekf_north_m: Optional[float] = None
        self.gps_sample_id = 0
        self.ekf_sample_id = 0

        self.yaw_deg: Optional[float] = None
        self.speed_mps: Optional[float] = None
        self.position_uncertainty_m: Optional[float] = None
        self.velocity_uncertainty_mps: Optional[float] = None
        self.accel_x: Optional[float] = None
        self.accel_y: Optional[float] = None
        self.accel_z: Optional[float] = None
        self.gyro_x: Optional[float] = None
        self.gyro_y: Optional[float] = None
        self.gyro_z: Optional[float] = None
        self.gst_horizontal_sigma_m: Optional[float] = None

    def _event_monotonic(self, event: Dict[str, Any]) -> float:
        elapsed = _float(event.get("elapsed_s"))
        if elapsed is None:
            return time.monotonic()
        return self.start_monotonic + elapsed

    def _set_origin_if_needed(self, latitude: float, longitude: float) -> None:
        if self.origin is None and _valid_position(latitude, longitude):
            self.origin = (latitude, longitude)

    def _to_local(
        self, latitude: Optional[float], longitude: Optional[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        if self.origin is None or not _valid_position(latitude, longitude):
            return None, None
        ref_lat, ref_lon = self.origin
        east = (
            math.radians(float(longitude) - ref_lon)
            * EARTH_RADIUS_M
            * math.cos(math.radians(ref_lat))
        )
        north = math.radians(float(latitude) - ref_lat) * EARTH_RADIUS_M
        return east, north

    @staticmethod
    def _fresh(last_time: Optional[float], now: float, timeout: float = 1.0) -> bool:
        return last_time is not None and now - last_time <= timeout

    def update(self, event: Dict[str, Any]) -> None:
        source = str(event.get("source", "UNKNOWN"))
        now = self._event_monotonic(event)
        self.row_count += 1
        self.source_counts[source] = self.source_counts.get(source, 0) + 1

        if source.startswith("CV7_"):
            self.last_cv7_time = now
        if source == "CV7_IMU":
            self.last_imu_time = now
        if source == "CV7_EKF":
            self.last_ekf_time = now
        if source == "GPS_NMEA":
            self.last_gps_time = now

        mip = event.get("_mip") or {}
        if mip:
            self._update_mip(mip)
        if source == "GPS_NMEA":
            self._update_gps(event, now)

    def _update_mip(self, mip: Dict[str, Any]) -> None:
        if "estFilterState" in mip:
            self.filter_state = _int(mip.get("estFilterState"))
        if "estFilterStatusFlags" in mip:
            self.filter_flags = _int(mip.get("estFilterStatusFlags"))

        indicators = []
        for key, value in mip.items():
            if key.startswith("aidingSummary_status"):
                indicator = _int(value)
                if indicator is not None:
                    indicators.append(indicator)
        if indicators:
            self.aiding_indicators = tuple(indicators)

        latitude = _float(mip.get("estLatitude"))
        longitude = _float(mip.get("estLongitude"))
        if _valid_position(latitude, longitude):
            self.latitude_ekf = latitude
            self.longitude_ekf = longitude
            self._set_origin_if_needed(float(latitude), float(longitude))
            self.ekf_east_m, self.ekf_north_m = self._to_local(latitude, longitude)
            self.ekf_sample_id += 1

        yaw = _float(mip.get("estYaw"))
        if yaw is not None:
            self.yaw_deg = math.degrees(yaw) % 360.0

        vn = _float(mip.get("estNorthVelocity"))
        ve = _float(mip.get("estEastVelocity"))
        if vn is not None and ve is not None:
            self.speed_mps = math.hypot(vn, ve)

        north_uncert = _float(mip.get("estNorthPositionUncert"))
        east_uncert = _float(mip.get("estEastPositionUncert"))
        available_pos_uncert = [
            value for value in (north_uncert, east_uncert) if value is not None
        ]
        if available_pos_uncert:
            self.position_uncertainty_m = max(available_pos_uncert)

        vn_uncert = _float(mip.get("estNorthVelocityUncert"))
        ve_uncert = _float(mip.get("estEastVelocityUncert"))
        if vn_uncert is not None and ve_uncert is not None:
            self.velocity_uncertainty_mps = math.hypot(vn_uncert, ve_uncert)

        for attribute, channel in (
            ("accel_x", "scaledAccelX"),
            ("accel_y", "scaledAccelY"),
            ("accel_z", "scaledAccelZ"),
            ("gyro_x", "scaledGyroX"),
            ("gyro_y", "scaledGyroY"),
            ("gyro_z", "scaledGyroZ"),
        ):
            value = _float(mip.get(channel))
            if value is not None:
                setattr(self, attribute, value)

    def _update_gps(self, event: Dict[str, Any], now: float) -> None:
        message_type = str(event.get("gps_message_type", ""))
        if message_type == "GGA":
            self.last_gga_time = now
            self.fix_quality = _int(event.get("gps_fix_quality"))
            self.satellites = _int(event.get("gps_num_satellites"))
            self.hdop = _float(event.get("gps_hdop"))
            latitude = _float(event.get("gps_latitude_deg"))
            longitude = _float(event.get("gps_longitude_deg"))
            if _valid_position(latitude, longitude):
                self.latitude_gps = latitude
                self.longitude_gps = longitude
                self._set_origin_if_needed(float(latitude), float(longitude))
                self.gps_east_m, self.gps_north_m = self._to_local(
                    latitude, longitude
                )
                self.gps_sample_id += 1

            gps_datetime = str(event.get("gps_datetime_utc", ""))
            host_unix_s = _float(event.get("host_unix_s"))
            if gps_datetime and host_unix_s is not None:
                try:
                    from datetime import datetime

                    gps_unix_s = datetime.fromisoformat(
                        gps_datetime.replace("Z", "+00:00")
                    ).timestamp()
                    self.gps_time_offset_s = host_unix_s - gps_unix_s
                except ValueError:
                    pass

        if message_type in {"RMC", "VTG"}:
            self.last_velocity_time = now

        if message_type == "HDT":
            self.last_hdt_sentence_time = now
            heading = _float(event.get("gps_heading_true_deg"))
            self.gps_heading_deg = heading % 360.0 if heading is not None else None

        if message_type == "GST":
            lat_sigma = _float(event.get("gps_lat_sigma_m"))
            lon_sigma = _float(event.get("gps_lon_sigma_m"))
            if lat_sigma is not None and lon_sigma is not None:
                self.gst_horizontal_sigma_m = math.hypot(lat_sigma, lon_sigma)

    def _update_rates(self, now: float) -> None:
        elapsed = now - self.last_rate_time
        if elapsed < 0.5:
            return
        for source, count in self.source_counts.items():
            previous = self.previous_rate_counts.get(source, 0)
            instantaneous = max(0.0, (count - previous) / elapsed)
            old = self.rates.get(source)
            self.rates[source] = instantaneous if old is None else old * 0.65 + instantaneous * 0.35
            self.previous_rate_counts[source] = count
        self.last_rate_time = now

    def snapshot(
        self,
        recorder_status: str = "Recording",
        shutdown: bool = False,
    ) -> Dict[str, Any]:
        now = time.monotonic()
        self._update_rates(now)
        self.sequence += 1

        cv7_connected = self._fresh(self.last_cv7_time, now)
        imu_connected = self._fresh(self.last_imu_time, now)
        ekf_connected = self._fresh(self.last_ekf_time, now)
        gps_connected = self._fresh(self.last_gps_time, now)
        gga_fresh = self._fresh(self.last_gga_time, now)
        velocity_fresh = self._fresh(self.last_velocity_time, now)
        hdt_sentence_fresh = self._fresh(self.last_hdt_sentence_time, now)

        general_aiding = aiding_status_text(self.aiding_indicators)
        position_status = (
            "Enabled and Used"
            if gga_fresh and general_aiding == "Enabled and Used"
            else "GGA available; use not individually identified"
            if gga_fresh and general_aiding == "Partially used"
            else "GGA available, awaiting confirmed use"
            if gga_fresh
            else "Waiting for GGA"
        )
        velocity_status = (
            "Enabled and Used"
            if velocity_fresh and general_aiding == "Enabled and Used"
            else "Velocity available; use not individually identified"
            if velocity_fresh and general_aiding == "Partially used"
            else "Velocity input available, awaiting confirmed use"
            if velocity_fresh
            else "Waiting for RMC or VTG"
        )
        heading_status = (
            "HDT available"
            if hdt_sentence_fresh and self.gps_heading_deg is not None
            else "HDT sentence has no heading"
            if hdt_sentence_fresh
            else "Waiting for HDT"
        )

        time_status = "Waiting for GPS time"
        if self.gps_time_offset_s is not None:
            time_status = (
                "Synchronized"
                if abs(self.gps_time_offset_s) <= 0.5
                else "Time offset warning"
            )

        return {
            "sequence": self.sequence,
            "shutdown": shutdown,
            "recorder_status": recorder_status,
            "elapsed_s": now - self.start_monotonic,
            "row_count": self.row_count,
            "csv_path": self.output_path,
            "cv7_port": self.cv7_port,
            "gps_port": self.gps_port,
            "cv7_connection": "Connected" if cv7_connected else "No MIP data",
            "gps_connection": "Receiving NMEA" if gps_connected else "No NMEA data",
            "imu_status": "Receiving data" if imu_connected else "No IMU data",
            "ekf_status": "Receiving data" if ekf_connected else "No EKF data",
            "filter_state": filter_state_text(self.filter_state),
            "filter_condition": filter_condition_text(self.filter_flags),
            "filter_warnings": filter_warnings_text(self.filter_flags),
            "aiding_status": general_aiding,
            "position_aiding_status": position_status,
            "velocity_aiding_status": velocity_status,
            "heading_status": heading_status,
            "gnss_fix": fix_quality_text(self.fix_quality),
            "time_sync_status": time_status,
            "cv7_imu_rate_hz": self.rates.get("CV7_IMU"),
            "cv7_ekf_rate_hz": self.rates.get("CV7_EKF"),
            "gps_nmea_rate_hz": self.rates.get("GPS_NMEA"),
            "satellites": self.satellites,
            "hdop": self.hdop,
            "gps_heading_deg": self.gps_heading_deg,
            "gps_time_offset_s": self.gps_time_offset_s,
            "yaw_deg": self.yaw_deg,
            "speed_mps": self.speed_mps,
            "position_uncertainty_m": self.position_uncertainty_m,
            "velocity_uncertainty_mps": self.velocity_uncertainty_mps,
            "gst_horizontal_sigma_m": self.gst_horizontal_sigma_m,
            "accel_x": self.accel_x,
            "accel_y": self.accel_y,
            "accel_z": self.accel_z,
            "gyro_x": self.gyro_x,
            "gyro_y": self.gyro_y,
            "gyro_z": self.gyro_z,
            "gps_latitude": self.latitude_gps,
            "gps_longitude": self.longitude_gps,
            "ekf_latitude": self.latitude_ekf,
            "ekf_longitude": self.longitude_ekf,
            "gps_east_m": self.gps_east_m,
            "gps_north_m": self.gps_north_m,
            "ekf_east_m": self.ekf_east_m,
            "ekf_north_m": self.ekf_north_m,
            "gps_sample_id": self.gps_sample_id,
            "ekf_sample_id": self.ekf_sample_id,
        }


class UdpTelemetryPublisher:
    """Publish snapshots and receive an optional stop command from the viewer."""

    def __init__(self, host: str, port: int, rate_hz: float) -> None:
        if not (1 <= port <= 65534):
            raise ValueError("GUI UDP port must be between 1 and 65534")
        if rate_hz <= 0:
            raise ValueError("GUI update rate must be positive")
        self.target = (host, port)
        self.period_s = 1.0 / rate_hz
        self.last_send_time = 0.0
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.control_socket: Optional[socket.socket] = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM
        )
        try:
            self.control_socket.bind(("0.0.0.0", port + 1))
            self.control_socket.setblocking(False)
        except OSError as exc:
            print(
                f"WARNING: GUI stop-control port {port + 1} unavailable ({exc}); "
                "viewer display will still work."
            )
            self.control_socket.close()
            self.control_socket = None

    def maybe_send(
        self,
        state: TelemetryState,
        recorder_status: str = "Recording",
        force: bool = False,
        shutdown: bool = False,
    ) -> bool:
        now = time.monotonic()
        if not force and now - self.last_send_time < self.period_s:
            return False
        payload = encode_snapshot(state.snapshot(recorder_status, shutdown))
        try:
            self.send_socket.sendto(payload, self.target)
        except OSError as exc:
            print(f"WARNING: GUI telemetry send failed: {exc}")
            return False
        self.last_send_time = now
        return True

    def stop_requested(self) -> bool:
        if self.control_socket is None:
            return False
        requested = False
        while True:
            try:
                payload, _address = self.control_socket.recvfrom(2048)
            except BlockingIOError:
                return requested
            except OSError:
                return requested
            try:
                message = decode_snapshot(payload)
            except ValueError:
                continue
            if message.get("command") == "stop":
                requested = True

    def close(self) -> None:
        self.send_socket.close()
        if self.control_socket is not None:
            self.control_socket.close()


def find_viewer_executable(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        return candidate if candidate.is_file() else None

    base = Path(__file__).resolve().parent / "pangolin_viewer" / "build"
    release = (
        Path(__file__).resolve().parent
        / "pangolin_viewer"
        / "release"
        / "windows-x64"
    )
    candidates = [
        release / "cv7_pangolin_viewer.exe",
        base / "Release" / "cv7_pangolin_viewer.exe",
        base / "cv7_pangolin_viewer.exe",
        base / "cv7_pangolin_viewer",
    ]
    on_path = shutil.which("cv7_pangolin_viewer")
    if on_path:
        candidates.append(Path(on_path))
    return next((path.resolve() for path in candidates if path.is_file()), None)


def launch_viewer(executable: Path, port: int) -> subprocess.Popen[Any]:
    return subprocess.Popen([str(executable), "--port", str(port)])
