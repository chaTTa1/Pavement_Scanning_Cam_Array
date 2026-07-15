import socket
import time
import unittest
from pathlib import Path

from IMU_EKF.cv7_gui_telemetry import (
    TelemetryState,
    UdpTelemetryPublisher,
    aiding_status_text,
    decode_snapshot,
    encode_snapshot,
    filter_condition_text,
    filter_state_text,
    filter_warnings_text,
    fix_quality_text,
)


class TextStatusTests(unittest.TestCase):
    def test_known_codes_are_exposed_as_words(self) -> None:
        self.assertEqual(filter_state_text(4), "Full Navigation")
        self.assertEqual(filter_condition_text(1), "Stable")
        self.assertEqual(filter_warnings_text(1), "No warnings")
        self.assertEqual(aiding_status_text([3, 3, 3]), "Enabled and Used")
        self.assertEqual(aiding_status_text([3, 3, 1]), "Partially used")
        self.assertEqual(fix_quality_text(4), "RTK Fixed")
        self.assertEqual(fix_quality_text(5), "RTK Float")

    def test_snapshot_round_trip_preserves_text_status(self) -> None:
        start = time.monotonic()
        state = TelemetryState("COM13", "COM16", Path("record.csv"), start)
        state.update(
            {
                "source": "CV7_EKF",
                "elapsed_s": 0.01,
                "_mip": {
                    "estFilterState": 4,
                    "estFilterStatusFlags": 1,
                    "aidingSummary_status_aidType_1": 3,
                    "aidingSummary_status_aidType_1__2": 3,
                    "aidingSummary_status_aidType_1__3": 3,
                    "estLatitude": 40.0,
                    "estLongitude": -80.0,
                    "estYaw": 0.25,
                },
            }
        )
        state.update(
            {
                "source": "GPS_NMEA",
                "elapsed_s": 0.02,
                "gps_message_type": "GGA",
                "gps_fix_quality": 4,
                "gps_num_satellites": 18,
                "gps_hdop": 0.7,
                "gps_latitude_deg": 40.000001,
                "gps_longitude_deg": -80.000001,
            }
        )

        decoded = decode_snapshot(encode_snapshot(state.snapshot()))
        self.assertEqual(decoded["filter_state"], "Full Navigation")
        self.assertEqual(decoded["filter_condition"], "Stable")
        self.assertEqual(decoded["aiding_status"], "Enabled and Used")
        self.assertEqual(decoded["position_aiding_status"], "Enabled and Used")
        self.assertEqual(decoded["gnss_fix"], "RTK Fixed")

    def test_udp_stop_command_is_text_protocol(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        if port >= 65534:
            self.skipTest("ephemeral UDP port has no adjacent control port")

        publisher = UdpTelemetryPublisher("127.0.0.1", port, 10.0)
        command = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            command.sendto(
                encode_snapshot({"command": "stop"}),
                ("127.0.0.1", port + 1),
            )
            deadline = time.monotonic() + 1.0
            received = False
            while time.monotonic() < deadline:
                if publisher.stop_requested():
                    received = True
                    break
                time.sleep(0.01)
            self.assertTrue(received, "publisher did not receive GUI stop command")
        finally:
            command.close()
            publisher.close()


if __name__ == "__main__":
    unittest.main()
