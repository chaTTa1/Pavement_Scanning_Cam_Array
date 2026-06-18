#!/usr/bin/env python3
"""
Configure a MicroStrain 3DM-CV7/GV7 for Teensy external GNSS aiding.

This sends MIP commands over the CV7 USB/main serial port:

    - PPS source = GPIO
    - GPIO1 = PPS input
    - Optional: GPIO2/GPIO3 = UART2 RX/TX for external MIP aiding input
      (TX can be skipped if the device does not support UART2 TX on GPIO)
    - GNSS position/velocity aiding = enabled
    - External heading aiding = disabled by default
    - Save settings to startup memory

Close SensorConnect before running this script because only one program can own
the COM port at a time on Windows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time

try:
    import serial
except ImportError as exc:
    raise SystemExit("pyserial is required. Install it with: python -m pip install pyserial") from exc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from CV7_INS_EKF import (  # noqa: E402
    CMD_BASE_SET_TO_IDLE,
    DESC_3DM_CMD,
    DESC_BASE,
    MipError,
    MipReader,
    detect_serial_port,
    send_command,
)


MIP_FUNCTION_WRITE = 0x01
MIP_FUNCTION_SAVE = 0x03

CMD_3DM_PPS_SOURCE = 0x28
CMD_3DM_DEVICE_SETTINGS = 0x30
CMD_3DM_GPIO_CONFIG = 0x41
CMD_FILTER_RESET_FILTER = 0x01
CMD_FILTER_RUN = 0x05
CMD_FILTER_AIDING_MEASUREMENT_ENABLE = 0x50
CMD_FILTER_INITIALIZATION_CONFIGURATION = 0x52
CMD_SYSTEM_INTERFACE_CONTROL = 0x02

DESC_FILTER_CMD = 0x0D
DESC_SYSTEM_CMD = 0x7F

PPS_SOURCE_GPIO = 0x03

GPIO_FEATURE_PPS = 0x02
GPIO_FEATURE_UART = 0x05
GPIO_BEHAVIOR_PPS_INPUT = 0x01
GPIO_BEHAVIOR_UART_PORT2_TX = 0x21
GPIO_BEHAVIOR_UART_PORT2_RX = 0x22
GPIO_PIN_MODE_NONE = 0x00

AIDING_SOURCE_GNSS_POS_VEL = 0x0000
AIDING_SOURCE_EXTERNAL_HEADING = 0x0005

MIP_COMMS_INTERFACE_UART_2 = 18
MIP_COMMS_PROTOCOL_NONE = 0x00000000
MIP_COMMS_PROTOCOL_MIP = 0x00000001


def bool_u8(value: bool) -> int:
    return 1 if value else 0


def configure_cv7(args: argparse.Namespace) -> None:
    port = args.port or detect_serial_port()
    print(json.dumps({"event": "open_serial", "port": port, "baud": args.baud}, sort_keys=True))

    if args.configure_uart2_mip_on_gpio:
        used_pins = {args.pps_gpio_pin, args.uart2_rx_gpio_pin}
        if args.configure_uart2_tx:
            used_pins.add(args.uart2_tx_gpio_pin)
        expected_pin_count = 3 if args.configure_uart2_tx else 2
        if len(used_pins) != expected_pin_count:
            raise SystemExit("PPS, UART2 RX, and UART2 TX must use different GPIO pins.")

    with serial.Serial(port=port, baudrate=args.baud, timeout=args.timeout_s) as ser:
        reader = MipReader(ser)

        if args.filter_only:
            if args.init_heading_deg is None and not args.reset_filter and not args.run_filter:
                raise SystemExit("--filter-only requires --init-heading-deg, --reset-filter, and/or --run-filter.")
            run_initialization_user_heading(
                ser,
                reader,
                args.init_heading_deg,
                args.save_init_config,
                args.timeout_s,
            )
            run_filter_controls(ser, reader, args.reset_filter, args.run_filter, args.timeout_s)
            return

        if not args.no_idle:
            run_step(ser, reader, DESC_BASE, CMD_BASE_SET_TO_IDLE, b"", "set_to_idle", args.timeout_s)
            time.sleep(0.1)

        run_step(
            ser,
            reader,
            DESC_3DM_CMD,
            CMD_3DM_PPS_SOURCE,
            bytes((MIP_FUNCTION_WRITE, PPS_SOURCE_GPIO)),
            "pps_source_gpio",
            args.timeout_s,
        )

        run_step(
            ser,
            reader,
            DESC_3DM_CMD,
            CMD_3DM_GPIO_CONFIG,
            bytes(
                (
                    MIP_FUNCTION_WRITE,
                    args.pps_gpio_pin,
                    GPIO_FEATURE_PPS,
                    GPIO_BEHAVIOR_PPS_INPUT,
                    GPIO_PIN_MODE_NONE,
                )
            ),
            f"gpio{args.pps_gpio_pin}_pps_input",
            args.timeout_s,
        )

        if args.configure_uart2_mip_on_gpio:
            run_gpio_uart2_config(
                ser,
                reader,
                args.uart2_rx_gpio_pin,
                GPIO_BEHAVIOR_UART_PORT2_RX,
                f"gpio{args.uart2_rx_gpio_pin}_uart2_rx_mip_input",
                args.timeout_s,
            )
            if args.configure_uart2_tx:
                run_gpio_uart2_config(
                    ser,
                    reader,
                    args.uart2_tx_gpio_pin,
                    GPIO_BEHAVIOR_UART_PORT2_TX,
                    f"gpio{args.uart2_tx_gpio_pin}_uart2_tx_mip_ack",
                    args.timeout_s,
                )
            run_interface_control(
                ser,
                reader,
                MIP_COMMS_INTERFACE_UART_2,
                MIP_COMMS_PROTOCOL_MIP,
                MIP_COMMS_PROTOCOL_NONE,
                "uart2_enable_mip_incoming",
                args.timeout_s,
            )

        run_aiding_enable(
            ser,
            reader,
            AIDING_SOURCE_GNSS_POS_VEL,
            True,
            "enable_gnss_pos_vel_aiding",
            args.timeout_s,
        )
        run_aiding_enable(
            ser,
            reader,
            AIDING_SOURCE_EXTERNAL_HEADING,
            args.enable_external_heading,
            "set_external_heading_aiding",
            args.timeout_s,
        )

        run_initialization_user_heading(
            ser,
            reader,
            args.init_heading_deg,
            False,
            args.timeout_s,
        )

        if args.save:
            run_step(
                ser,
                reader,
                DESC_3DM_CMD,
                CMD_3DM_PPS_SOURCE,
                bytes((MIP_FUNCTION_SAVE,)),
                "save_pps_source",
                args.timeout_s,
            )
            run_step(
                ser,
                reader,
                DESC_3DM_CMD,
                CMD_3DM_GPIO_CONFIG,
                bytes((MIP_FUNCTION_SAVE, args.pps_gpio_pin)),
                f"save_gpio{args.pps_gpio_pin}_config",
                args.timeout_s,
            )
            if args.configure_uart2_mip_on_gpio:
                run_step(
                    ser,
                    reader,
                    DESC_3DM_CMD,
                    CMD_3DM_GPIO_CONFIG,
                    bytes((MIP_FUNCTION_SAVE, args.uart2_rx_gpio_pin)),
                    f"save_gpio{args.uart2_rx_gpio_pin}_uart2_rx",
                    args.timeout_s,
                )
                if args.configure_uart2_tx:
                    run_step(
                        ser,
                        reader,
                        DESC_3DM_CMD,
                        CMD_3DM_GPIO_CONFIG,
                        bytes((MIP_FUNCTION_SAVE, args.uart2_tx_gpio_pin)),
                        f"save_gpio{args.uart2_tx_gpio_pin}_uart2_tx",
                        args.timeout_s,
                    )
                run_step(
                    ser,
                    reader,
                    DESC_SYSTEM_CMD,
                    CMD_SYSTEM_INTERFACE_CONTROL,
                    bytes((MIP_FUNCTION_SAVE, MIP_COMMS_INTERFACE_UART_2)),
                    "save_uart2_interface_control",
                    args.timeout_s,
                )
            run_aiding_save(ser, reader, AIDING_SOURCE_GNSS_POS_VEL, "save_gnss_pos_vel_aiding", args.timeout_s)
            run_aiding_save(
                ser,
                reader,
                AIDING_SOURCE_EXTERNAL_HEADING,
                "save_external_heading_aiding",
                args.timeout_s,
            )
            if args.init_heading_deg is not None and args.save_init_config:
                run_step(
                    ser,
                    reader,
                    DESC_FILTER_CMD,
                    CMD_FILTER_INITIALIZATION_CONFIGURATION,
                    bytes((MIP_FUNCTION_SAVE,)),
                    "save_initialization_user_heading",
                    args.timeout_s,
                )
            run_step(
                ser,
                reader,
                DESC_3DM_CMD,
                CMD_3DM_DEVICE_SETTINGS,
                bytes((MIP_FUNCTION_SAVE,)),
                "save_device_settings",
                args.timeout_s,
            )

        run_filter_controls(ser, reader, args.reset_filter, args.run_filter, args.timeout_s)


def run_aiding_enable(
    ser: serial.Serial,
    reader: MipReader,
    source: int,
    enable: bool,
    name: str,
    timeout_s: float,
) -> None:
    payload = bytes((MIP_FUNCTION_WRITE,)) + source.to_bytes(2, "big") + bytes((bool_u8(enable),))
    run_step(ser, reader, DESC_FILTER_CMD, CMD_FILTER_AIDING_MEASUREMENT_ENABLE, payload, name, timeout_s)


def run_aiding_save(
    ser: serial.Serial,
    reader: MipReader,
    source: int,
    name: str,
    timeout_s: float,
) -> None:
    payload = bytes((MIP_FUNCTION_SAVE,)) + source.to_bytes(2, "big")
    run_step(ser, reader, DESC_FILTER_CMD, CMD_FILTER_AIDING_MEASUREMENT_ENABLE, payload, name, timeout_s)


def run_gpio_uart2_config(
    ser: serial.Serial,
    reader: MipReader,
    pin: int,
    behavior: int,
    name: str,
    timeout_s: float,
) -> None:
    payload = bytes((MIP_FUNCTION_WRITE, pin, GPIO_FEATURE_UART, behavior, GPIO_PIN_MODE_NONE))
    run_step(ser, reader, DESC_3DM_CMD, CMD_3DM_GPIO_CONFIG, payload, name, timeout_s)


def run_interface_control(
    ser: serial.Serial,
    reader: MipReader,
    interface: int,
    incoming_protocols: int,
    outgoing_protocols: int,
    name: str,
    timeout_s: float,
) -> None:
    payload = (
        bytes((MIP_FUNCTION_WRITE, interface))
        + incoming_protocols.to_bytes(4, "big")
        + outgoing_protocols.to_bytes(4, "big")
    )
    run_step(ser, reader, DESC_SYSTEM_CMD, CMD_SYSTEM_INTERFACE_CONTROL, payload, name, timeout_s)


def make_user_heading_initialization_payload(heading_deg: float) -> bytes:
    return (
        bytes((MIP_FUNCTION_WRITE, 0x00, 0x01, 0x00))
        + struct.pack(">fff", heading_deg, 0.0, 0.0)
        + struct.pack(">fff", 0.0, 0.0, 0.0)
        + struct.pack(">fff", 0.0, 0.0, 0.0)
        + bytes((0x02,))
    )


def run_initialization_user_heading(
    ser: serial.Serial,
    reader: MipReader,
    heading_deg: float | None,
    save_init_config: bool,
    timeout_s: float,
) -> None:
    if heading_deg is None:
        return
    run_step(
        ser,
        reader,
        DESC_FILTER_CMD,
        CMD_FILTER_INITIALIZATION_CONFIGURATION,
        make_user_heading_initialization_payload(heading_deg),
        "set_initialization_user_heading",
        timeout_s,
    )
    if save_init_config:
        run_step(
            ser,
            reader,
            DESC_FILTER_CMD,
            CMD_FILTER_INITIALIZATION_CONFIGURATION,
            bytes((MIP_FUNCTION_SAVE,)),
            "save_initialization_user_heading",
            timeout_s,
        )


def run_filter_controls(
    ser: serial.Serial,
    reader: MipReader,
    reset_filter: bool,
    run_filter: bool,
    timeout_s: float,
) -> None:
    if reset_filter:
        run_step(ser, reader, DESC_FILTER_CMD, CMD_FILTER_RESET_FILTER, b"", "filter_reset", timeout_s)
        time.sleep(0.2)

    if run_filter:
        run_step(ser, reader, DESC_FILTER_CMD, CMD_FILTER_RUN, b"", "filter_run", timeout_s)


def run_step(
    ser: serial.Serial,
    reader: MipReader,
    descriptor_set: int,
    field_descriptor: int,
    payload: bytes,
    name: str,
    timeout_s: float,
) -> None:
    print(
        json.dumps(
            {
                "event": "send",
                "step": name,
                "descriptor_set": f"0x{descriptor_set:02X}",
                "field": f"0x{field_descriptor:02X}",
                "payload_hex": payload.hex(),
            },
            sort_keys=True,
        )
    )
    send_command(ser, reader, descriptor_set, field_descriptor, payload, timeout_s=timeout_s)
    print(json.dumps({"event": "ack", "step": name}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure CV7 for Teensy MIP external GNSS aiding.")
    parser.add_argument("--port", help="CV7 COM port, for example COM13. Auto-detect if omitted.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--pps-gpio-pin", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument(
        "--configure-uart2-mip-on-gpio",
        action="store_true",
        help="Configure GPIO pins as UART2 RX/TX and enable incoming MIP on UART2.",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Skip GPIO/UART/aiding setup and only send requested filter control commands.",
    )
    parser.add_argument("--uart2-rx-gpio-pin", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--uart2-tx-gpio-pin", type=int, default=3, choices=(1, 2, 3, 4))
    parser.add_argument(
        "--no-uart2-tx",
        dest="configure_uart2_tx",
        action="store_false",
        help="Do not configure a UART2 TX pin. CV7 can still receive MIP aiding, but Teensy will not receive ACKs.",
    )
    parser.add_argument("--enable-external-heading", action="store_true")
    parser.add_argument(
        "--init-heading-deg",
        type=float,
        help=(
            "Set EKF initialization to automatic position/velocity/pitch/roll "
            "with this user heading in degrees."
        ),
    )
    parser.add_argument(
        "--save-init-config",
        action="store_true",
        help="Save --init-heading-deg initialization configuration to startup memory.",
    )
    parser.add_argument(
        "--reset-filter",
        action="store_true",
        help="Send the CV7 EKF Reset Filter command after configuration.",
    )
    parser.add_argument(
        "--run-filter",
        action="store_true",
        help="Send the CV7 EKF Run Filter command after configuration.",
    )
    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not save settings to startup memory.")
    parser.add_argument("--no-idle", action="store_true", help="Do not send Set To Idle before configuring.")
    parser.set_defaults(save=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        configure_cv7(args)
    except MipError as exc:
        print(json.dumps({"event": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "event": "done",
                "pps_source": "GPIO",
                "pps_gpio_pin": args.pps_gpio_pin,
                "filter_only": args.filter_only,
                "uart2_mip_on_gpio": args.configure_uart2_mip_on_gpio,
                "uart2_rx_gpio_pin": args.uart2_rx_gpio_pin if args.configure_uart2_mip_on_gpio else None,
                "uart2_tx_gpio_pin": (
                    args.uart2_tx_gpio_pin
                    if args.configure_uart2_mip_on_gpio and args.configure_uart2_tx
                    else None
                ),
                "gnss_pos_vel_aiding": True if not args.filter_only else None,
                "external_heading_aiding": args.enable_external_heading,
                "init_heading_deg": args.init_heading_deg,
                "saved_init_config": args.save_init_config and args.init_heading_deg is not None,
                "reset_filter": args.reset_filter,
                "run_filter": args.run_filter,
                "saved": args.save and not args.filter_only,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
