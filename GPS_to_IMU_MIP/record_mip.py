#!/usr/bin/env python3
"""Record and summarize MIP packets from a serial port.

Connect the MIP TX line to a USB-TTL adapter RX, then run:
  python record_mip.py --port COM12 --duration 10
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Iterable


MIP_SYNC = b"\x75\x65"


@dataclass
class MipPacket:
    wall_time: float
    index: int
    desc_set: int
    payload: bytes
    checksum_expected: tuple[int, int]
    checksum_actual: tuple[int, int]

    @property
    def checksum_ok(self) -> bool:
        return self.checksum_expected == self.checksum_actual

    @property
    def fields(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        i = 0
        while i + 2 <= len(self.payload):
            field_len = self.payload[i]
            if field_len < 2 or i + field_len > len(self.payload):
                break
            field_desc = self.payload[i + 1]
            out.append((field_desc, field_len - 2))
            i += field_len
        return out


def fletcher16(data: bytes) -> tuple[int, int]:
    c0 = 0
    c1 = 0
    for b in data:
        c0 = (c0 + b) & 0xFF
        c1 = (c1 + c0) & 0xFF
    return c0, c1


def parse_stream(chunks: Iterable[bytes]) -> Iterable[MipPacket]:
    buf = bytearray()
    index = 0

    for chunk in chunks:
        buf.extend(chunk)

        while True:
            sync_pos = buf.find(MIP_SYNC)
            if sync_pos < 0:
                del buf[:-1]
                break
            if sync_pos:
                del buf[:sync_pos]

            if len(buf) < 6:
                break

            desc_set = buf[2]
            payload_len = buf[3]
            packet_len = 2 + 2 + payload_len + 2
            if len(buf) < packet_len:
                break

            packet = bytes(buf[:packet_len])
            del buf[:packet_len]

            payload = packet[4 : 4 + payload_len]
            checksum_expected = (packet[-2], packet[-1])
            checksum_actual = fletcher16(packet[2:-2])

            yield MipPacket(
                wall_time=time.time(),
                index=index,
                desc_set=desc_set,
                payload=payload,
                checksum_expected=checksum_expected,
                checksum_actual=checksum_actual,
            )
            index += 1


def serial_chunks(port: str, baud: int, duration: float, raw_path: str) -> Iterable[bytes]:
    import serial

    deadline = time.monotonic() + duration
    with serial.Serial(port, baud, timeout=0.1) as ser, open(raw_path, "wb") as raw:
        while time.monotonic() < deadline:
            chunk = ser.read(4096)
            if not chunk:
                continue
            raw.write(chunk)
            raw.flush()
            yield chunk


def default_prefix() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"mip_capture_{stamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Record MIP serial output and verify checksums.")
    parser.add_argument("--port", required=True, help="USB-TTL serial port, for example COM12")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--duration", type=float, default=10.0, help="Capture duration in seconds")
    parser.add_argument("--out-prefix", default=default_prefix(), help="Output filename prefix")
    args = parser.parse_args()

    raw_path = args.out_prefix + ".bin"
    csv_path = args.out_prefix + ".csv"

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    packets = 0
    bad_checksums = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "wall_time",
                "packet_index",
                "desc_set",
                "payload_len",
                "checksum_ok",
                "checksum_expected",
                "checksum_actual",
                "field_count",
                "fields",
            ]
        )

        for pkt in parse_stream(serial_chunks(args.port, args.baud, args.duration, raw_path)):
            packets += 1
            bad_checksums += 0 if pkt.checksum_ok else 1
            fields = " ".join(f"0x{desc:02X}:{length}" for desc, length in pkt.fields)
            writer.writerow(
                [
                    f"{pkt.wall_time:.6f}",
                    pkt.index,
                    f"0x{pkt.desc_set:02X}",
                    len(pkt.payload),
                    "yes" if pkt.checksum_ok else "no",
                    f"0x{pkt.checksum_expected[0]:02X} 0x{pkt.checksum_expected[1]:02X}",
                    f"0x{pkt.checksum_actual[0]:02X} 0x{pkt.checksum_actual[1]:02X}",
                    len(pkt.fields),
                    fields,
                ]
            )

    print(f"Raw capture: {raw_path}")
    print(f"CSV summary: {csv_path}")
    print(f"Packets: {packets}")
    print(f"Bad checksums: {bad_checksums}")
    return 0 if packets > 0 and bad_checksums == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
