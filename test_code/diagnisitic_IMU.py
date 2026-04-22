# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:45:48 2026

@author: rqy19
"""

import serial
import struct
import numpy as np
import time

PORT = "COM7"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=0)
buf = bytearray()

last_gravity = np.array([0.0, 0.0, 0.0])
gravity_received = False

raw_count = 0
quat_count = 0
rpy_count = 0
gravity_count = 0
other_count = 0

start = time.time()
DURATION = 10  # seconds

print(f"Reading IMU for {DURATION}s...\n")

while time.time() - start < DURATION:
    if ser.in_waiting:
        buf.extend(ser.read(min(ser.in_waiting, 4096)))

    while len(buf) >= 5:
        # Find sync
        if buf[0] != 0xAA or buf[1] != 0x55:
            buf.pop(0)
            continue

        length = buf[2]
        total = 3 + length + 2  # header + len + payload + CRC

        if len(buf) < total:
            break

        payload = bytes(buf[3:3 + length])
        buf = buf[total:]

        if len(payload) < 4:
            continue

        header_info = struct.unpack("<I", payload[:4])[0]
        cmd_id = header_info & 0x7F
        data = payload[4:]

        # ── GRAVITY (cmd_id 36) ──
        if cmd_id == 36 and len(data) >= 16:
            vals = struct.unpack("<Ifff", data[:16])
            last_gravity = np.array([vals[1], vals[2], vals[3]])
            gravity_received = True
            gravity_count += 1

            if gravity_count <= 3:
                print(f"[GRAVITY] x={vals[1]:.4f} y={vals[2]:.4f} z={vals[3]:.4f} "
                      f"mag={np.linalg.norm(last_gravity):.4f}")

        # ── RAW DATA (cmd_id 41) ──
        elif cmd_id == 41 and len(data) >= 28:
            num_floats = (len(data) - 4) // 4
            fmt = "<I" + ("f" * num_floats)
            vals = struct.unpack(fmt, data[:4 + num_floats * 4])

            accel_raw = np.array([vals[1], vals[2], vals[3]])
            gyro = np.array([vals[4], vals[5], vals[6]])

            raw_count += 1

            if raw_count <= 3:
                mag_raw = np.linalg.norm(accel_raw)
                print(f"\n[RAW ACCEL] ax={accel_raw[0]:.6f} ay={accel_raw[1]:.6f} "
                      f"az={accel_raw[2]:.6f} mag={mag_raw:.6f}")

                if gravity_received:
                    accel_with_gravity = accel_raw + last_gravity
                    mag_reconstructed = np.linalg.norm(accel_with_gravity)
                    print(f"[+ GRAVITY] ax={accel_with_gravity[0]:.6f} "
                          f"ay={accel_with_gravity[1]:.6f} "
                          f"az={accel_with_gravity[2]:.6f} "
                          f"mag={mag_reconstructed:.4f} "
                          f"(should be ~1.0)")
                else:
                    print("[+ GRAVITY] waiting for gravity packet...")

        # ── QUATERNION (cmd_id 32) ──
        elif cmd_id == 32 and len(data) >= 20:
            quat_count += 1
            if quat_count <= 2:
                vals = struct.unpack("<Iffff", data[:20])
                print(f"\n[QUAT] q1={vals[1]:.4f} q2={vals[2]:.4f} "
                      f"q3={vals[3]:.4f} q4={vals[4]:.4f}")

        # ── RPY (cmd_id 35) ──
        elif cmd_id == 35 and len(data) >= 16:
            rpy_count += 1
            if rpy_count <= 2:
                vals = struct.unpack("<Ifff", data[:16])
                print(f"[RPY] roll={vals[1]:.2f} pitch={vals[2]:.2f} yaw={vals[3]:.2f}")

        else:
            other_count += 1

    time.sleep(0.0001)

elapsed = time.time() - start
ser.close()

# ── Summary ──
print(f"\n{'='*60}")
print(f"SUMMARY ({elapsed:.1f}s)")
print(f"{'='*60}")
print(f"RAW packets:     {raw_count:6d} ({raw_count/elapsed:.0f}/s)")
print(f"QUAT packets:    {quat_count:6d} ({quat_count/elapsed:.0f}/s)")
print(f"RPY packets:     {rpy_count:6d} ({rpy_count/elapsed:.0f}/s)")
print(f"GRAVITY packets: {gravity_count:6d} ({gravity_count/elapsed:.0f}/s)")
print(f"Other packets:   {other_count:6d} ({other_count/elapsed:.0f}/s)")
print(f"Total:           {raw_count+quat_count+rpy_count+gravity_count+other_count:6d}")

print(f"\n{'='*60}")
print(f"GRAVITY RECONSTRUCTION CHECK")
print(f"{'='*60}")
if gravity_received:
    print(f"Last gravity vector: ({last_gravity[0]:.4f}, {last_gravity[1]:.4f}, {last_gravity[2]:.4f})")
    print(f"Gravity magnitude:   {np.linalg.norm(last_gravity):.4f} g (should be ~1.0)")
    print(f"✅ Gravity packets received — reconstruction will work")
else:
    print(f"❌ No gravity packets received!")
    print(f"   Enable 'Gravity' in ImuAssistant Output Data settings")

print(f"\n{'='*60}")
print(f"CONFIGURATION CHECK")
print(f"{'='*60}")

if raw_count > 0 and gravity_count > 0:
    print(f"✅ Raw Data + Gravity both available")
    print(f"   Set: ACCEL_INPUT_IS_G = True")
    print(f"   Set: IMU_ACCEL_IS_GRAVITY_FREE = False")
    print(f"   Add gravity to accel before EKF")
elif raw_count > 0 and gravity_count == 0:
    print(f"⚠️  Raw Data available but NO Gravity")
    print(f"   Set: ACCEL_INPUT_IS_G = False")
    print(f"   Set: IMU_ACCEL_IS_GRAVITY_FREE = True")
    print(f"   EKF position prediction will be poor")
else:
    print(f"❌ No raw data received")