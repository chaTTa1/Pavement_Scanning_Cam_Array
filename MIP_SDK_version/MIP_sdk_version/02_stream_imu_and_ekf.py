"""
3DM-CV7-INS streamer for NVIDIA Jetson AGX Orin Nano.
Uses the official MicroStrain MSCL Python wheel.

Streams BOTH:
  - All raw / scaled IMU channels  (descriptor set 0x80, CLASS_AHRS_IMU)
  - All EKF estimation channels    (descriptor set 0x82, CLASS_ESTFILTER)

Writes two CSV files for offline analysis and prints throughput once per second.

Install:
    python3 -m venv ~/cv7env && source ~/cv7env/bin/activate
    pip install python-mscl        # or install official wheel/.deb from GitHub

Run:
    python 02_stream_imu_and_ekf.py
"""

import csv
import signal
import sys
import time

import mscl

# ---------------- USER SETTINGS ----------------
PORT           = "/dev/ttyACM0"   # or "/dev/microstrain" with the udev rule
BAUD           = 115200           # ignored on USB CDC but MSCL requires it
IMU_RATE_HZ    = 500              # CV7 base rate = 1000 Hz
FILTER_RATE_HZ = 500
IMU_CSV        = "cv7_imu.csv"
FILTER_CSV     = "cv7_filter.csv"
# -----------------------------------------------


def enable_all_channels(node, klass, rate_hz):
    """Subscribe to every channel field the firmware exposes for this class."""
    if not node.features().supportsCategory(klass):
        print(f"  class 0x{klass:02X} not supported, skipping")
        return 0
    supported = node.features().supportedChannelFields(klass)
    chans = mscl.MipChannels()
    for field in supported:
        chans.append(mscl.MipChannel(field, mscl.SampleRate.Hertz(rate_hz)))
    node.setActiveChannelFields(klass, chans)
    return len(chans)


def main():
    print(f"Opening {PORT} ...")
    conn = mscl.Connection.Serial(PORT, BAUD)
    node = mscl.InertialNode(conn)

    if not node.ping():
        print("ERROR: device did not respond to ping. Check cable / port / permissions.")
        sys.exit(1)

    print(f"Connected: {node.modelName()}  SN={node.serialNumber()}  FW={node.firmwareVersion().str()}")

    # Stop streams, reach a known state before reconfiguring.
    node.setToIdle()

    print("Configuring channels...")
    n_imu = enable_all_channels(node, mscl.MipTypes.CLASS_AHRS_IMU,  IMU_RATE_HZ)
    n_flt = enable_all_channels(node, mscl.MipTypes.CLASS_ESTFILTER, FILTER_RATE_HZ)
    print(f"  IMU    : {n_imu} channels @ {IMU_RATE_HZ} Hz")
    print(f"  Filter : {n_flt} channels @ {FILTER_RATE_HZ} Hz")

    # Enable data streams for both classes.
    node.enableDataStream(mscl.MipTypes.CLASS_AHRS_IMU)
    node.enableDataStream(mscl.MipTypes.CLASS_ESTFILTER)

    # Bring the device out of idle.
    node.resume()
    print("Streaming. Ctrl+C to stop.\n")

    # CSV writers (header captured from the first packet of each class).
    imu_f = open(IMU_CSV,    "w", newline="")
    flt_f = open(FILTER_CSV, "w", newline="")
    imu_w = csv.writer(imu_f)
    flt_w = csv.writer(flt_f)
    imu_header_done = False
    flt_header_done = False

    n_imu_pkts = 0
    n_flt_pkts = 0
    t0 = time.time()
    last_print = t0

    def shutdown(*_):
        print("\nStopping ...")
        try:
            node.setToIdle()
        except Exception:
            pass
        imu_f.close()
        flt_f.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        # Block up to 500 ms; collect up to 1000 packets per call.
        packets = node.getDataPackets(500, 1000)

        for pkt in packets:
            ts = pkt.collectedTimestamp().nanoseconds() * 1e-9 - t0
            row = {"t_s": f"{ts:.6f}"}
            for pt in pkt.data():
                try:
                    row[pt.channelName()] = pt.as_float()
                except Exception:
                    row[pt.channelName()] = pt.as_string()

            ds = pkt.descriptorSet()
            if ds == mscl.MipTypes.CLASS_AHRS_IMU:
                if not imu_header_done:
                    imu_w.writerow(row.keys())
                    imu_header_done = True
                imu_w.writerow(row.values())
                n_imu_pkts += 1
            elif ds == mscl.MipTypes.CLASS_ESTFILTER:
                if not flt_header_done:
                    flt_w.writerow(row.keys())
                    flt_header_done = True
                flt_w.writerow(row.values())
                n_flt_pkts += 1

        now = time.time()
        if now - last_print >= 1.0:
            dt = now - last_print
            print(f"[{now - t0:7.1f}s] "
                  f"IMU {n_imu_pkts:7d} pkt  "
                  f"FILTER {n_flt_pkts:7d} pkt  "
                  f"(rate ~ {n_imu_pkts / (now - t0):6.1f} / {n_flt_pkts / (now - t0):6.1f} Hz)")
            last_print = now


if __name__ == "__main__":
    main()
