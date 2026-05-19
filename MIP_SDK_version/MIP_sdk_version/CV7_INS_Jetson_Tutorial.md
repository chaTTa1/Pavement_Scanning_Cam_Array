# Reading IMU & EKF Data from a MicroStrain 3DM‑CV7‑INS on NVIDIA Jetson AGX Orin Nano

**A step‑by‑step tutorial using Python and the official MSCL `.whl`**

This tutorial walks you through, end‑to‑end, how to:

1. Wire a 3DM‑CV7‑INS to a Jetson AGX Orin Nano over USB
2. Install the official MSCL Python wheel on aarch64 (Jetson)
3. Configure the device for **both** raw IMU output and EKF (Estimation Filter) output at high rate
4. Stream and decode every available channel from both classes simultaneously
5. Log to CSV for later analysis

---

## 0. Reference & version notes

* The CV7‑INS is a tactical‑grade INS with onboard Extended Kalman Filter; IMU and EKF both run up to **1 kHz**[doc5][doc8].
* Internal sensors: 3‑axis accel, 3‑axis gyro, 3‑axis mag, pressure altimeter[doc9].
* MSCL (MicroStrain Communication Library) is the official open‑source API for MicroStrain inertial sensors[doc5]. As of February 2026 the GitHub repository is archived in read‑only mode, but releases up to **v68.x** remain fully functional[doc6][doc7]. MicroStrain recommends MIP SDK for new long‑term projects, but MSCL is still the easiest Python path[doc5].
* `python-mscl` is also published on PyPI (currently up to v67.1.0) with prebuilt wheels for `armv6l/armv7l/aarch64` Linux[doc1][doc3].

---

## 1. Hardware setup

1. Plug the CV7‑INS Micro‑D‑to‑USB cable into the Jetson's USB‑A or USB‑C port (the cable also delivers power: 3.2–5.2 V via USB[doc8]).
2. Verify the device enumerated:

   ```bash
   dmesg | tail -n 20
   ls /dev/ttyACM*
   ```

   You should see `/dev/ttyACM0` (or `ttyACM1` if other CDC devices exist).

3. Give your user permission to access the serial device:

   ```bash
   sudo usermod -a -G dialout $USER
   # log out & log in, or:
   newgrp dialout
   ```

4. (Optional) Create a stable udev symlink so the port is always `/dev/microstrain`:

   ```bash
   sudo tee /etc/udev/rules.d/99-microstrain.rules >/dev/null <<'EOF'
   SUBSYSTEM=="tty", ATTRS{idVendor}=="199b", SYMLINK+="microstrain", MODE="0666"
   EOF
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

---

## 2. Python environment on Jetson

Jetson AGX Orin Nano (JetPack 6.x) ships with Python 3.10. We'll create an isolated venv.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential

python3 -m venv ~/cv7env
source ~/cv7env/bin/activate

python -m pip install --upgrade pip
```

Verify your interpreter version (used to pick the matching wheel below):

```bash
python -c "import sys, platform; print(sys.version); print(platform.machine())"
# expected: Python 3.10.x ... aarch64
```

---

## 3. Install MSCL on aarch64

You have two options. **Try option A first** — it is by far the simplest.

### Option A — Install from PyPI (recommended)

```bash
pip install python-mscl
```

This pulls a precompiled wheel for `aarch64` Linux[doc1][doc3]. Verify:

```bash
python -c "import mscl; print('MSCL version:', mscl.MSCL_VERSION.str())"
```

If you see a version string (e.g. `67.1.0.0`), you are done with installation.

### Option B — Official wheel from GitHub Releases

If PyPI doesn't have a wheel for your exact Python version, grab it from the archived MSCL repo:

1. Open <https://github.com/LORD-MicroStrain/MSCL/releases> on a browser[doc7].
2. Download the matching asset, e.g.
   `python3-mscl_68.1.0_arm64.deb` (Ubuntu 22.04 / Jetpack 6) or `python_mscl-…-cp310-cp310-linux_aarch64.whl`.
3. Install:

   ```bash
   # Wheel
   pip install ./python_mscl-*-cp310-cp310-linux_aarch64.whl

   # …or .deb (installs system‑wide to /usr/lib/python3/dist-packages)
   sudo dpkg -i ./python3-mscl_*_arm64.deb
   ```

4. If you installed the `.deb` system‑wide and want it visible inside your venv:

   ```bash
   python -m venv --system-site-packages ~/cv7env
   ```

Verify:

```bash
python -c "import mscl; print(mscl.MSCL_VERSION.str())"
```

---

## 4. Smoke test: ping the device

Create `01_ping.py`:

```python
import mscl

PORT = "/dev/ttyACM0"          # or "/dev/microstrain" if you set the udev rule
BAUD = 115200                  # CV7 default; USB ignores baudrate but MSCL still wants it

conn = mscl.Connection.Serial(PORT, BAUD)
node = mscl.InertialNode(conn)

print("Ping OK :", node.ping())
print("Model   :", node.modelName())
print("Number  :", node.modelNumber())
print("Serial  :", node.serialNumber())
print("Firmware:", node.firmwareVersion().str())
```

Run:

```bash
python 01_ping.py
```

Expected output:

```
Ping OK : True
Model   : 3DM-CV7-INS
Number  : 6286-...
Serial  : ...
Firmware: 1.x.x
```

If `Ping OK: False`, double‑check the port path and that no other process (SensorConnect, an old script, a ROS driver) is holding the port open.

---

## 5. Understanding the channel model

MSCL exposes three data **classes** that correspond to MIP descriptor sets[doc6][doc10]:

| MSCL constant | MIP descriptor set | Contents |
|---|---|---|
| `MipTypes.CLASS_AHRS_IMU` | `0x80` | Raw / scaled accel, gyro, mag, delta‑theta, delta‑V, ambient pressure, internal timestamp |
| `MipTypes.CLASS_ESTFILTER` | `0x82` | EKF outputs: orientation (quaternion, Euler, DCM), linear accel, angular rate, position, velocity, biases, uncertainties, filter status |
| `MipTypes.CLASS_GNSS` | `0x81` | GNSS receivers (the CV7‑INS uses **external** GNSS aiding, so this class is normally not active onboard) |

A **channel** is one field within a class, addressed by a `CH_FIELD_*` constant such as `CH_FIELD_SENSOR_SCALED_ACCEL_VEC` or `CH_FIELD_ESTFILTER_ESTIMATED_ORIENT_QUATERNION`[doc6]. You tell the node which channels you want and at what rate; the node streams them as MIP packets, and MSCL hands you decoded `MipDataPoint` objects[doc6].

---

## 6. Configure & start streaming (IMU + EKF together)

Below is the **core script** for the tutorial. It:

* Sets the node to idle (clean state before reconfiguring)[doc6]
* Computes per‑class decimation from the device's *base rate* (CV7: 1000 Hz)[doc5][doc10]
* Subscribes to **every IMU channel** at 500 Hz and **every EKF channel** at 500 Hz
* Enables both data streams and resumes the device[doc6]
* Reads packets in a tight loop, demuxes by descriptor set, prints to console, and writes two CSV files

Save as `02_stream_imu_and_ekf.py` (also provided as a downloadable artifact alongside this tutorial).

```python
import csv, time, signal, sys
import mscl

PORT = "/dev/ttyACM0"
BAUD = 115200
IMU_RATE_HZ    = 500       # CV7 base rate is 1000 Hz; 500 Hz keeps USB happy
FILTER_RATE_HZ = 500
IMU_CSV    = "cv7_imu.csv"
FILTER_CSV = "cv7_filter.csv"

# ---------- connect ----------
conn = mscl.Connection.Serial(PORT, BAUD)
node = mscl.InertialNode(conn)
assert node.ping(), "Device did not respond to ping"
print("Connected to", node.modelName(), "SN", node.serialNumber())

# ---------- stop streaming, clean slate ----------
node.setToIdle()

# ---------- helper: enable every supported channel in a class ----------
def enable_all_channels(node, klass, rate_hz):
    if not node.features().supportsCategory(klass):
        print(f"Class {klass} not supported, skipping")
        return
    # Channels the firmware reports as supported for this class
    supported = node.features().supportedChannelFields(klass)
    chans = mscl.MipChannels()
    for field in supported:
        chans.append(mscl.MipChannel(field, mscl.SampleRate.Hertz(rate_hz)))
    node.setActiveChannelFields(klass, chans)
    print(f"  enabled {len(chans)} channels @ {rate_hz} Hz in class 0x{klass:02X}")

print("Configuring IMU channels...")
enable_all_channels(node, mscl.MipTypes.CLASS_AHRS_IMU,   IMU_RATE_HZ)
print("Configuring EKF channels...")
enable_all_channels(node, mscl.MipTypes.CLASS_ESTFILTER,  FILTER_RATE_HZ)

# ---------- enable streams ----------
node.enableDataStream(mscl.MipTypes.CLASS_AHRS_IMU)
node.enableDataStream(mscl.MipTypes.CLASS_ESTFILTER)

# ---------- (optional) reset the EKF so it re-initializes ----------
# node.resetFilter()

node.resume()
print("Streaming. Ctrl+C to stop.\n")

# ---------- open CSVs ----------
imu_f = open(IMU_CSV,    "w", newline="")
flt_f = open(FILTER_CSV, "w", newline="")
imu_w = csv.writer(imu_f)
flt_w = csv.writer(flt_f)

# We don't know the column order ahead of time; capture from first packet
imu_header_done = False
flt_header_done = False

def handle_sigint(sig, frame):
    print("\nStopping...")
    try: node.setToIdle()
    except Exception: pass
    imu_f.close(); flt_f.close()
    sys.exit(0)
signal.signal(signal.SIGINT, handle_sigint)

t0 = time.time()
n_imu = n_flt = 0
last_print = t0

while True:
    # Block up to 500 ms, take up to 1000 packets at a time
    packets = node.getDataPackets(500, 1000)
    for pkt in packets:
        ts = pkt.collectedTimestamp().nanoseconds() * 1e-9 - t0
        desc_set = pkt.descriptorSet()

        # Build a (name -> value) dict for this packet
        row = {"t_s": f"{ts:.6f}"}
        for pt in pkt.data():
            try:
                row[pt.channelName()] = pt.as_float()
            except Exception:
                row[pt.channelName()] = pt.as_string()

        if desc_set == mscl.MipTypes.CLASS_AHRS_IMU:
            if not imu_header_done:
                imu_w.writerow(row.keys());  imu_header_done = True
            imu_w.writerow(row.values()); n_imu += 1
        elif desc_set == mscl.MipTypes.CLASS_ESTFILTER:
            if not flt_header_done:
                flt_w.writerow(row.keys());  flt_header_done = True
            flt_w.writerow(row.values()); n_flt += 1

    # Status print once per second
    now = time.time()
    if now - last_print > 1.0:
        print(f"[{now-t0:7.1f}s] imu pkts: {n_imu:7d}   filter pkts: {n_flt:7d}")
        last_print = now
```

Run it:

```bash
python 02_stream_imu_and_ekf.py
```

You should see something like:

```
Connected to 3DM-CV7-INS SN 6286-...
Configuring IMU channels...
  enabled 9 channels @ 500 Hz in class 0x80
Configuring EKF channels...
  enabled 22 channels @ 500 Hz in class 0x82
Streaming. Ctrl+C to stop.

[    1.0s] imu pkts:     500   filter pkts:     500
[    2.0s] imu pkts:    1000   filter pkts:    1000
```

Two CSV files (`cv7_imu.csv`, `cv7_filter.csv`) are written in the current directory.

---

## 7. What data are you getting?

After running the script above, the CSV column headers tell you exactly which channels are populated. On a typical CV7‑INS firmware you'll see:

### `cv7_imu.csv` (Class `0x80`)

| Column | Units | Meaning |
|---|---|---|
| `scaledAccelX / Y / Z` | g | Factory‑calibrated accelerometer[doc10] |
| `scaledGyroX / Y / Z` | rad/s | Factory‑calibrated gyroscope[doc10] |
| `scaledMagX / Y / Z` | Gauss | Calibrated magnetometer[doc10] |
| `deltaThetaX / Y / Z` | rad | Coning‑compensated integrated rotation |
| `deltaVelX / Y / Z` | g·s | Sculling‑compensated integrated velocity |
| `scaledAmbientPressure` | mbar | Pressure altimeter[doc9] |
| `gpsTimestamp*` | s, week | Internal time reference[doc10] |

### `cv7_filter.csv` (Class `0x82` — the EKF output)

| Column | Units | Meaning |
|---|---|---|
| `estOrientQuaternion_*` | unitless | Body→NED quaternion |
| `estRoll / estPitch / estYaw` | rad | Euler angles |
| `estOrientMatrix_*` | — | 3×3 DCM (body→NED) |
| `estLinearAccelX / Y / Z` | m/s² | Gravity‑removed accel in body frame |
| `estAngularRateX / Y / Z` | rad/s | EKF‑smoothed body rates |
| `estGyroBiasX / Y / Z` | rad/s | Estimated gyro bias[doc6] |
| `estAccelBiasX / Y / Z` | m/s² | Estimated accel bias |
| `estLatitude / estLongitude / estHeightAboveEllipsoid` | rad, m | INS position (with aiding) |
| `estNorthVelocity / estEastVelocity / estDownVelocity` | m/s | NED velocity |
| `estAttitudeUncertQuaternion_*` | — | 1‑σ attitude uncertainty |
| `estPosUncertNorth / East / Down` | m | 1‑σ position uncertainty |
| `estVelUncertNorth / East / Down` | m/s | 1‑σ velocity uncertainty |
| `filterState` | enum | INIT / VERT_GYRO / AHRS / FULL_NAV[doc10] |
| `filterStatusFlags` | bitfield | Health flags |

Channels that depend on external GNSS aiding (position, NED velocity) will report uncertainty until you feed the CV7 with external position/velocity[doc8].

---

## 8. Persist configuration to flash (optional)

If you want the CV7 to come up streaming this exact configuration every time it's powered:

```python
node.saveSettingsAsStartup()
```

To revert to defaults later:

```python
node.loadFactoryDefaultSettings()
```

---

## 9. Performance tips on Jetson

* **USB latency:** `/dev/ttyACM*` is CDC‑ACM. At 1 kHz both streams it's ~60–80 kB/s — trivial — but kernel latency can clump packets. Setting `node.getDataPackets(50, 1000)` (50 ms timeout) gives smoother real‑time behavior at the cost of a tighter Python loop.
* **CPU pinning:** `taskset -c 4 python 02_stream_imu_and_ekf.py` parks the reader on a performance core, useful when other workloads are running.
* **Power mode:** `sudo nvpmodel -m 0 && sudo jetson_clocks` ensures no CPU down‑clocking during long captures.
* **Disk I/O:** for hour‑long captures write to NVMe, not the eMMC. Or buffer to a `queue.Queue` and flush from a writer thread.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ImportError: No module named mscl` | Wheel installed in a different Python; check `which python`, re‑activate venv |
| `RuntimeError: Failed to open the port` | `sudo chmod 666 /dev/ttyACM0` or fix the `dialout` group membership |
| `ping()` returns `False` | Another process owns the port (SensorConnect, ROS, an orphan Python). `sudo fuser /dev/ttyACM0` |
| No filter packets, only IMU | EKF still initializing. Stay still 5–10 s, then move slowly to let it transition to AHRS mode[doc10] |
| `setActiveChannelFields` raises "rate too high" | Your requested rate doesn't divide the base rate cleanly; pick 1000/2/4/5/10/… Hz |
| Random checksum errors in log | USB hub or long cable; plug directly into Jetson, prefer the rear USB‑C |

---

## 11. Where to go next

* Add **external GNSS aiding** with `node.sendExternalGNSSUpdate(...)` to unlock full INS navigation outputs[doc8].
* Use `node.sendExternalHeadingUpdate(...)` if you have a dual‑antenna heading source or magnetometer compass external to the CV7.
* Migrate to **MIP SDK** if you need a C++ deployment or long‑term support beyond MSCL's EOL window[doc5].
* The C++ inertial example in the MSCL repo demonstrates the same patterns (idle / configure / enable / parse) and is a great reference for advanced commands[doc6].

---

### Citations

* MicroStrain 3DM‑CV7‑INS product page & specs[doc5][doc8]
* MicroStrain CV7 product description, internal sensors[doc9]
* MSCL repository archival notice and example code[doc6]
* MSCL API documentation[doc7]
* `python-mscl` PyPI/piwheels availability for ARM Linux[doc1][doc3]
* CV7 example (MIP‑SDK reference for channel descriptors and EKF flow)[doc10]
