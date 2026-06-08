# CV7 INS EKF Reader And Configuration

This folder reads the MicroStrain 3DM-CV7-INS EKF fused output and uses a Teensy 4.1 to convert Septentrio mosaic-H GNSS/SBF data into MicroStrain MIP external aiding commands accepted by the CV7.

Current data path:

```text
Septentrio mosaic-H -> Teensy 4.1 -> 3DM-CV7-INS -> Windows Python CSV
```

The CV7 does not directly read the raw GPS stream in this setup. mosaic-H PVT, velocity, and attitude data first go to the Teensy. The Teensy then sends CV7 external aiding commands. If `CV7_INS_EKF.py` is run with `--gps-port`, it also opens the GPS receiver's NMEA port and writes raw GPS rows to `cv7_gps_raw_*.csv`. The fused EKF position is written to `cv7_ekf_fused_*.csv`.

## Current Hardware Wiring

The CV7 C-Series Connectivity Board uses four signal groups connected to the Teensy:

| CV7 C-Series signal | Teensy 4.1 | Direction | Purpose |
| --- | --- | --- | --- |
| UUT_RX / RxD | Pin 8, Serial2 TX | Teensy -> CV7 | MIP external aiding input |
| UUT_TX / TxD | Pin 7, Serial2 RX | CV7 -> Teensy | CV7 ACK/NACK feedback |
| GPIO1 | Pin 3 | Teensy -> CV7 | Forwarded GPS PPS input |
| GND | GND | common | Common ground |

Teensy to mosaic-H:

| mosaic-H signal | Teensy 4.1 | Purpose |
| --- | --- | --- |
| SBF UART TX | Pin 0, Serial1 RX | mosaic-H SBF data into Teensy |
| 1 Hz PPS output | Pin 2 | PPS input into Teensy |
| GND | GND | Common ground |

Notes:

- The CV7 is powered and connected to Windows through the C-Series USB connection.
- The Teensy can be externally powered by the GPS setup and connected to USB for debugging at the same time, but all grounds must be common.
- Do not power the CV7 from the Teensy 3.3 V pin.
- If the left-side UART header on the C-Series board is used for TTL serial input into the CV7, keep the J4 jumper removed so the onboard RS232 transceiver does not drive or interfere with UUT_RX.

## INS And GPS Installation Requirements

The 3DM-CV7-INS output depends on the physical installation, not only on serial data. Before collecting data, verify these items.

### CV7 INS mounting

- Mount the CV7 rigidly to the vehicle/body. It should not move relative to the GPS antennas after calibration.
- Define the intended vehicle/body frame. For pavement scanning, use a consistent convention such as X forward, Y right, Z down or the convention required by your post-processing.
- Check the CV7 body axes against the vehicle frame. If the CV7 is rotated relative to the vehicle, set the installation/mounting transform in SensorConnect or in the device configuration before using EKF output as vehicle motion.
- Avoid mounting the CV7 on a flexible plate, cable bundle, or vibrating loose bracket. Mechanical vibration or shifting causes the EKF to correct itself whenever GPS aiding arrives.
- Keep the IMU location fixed and document the measurement reference point. The GNSS lever arm must be measured from this IMU/body reference.

### GPS antenna placement

- Place the GNSS antennas with clear sky view. Avoid roof edges, metal obstructions, nearby high-current cables, and camera/computer structures that can cause multipath.
- For mosaic-H dual-antenna heading, both antennas must be rigidly mounted to the same vehicle body as the CV7.
- The main-to-aux antenna baseline must be fixed and measured. Longer, cleaner baselines usually give more stable heading.
- The heading produced by mosaic-H is the direction of the antenna baseline, not automatically the vehicle forward direction. If the antennas are mounted left-right, apply the correct heading offset before using it as CV7 external heading.
- If heading is not valid in RxControl Attitude View, do not enable CV7 external heading as a trusted final heading source.

### Lever arm and offsets

- Measure the vector from the CV7 IMU reference point to the GNSS antenna phase center. Use the same body-frame convention as the CV7 installation.
- If dual antennas are used, document both the main antenna lever arm and the main-to-aux baseline direction.
- Configure the GNSS antenna offset/lever arm in the CV7 installation settings when available. An unconfigured lever arm can make turns look like periodic sideways corrections in the EKF trajectory.
- Recheck the lever arm after moving antennas, changing roof mounts, or changing the CV7 mounting position.

### Time synchronization

- Feed the GPS 1 Hz PPS to CV7 GPIO1 through the Teensy path used in this project.
- CV7 should be configured with `PPS Source = GPIO` and `GPIO1 = PPS Input`.
- Teensy status should show `PPS rate: 1 /s`.
- The POS/VEL/heading aiding messages sent by Teensy should use GPS time/TOW matching the original mosaic-H measurements. Time mismatch is a common reason for a 500 Hz EKF trace to appear as small repeated correction strokes around 10 Hz GPS updates.

### Minimum healthy operating state

Before trusting a log, check:

```text
mosaic-H fix: RTK Float or RTK Fixed
mosaic-H AttEuler: valid, if external heading is used
Teensy PPS rate: 1 /s
Teensy CV7 ACK POS/VEL: increasing
Teensy CV7 NACK total: 0
CV7 filter state: Full Navigation
CV7 GNSS Position/Velocity aiding: enabled and used
CV7 Heading aiding: enabled and used only when real heading is valid
```

## Target CV7 Configuration

Target configuration:

```text
PPS Source: GPIO
GPIO1: PPS Input
MAIN - Main USB or UART: MIP parser enabled
GNSS Position and Velocity Aiding: enabled
External Heading Aiding: enabled only when mosaic-H AttEuler heading is valid
```

Known good state:

```text
GPIO1 feature = PPS, behavior = input
Main interface incoming_protocols = MIP
GNSS position/velocity aiding = true
```

This wiring uses the CV7 C-Series `UUT_RX / UUT_TX` pins. It does not require GPIO2 to be configured as UART2. GPIO2/UART2 is only for an alternate wiring method where the external serial input is connected to GPIO2 pin 9.

## Configure The CV7 From Command Line

On Windows, the same COM port can normally be opened by only one program at a time. Close SensorConnect before running these commands.

Open cmd/PowerShell on Windows, or Terminal on Linux, from the project root folder and run the commands with `python`.

Port examples:

```text
Windows: COM13
Linux  : /dev/ttyACM0, /dev/ttyUSB0, or /dev/serial/by-id/...
```

Base configuration for Teensy POS/VEL external aiding:

```powershell
python IMU_EKF\CV7_config_aiding.py --port COM13 --pps-gpio-pin 1
```

Linux example:

```bash
python IMU_EKF/CV7_config_aiding.py --port /dev/ttyACM0 --pps-gpio-pin 1
```

When mosaic-H dual-antenna heading is valid and the Teensy is sending heading, enable CV7 external heading aiding:

```powershell
python IMU_EKF\CV7_config_aiding.py --port COM13 --pps-gpio-pin 1 --enable-external-heading
```

If no valid heading is available but the CV7 needs to enter Full Navigation while stationary, provide a temporary initial heading:

```powershell
python IMU_EKF\CV7_INS_EKF.py --port COM13 --init-heading-deg 0 --reset-filter --run-filter
```

This is only a runtime initialization, not a real GNSS heading. Once mosaic-H heading is working, do not rely on a fixed `--init-heading-deg 0` as the long-term heading source.

## Check CV7 Status

Capture one configuration and latest EKF/aiding snapshot:

```powershell
python IMU_EKF\CV7_INS_EKF.py --port COM13 --status --configure --rate-hz 10 --status-listen-s 6 --pretty
```

Initialize while stationary and then capture status:

```powershell
python IMU_EKF\CV7_INS_EKF.py --port COM13 --status --configure --rate-hz 10 --status-listen-s 6 --init-heading-deg 0 --reset-filter --run-filter --pretty
```

Important fields:

```text
filter_state_name: full_nav
position_valid: 1
velocity_valid: 1
aid_measurement_summary ... aiding_pos_llh ... enabled + used
aid_measurement_summary ... aiding_vel_ned ... enabled + used
```

If heading is also valid, heading aiding should become enabled/used. In SensorConnect, Heading under Aiding Measurements should also turn on.

## Record 500 Hz EKF CSV

First make sure the CV7 is in Full Navigation. If real heading is not available, run once:

```powershell
python IMU_EKF\CV7_INS_EKF.py --port COM13 --init-heading-deg 0 --reset-filter --run-filter
```

Start 500 Hz logging:

```powershell
python IMU_EKF\CV7_INS_EKF.py --port COM13 --configure --rate-hz 500 --stream-preset csv --summary --print-hz 1 --record-csv --skip-check --expected-ekf-hz 500
```

To record CV7 EKF and external GPS raw NMEA at the same time, use GPS auto-detection:

```powershell
python IMU_EKF\CV7_INS_EKF.py --list-ports
python IMU_EKF\CV7_INS_EKF.py --port COM13 --gps-port auto --configure --rate-hz 500 --stream-preset csv --summary --print-hz 1 --record-csv --skip-check --expected-ekf-hz 500 --expected-gps-hz 10
```

Use the CV7 port for `--port`. `--gps-port auto` scans candidate serial ports and selects the one that is actually streaming NMEA sentences. If `--record-csv` is used and `--gps-port` is omitted, the script also tries GPS auto-detection.

Linux example:

```bash
python IMU_EKF/CV7_INS_EKF.py --port /dev/ttyACM0 --configure --rate-hz 500 --stream-preset csv --summary --print-hz 1 --record-csv --skip-check --expected-ekf-hz 500
```

`--stream-preset csv` requests only the core EKF fields needed for high-rate logging:

```text
filter timestamp
filter status
position LLH
velocity NED
Euler attitude
position uncertainty
velocity uncertainty
Euler uncertainty
```

Output files are saved in the `IMU_EKF` folder:

```text
cv7_ekf_fused_YYYYMMDD_HHMMSS.csv
cv7_gps_raw_YYYYMMDD_HHMMSS.csv
cv7_skip_validation_YYYYMMDD_HHMMSS.csv
```

Notes:

- `cv7_ekf_fused_*.csv` is the main file. It contains fused latitude, longitude, height, NED velocity, roll, pitch, yaw, and uncertainties.
- `cv7_gps_raw_*.csv` contains external GPS NMEA rows when `--gps-port` is used. Without `--gps-port`, it may contain only a header because GPS arrives as Teensy external aiding, not as the CV7 internal GNSS raw stream.
- `cv7_skip_validation_*.csv` only records detected gap/skipping events. A file with only the header means no skipping was detected.

## Record GPS Raw CSV From The Same Run

The CV7 does not output raw GPS rows in the current setup. `CV7_INS_EKF.py` can therefore open the GPS receiver's NMEA serial port at the same time as the CV7 port. The external GPS rows are written into the same timestamped GPS CSV:

```text
cv7_gps_raw_YYYYMMDD_HHMMSS.csv
```

Those rows include host receive time, NMEA UTC time when available, the raw NMEA sentence, and parsed fields from GGA, RMC, VTG, GST, HDT, and ZDA messages.

## Plot EKF/GPS Output

After recording, open interactive 2D and 3D EKF/GPS point plots from the newest CSV file:

```powershell
python IMU_EKF\CV7_plot_ekf_gps.py
```

Or specify a file:

```powershell
python IMU_EKF\CV7_plot_ekf_gps.py --ekf-csv IMU_EKF\cv7_ekf_fused_YYYYMMDD_HHMMSS.csv
```

The script prints the number of EKF and GPS raw points, then opens matplotlib windows. It does not save image files. EKF and GPS samples are shown as points.

If external GPS rows were recorded in the matching `cv7_gps_raw_*.csv`, the plot script uses them automatically. You can also pass a GPS CSV explicitly:

```powershell
python IMU_EKF\CV7_plot_ekf_gps.py --gps-csv IMU_EKF\cv7_gps_raw_YYYYMMDD_HHMMSS.csv
```

Stop logging:

```text
PowerShell / cmd: Ctrl+C
Spyder: interrupt/stop current execution
```

## Spyder Usage

You can open `CV7_INS_EKF.py` directly in Spyder and run it. The `SPYDER / IDE SETTINGS` section near the top of the file is configured for 500 Hz CSV logging:

```python
SPYDER_PORT = "COM13"
SPYDER_CONFIGURE = True
SPYDER_RATE_HZ = 500
SPYDER_STREAM_PRESET = "csv"
SPYDER_RECORD_CSV = True
SPYDER_SKIP_CHECK = True
SPYDER_EXPECTED_EKF_HZ = 500.0
```

To avoid freezing the Spyder console, keep:

```python
SPYDER_SUMMARY_OUTPUT = True
SPYDER_PRINT_HZ = 2.0
```

Do not print all JSON fields at 500 Hz to the Spyder console. The CSV files store the full data; the console should only show a low-rate summary.

You can also call these helper functions in the Spyder Console:

```python
run_cv7_status(port="COM13", init_heading_deg=0, reset_filter=True, run_filter=True)
run_cv7_reader(port="COM13", configure=True, rate_hz=500, stream_preset="csv", record_csv=True)
```

## SensorConnect Checks

In SensorConnect:

1. `System -> Interface Control`
   - Keep Main port as MIP.
   - No UART2/GPIO2 configuration is required for the current `UUT_RX / UUT_TX` wiring.
2. `GPIO`
   - PPS Source: GPIO.
   - GPIO1: PPS, Input.
   - GPIO2 can remain unused for the current wiring.
3. `Estimation Filter -> Aiding Source Enable`
   - Check GNSS Position and Velocity Aiding.
   - Check External Heading Aiding only when the Teensy is sending valid heading.
4. `Status Quickview`
   - State should be Full Navigation.
   - GNSS Position and GNSS Velocity Enabled/Used should be on.
   - If Heading remains gray, the CV7 is not using external heading aiding.

## mosaic-H And Teensy Heading Status

mosaic-H is a dual-antenna heading receiver and can output GNSS attitude while stationary. In RxControl Attitude View, you should see:

```text
Heading: valid angle
Mode: GNSS-based ...
```

In the Teensy status output, a valid heading path looks like:

```text
SBF rates       : PVT=10 PosCov=10 VelCov=10 Att=10 AttCov=10 /s
AttEuler        : TOW=... mode=... err=0 sv=... valid=yes
Heading         : 142.xx deg ... sigma=...
CV7 ACK TIME/HDG: xxx / nonzero
```

If you see:

```text
AttEuler : mode=0 err=1 sv=255 valid=no
Heading  : ovf deg
CV7 ACK TIME/HDG: xxx / 0
```

The Teensy is receiving the AttEuler block, but mosaic-H is marking attitude as invalid. In that case the Teensy will not send heading to the CV7. This is not a CV7 rejection and not a Python read problem. Check:

- Whether RxControl Attitude View is valid.
- Whether AttEuler/AttCovEuler are enabled on the UART stream that the Teensy reads.
- Whether the Aux antenna signal quality is stable.
- Whether the two-antenna baseline is fixed and long enough.
- Whether the antenna baseline direction matches the vehicle/IMU yaw direction.

## Heading Offset Note

mosaic-H heading is the baseline direction from the main antenna to the auxiliary antenna. It is not automatically the vehicle forward direction or the CV7 IMU body yaw. If the antennas are mounted left-right across the vehicle, the mosaic-H heading can be about 90 degrees away from the vehicle forward heading.

Use one of these approaches:

- Configure attitude offset in mosaic-H/RxControl.
- Add/subtract a fixed offset in the Teensy before forwarding heading.

Do not use external heading as the final vehicle heading until the offset is confirmed.

## Common Issues

**Why are there two COM ports, but only COM13 has data?**  
The CV7 USB connection exposes two virtual serial ports. The active MIP main data port is usually one of them. In this setup, COM13 is the active data port.

**Why can Python not read while SensorConnect is open?**  
Windows serial ports are normally exclusive. Close SensorConnect before running Python.

**Why are GNSS Position/Velocity on, but Heading is gray?**  
The CV7 is using POS/VEL aiding, but the Teensy is not sending valid `HEADING_TRUE 0x13/0x31`, or External Heading Aiding is not enabled on the CV7.

**Why can the CV7 enter Full Navigation without heading?**  
`--init-heading-deg 0` can provide a temporary initial heading so the EKF can enter Full Navigation. Real heading should come from mosaic-H or vehicle motion/kinematic alignment.

**Why is GPS 10 Hz while EKF logging is 500 Hz?**  
GPS/mosaic-H aiding is a low-rate correction source. The CV7 EKF can still output the fused state at 500 Hz using IMU propagation.
