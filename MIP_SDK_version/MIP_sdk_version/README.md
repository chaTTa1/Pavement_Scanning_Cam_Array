# Teensy SBF to MicroStrain MIP Bridge

This PlatformIO project runs on a Teensy 4.1. It reads Septentrio SBF GNSS data from a simpleRTK3B / mosaic receiver, converts the selected GNSS blocks into official MicroStrain MIP external-aiding packets, and sends them to a 3DM-CV7/GV7 INS.

The application entry point is [`src/main.cpp`](src/main.cpp). The vendored MicroStrain MIP SDK is under [`lib/mip_sdk`](lib/mip_sdk).

## Current Behavior

- Reads Septentrio SBF from `Serial1` at 115200 baud.
- Parses `PVTGeodetic`, `PosCovGeodetic`, `VelCovGeodetic`, `AttEuler`, and `AttCovEuler`.
- Sends MIP external aiding from `Serial2 TX` / Teensy pin 8 at 115200 baud.
- Mirrors GPS PPS from Teensy pin 2 to Teensy pin 3 for the CV7 PPS input.
- Flashes the Teensy built-in LED on pin 13 for 200 ms on every PPS rising edge.
- Uses MIP `EXTERNAL_TIME` timestamps, with `nanoseconds` derived from GPS week + GPS TOW.
- Sends GPS Time Update commands to the CV7 so PPS can be associated with GPS week/TOW.
- In debug mode, prints readable USB diagnostics instead of binary MIP on USB.

Current key defaults in `src/main.cpp`:

```cpp
static constexpr bool DEBUG_MODE = true;
static constexpr bool MIRROR_MIP_TO_USB = !DEBUG_MODE;
static constexpr bool PRINT_STATUS_TO_USB = DEBUG_MODE;

static constexpr mip_time_timebase AIDING_TIMEBASE =
    MIP_TIME_TIMEBASE_EXTERNAL_TIME;
static constexpr bool USE_GPS_TIME_NANOSECONDS = true;
```

## Hardware Connections

All UART and PPS signals must be 3.3 V logic and all devices must share ground.

Minimum runtime wiring:

| Signal | Teensy 4.1 pin | Connected device signal | Direction |
| --- | ---: | --- | --- |
| GNSS SBF UART | Pin 0, `Serial1 RX` | GNSS UART TX / simpleRTK3B `TX1` | GNSS to Teensy |
| GNSS PPS | Pin 2, `PPS_IN_PIN` | GNSS 1 Hz PPS output | GNSS to Teensy |
| MIP UART TX | Pin 8, `Serial2 TX` | CV7 UART RX | Teensy to CV7 |
| PPS output | Pin 3, `PPS_OUT_IMU_PIN` | CV7 PPS input | Teensy to CV7 |
| Ground | Teensy GND | GNSS GND and CV7 GND | Common ground |
| CV7 USB | Host PC | CV7 USB | Read IMU output |

Optional but recommended during first CV7 configuration:

| Signal | Teensy 4.1 pin | Connected device signal | Direction |
| --- | ---: | --- | --- |
| MIP UART RX | Pin 7, `Serial2 RX` | CV7 UART TX | CV7 to Teensy |

The CV7 can receive realtime POS/VEL aiding from Teensy with only the one-way line `Teensy pin 8 -> CV7 RX`. However, without `CV7 TX -> Teensy pin 7`, Teensy cannot read ACK/NACK replies for startup configuration commands. In that case the realtime aiding packets still transmit, but `[init] ACK` messages cannot be trusted.

Do not power the CV7 from the Teensy 3.3 V rail. Power/read the CV7 through its own USB or approved power input.

## Data Flow

```text
GPS SBF + PPS
    -> Teensy 4.1
        -> parse PVT / PosCov / VelCov
        -> convert to MIP POS LLH and VEL NED
        -> send MIP on pin 8 to CV7 RX
        -> forward PPS on pin 3 to CV7 PPS input

CV7
    -> uses PPS + GPS Time Update + MIP POS/VEL aiding
    -> outputs fused IMU/filter data through CV7 USB to Windows
```

## Serial Configuration

| Interface | Baud | Purpose |
| --- | ---: | --- |
| `Serial` USB | 115200 | debug text when `DEBUG_MODE=true`; binary MIP mirror when `DEBUG_MODE=false` |
| `Serial1` | 115200 | incoming Septentrio SBF from GNSS |
| `Serial2` | 115200 | outgoing MIP packets to the CV7, and optional CV7 replies |

Use USB serial monitor only for readable status when `DEBUG_MODE=true`. If `DEBUG_MODE=false`, the Teensy USB stream is binary MIP data for capture scripts, not text.

## GPS Receiver Output Setup

Use Septentrio RxControl to configure the SBF stream on the GNSS receiver.

Required SBF stream at 10 Hz:

```text
sso, Stream1, COM1, PVTGeodetic+PosCovGeodetic+VelCovGeodetic+AttEuler+AttCovEuler+ReceiverStatus+QualityInd+NTRIPClientStatus, msec100
```

Required PPS output:

```text
spps, sec1, Low2High, 0.0, GPS, 60, 1.0
```

Then save the receiver configuration:

```text
exeCopyConfigFile, Current, Boot
```

Notes:

- `msec100` gives a 10 Hz SBF epoch rate.
- The Teensy requires `PVTGeodetic`, `PosCovGeodetic`, and `VelCovGeodetic` for POS/VEL aiding.
- `AttEuler` and `AttCovEuler` are parsed for heading, but heading is not expected unless the GNSS receiver is configured and producing valid dual-antenna attitude.
- The PPS command uses GPS time as the PPS reference. This matters because Teensy sends GPS week/TOW timestamps to the CV7.

## CV7 Requirements

The CV7 side must be configured so the UART and PPS inputs match the Teensy output:

- CV7 UART connected to Teensy pin 8 must be set to `115200, 8N1`.
- CV7 PPS input must receive Teensy pin 3.
- CV7 ground must be common with Teensy ground.
- External GNSS / aiding input must be enabled.
- GNSS position/velocity aiding must be enabled.
- PPS time sync must be enabled.
- The CV7 must accept GPS Time Update commands so it can associate PPS with GPS week/TOW.

Teensy sends these setup commands on boot:

- Aiding Frame Configuration, command set `0x13`, field `0x01`.
- Aiding Measurement Enable, filter command set `0x0D`, field `0x50`.
- Save Device Settings, 3DM command set.

If CV7 TX is not connected back to Teensy pin 7, these commands may still be sent, but Teensy cannot confirm ACKs. For first-time setup, connect both UART directions or configure/save the CV7 through USB/MIP Monitor.

## MIP Timebase

This firmware currently sends aiding measurements with:

```text
timebase = 2 = EXTERNAL_TIME
nanoseconds = GPS week and GPS TOW converted to nanoseconds
```

This means the CV7 should interpret each POS/VEL measurement at the GPS epoch carried by the original SBF blocks, not at UART arrival time.

The code also sends GPS Time Update commands:

```text
0x01,0x72 TIME_OF_WEEK
0x01,0x72 WEEK_NUMBER
```

These updates are needed because PPS only marks the fractional second edge; the CV7 also needs whole GPS week/TOW information.

Important: With `USE_GPS_TIME_NANOSECONDS=true`, the firmware will not send aiding packets when SBF TOW is invalid. Sending `EXTERNAL_TIME` with a fake timestamp would mislead the CV7.

## MIP Output Format

Realtime output is binary MicroStrain MIP, produced with the official MIP SDK packet builder.

The code sends these realtime aiding fields:

| Descriptor set | Field descriptor | Official command | Contents |
| --- | --- | --- | --- |
| `0x13` | `0x22` | LLH Position | `time`, `frame_id`, latitude, longitude, height, N/E/U uncertainty, valid flags |
| `0x13` | `0x29` | NED Velocity | `time`, `frame_id`, N/E/D velocity, N/E/D uncertainty, valid flags |
| `0x13` | `0x31` | True Heading | `time`, `frame_id`, heading, heading uncertainty, valid flag |
| `0x01` | `0x72` | GPS Time Update | GPS TOW seconds and GPS week number |

Velocity conversion:

```cpp
cmd.velocity[0] = pvt.vn;   // North
cmd.velocity[1] = pvt.ve;   // East
cmd.velocity[2] = -pvt.vu;  // Down = -Up
```

Valid flags:

- `0x0007` for POS means latitude, longitude, and height are valid.
- `0x0007` for VEL means N/E/D velocity components are valid.
- `0x0000` means the packet is still emitted but the measurement should be treated as invalid.

## Debug Output

With `DEBUG_MODE=true`, the Teensy USB serial monitor prints status once per second.

Healthy GPS and MIP output should look like:

```text
PPS rate        : 1 /s
SBF rates       : PVT=10 PosCov=10 VelCov=10 Att=10 AttCov=10 /s
MIP rate        : 20 /s
Serial2 fail    : 0
valid PVT/PCV/VCV: 1 / 1 / 1
Aiding flags P/V: 0x7 / 0x7
TOW PVT/PCV/VCV : same / same / same
```

Interpretation:

- `PPS rate = 1 /s`: Teensy sees GPS PPS on pin 2.
- `SBF rates = 10 /s`: Teensy sees the expected GNSS SBF blocks.
- `MIP rate = 20 /s`: Teensy is sending 10 Hz POS and 10 Hz VEL.
- `Serial2 fail = 0`: Teensy successfully wrote bytes to Serial2.
- Matching `TOW PVT/PCV/VCV`: position, position covariance, and velocity covariance are from the same GNSS epoch.

If no GPS antenna/fix is present, SBF may still arrive but `valid_flags` can be `0x0000`, or packets may be skipped if TOW is invalid.

## ESP32 Validation

The companion ESP32 reader under:

```text
ESP_read_MIP/main_MIP_read
```

can be used to sniff Teensy pin 8 before connecting the CV7.

Typical good output:

```text
POS frame=1 flags=0x0007 valid(lat/lon/h)=yes/yes/yes ... timebase=2 ns=1463857431400000000
VEL frame=1 flags=0x0007 valid(N/E/D)=yes/yes/yes ... timebase=2 ns=1463857431400000000
STATS rx_bytes=... packets=... invalid=0 decode_errors=0 pos=... vel=...
```

Expected observations:

- `timebase=2`: Teensy is using `EXTERNAL_TIME`.
- `ns` increases by `100000000` at 10 Hz.
- POS and VEL from the same epoch have the same `ns`.
- `wrong_set` may increase because the ESP32 reader is primarily decoding aiding set `0x13`, while Teensy also sends base set `0x01` GPS Time Update packets.

## Input SBF Blocks

| SBF block | ID | Used for |
| --- | ---: | --- |
| `PVTGeodetic` | 4007 | GPS week/TOW, mode, latitude, longitude, height, velocity N/E/U, satellite count |
| `PosCovGeodetic` | 5906 | position uncertainty, converted from variance to 1-sigma N/E/U |
| `VelCovGeodetic` | 5908 | velocity uncertainty, converted from variance to 1-sigma N/E/U |
| `AttEuler` | 5938 | dual-antenna heading, pitch, roll |
| `AttCovEuler` | 5939 | heading/pitch/roll uncertainty, converted from variance to 1-sigma |

## Aiding Frame And Lever Arm

The code defines one external aiding frame:

```cpp
static constexpr uint8_t AIDING_FRAME_ID = 1;
static constexpr float ANTENNA_LEVER_ARM[3] = {0.0f, 0.0f, 0.0f};
```

`ANTENNA_LEVER_ARM` should be updated to the measured GNSS antenna position relative to the IMU body frame before field use. The same `AIDING_FRAME_ID` is included in every external POS/VEL/heading aiding command.

## Capturing Binary Output

Set:

```cpp
static constexpr bool DEBUG_MODE = false;
```

Then USB mirrors the generated MIP stream as binary. Use:

```powershell
.\test_recording\record_mip.ps1 -Port COMx -Baud 115200 -Duration 10
```

Do not use normal serial monitor text mode while capturing binary MIP.

## Building With PlatformIO

From this directory:

```powershell
pio run
pio run -t upload
pio device monitor -b 115200
```

The project is configured for Teensy 4.1:

```ini
[env:teensy41]
platform = teensy
board = teensy41
framework = arduino
```

## Quick Bring-Up Checklist

1. Build and upload Teensy firmware.
2. Confirm GPS receiver outputs required SBF blocks at 10 Hz.
3. Confirm Teensy debug shows `PPS rate : 1 /s`.
4. Confirm Teensy debug shows `MIP rate : 20 /s` and `Serial2 fail : 0`.
5. Optionally sniff Teensy pin 8 with ESP32 and confirm `timebase=2`, `flags=0x0007`, and increasing `ns`.
6. Connect Teensy pin 8 to CV7 RX.
7. Connect Teensy pin 3 to CV7 PPS input.
8. Confirm all grounds are common.
9. Confirm CV7 UART/PPS/external aiding settings.
10. Read fused CV7 output through CV7 USB on the Windows PC.

## Useful Official References

- [MicroStrain MIP Packet Overview](https://s3.amazonaws.com/files.microstrain.com/CV7_INS_Manual/dcp_content/introduction/MIP%20Packet%20Overview.htm)
- [MicroStrain Command Overview](https://s3.amazonaws.com/files.microstrain.com/GV7_INS_Manual/dcp_content/introduction/Command%20Overview.htm)
- [MIP SDK GitHub repository](https://github.com/LORD-MicroStrain/mip_sdk)
- [MIP SDK documentation](https://lord-microstrain.github.io/mip_sdk_documentation/)
- [Aiding Frame Configuration `(0x13,0x01)`](https://s3.amazonaws.com/files.microstrain.com/GV7_INS_Manual/Content/external_content/dcp/Commands/0x13/data/0x01.htm)
- [LLH Position `(0x13,0x22)`](https://s3.amazonaws.com/files.microstrain.com/GV7_INS_Manual/Content/external_content/dcp/Commands/0x13/data/0x22.htm)
- [NED Velocity `(0x13,0x29)`](https://s3.amazonaws.com/files.microstrain.com/GV7_INS_Manual/Content/external_content/dcp/Commands/0x13/data/0x29.htm)
- [True Heading `(0x13,0x31)`](https://s3.amazonaws.com/files.microstrain.com/GV7_User_Manual/external_content/dcp/Commands/0x13/data/0x31.htm)
- [Aiding Measurement Control `(0x0D,0x50)`](https://s3.amazonaws.com/files.microstrain.com/GV7_User_Manual/external_content/dcp/Commands/0x0d/data/0x50.htm)
