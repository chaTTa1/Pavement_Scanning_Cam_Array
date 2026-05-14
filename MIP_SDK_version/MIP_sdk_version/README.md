# Teensy SBF to MicroStrain MIP Bridge

This PlatformIO project runs on a Teensy 4.1 and converts Septentrio SBF GNSS output from a simpleRTK3B Heading / Mosaic-H receiver into official MicroStrain MIP external-aiding commands for a CV7/GV7 INS.

The application entry point is [`src/main.cpp`](src/main.cpp). The vendored MicroStrain MIP SDK is under [`lib/mip_sdk`](lib/mip_sdk).

## Hardware Connections

All serial and PPS signals must use 3.3 V logic and share a common ground.

| Signal | Teensy 4.1 pin | Connected device pin / signal | Direction |
| --- | ---: | --- | --- |
| GNSS SBF UART | Pin 0, `Serial1 RX` | simpleRTK3B Heading Arduino rail `TX1` | GNSS to Teensy |
| MIP UART TX | Pin 8, `Serial2 TX` | CV7-INS pin 4, `RxD` main UART | Teensy to IMU |
| MIP UART RX | Pin 7, `Serial2 RX` | CV7-INS pin 5, `TxD` main UART | IMU to Teensy |
| PPS input GPIO | Pin 2, `PPS_IN_PIN` | GNSS 1 Hz PPS output | GNSS to Teensy |
| PPS output GPIO | Pin 3, `PPS_OUT_IMU_PIN` | CV7/GV7 PPS input | Teensy to IMU |
| IMU power | Teensy 3.3 V | CV7-INS pin 3, `Vin` | Power |
| Ground | Teensy GND | CV7-INS pin 8, `GND` | Common ground |
| USB | Teensy USB | Host PC | programming / binary capture |

The simpleRTK3B IOREF switch should be set to 3.3 V. The code comment also assumes the simpleRTK3B XBee socket is used by a WiFi NTRIP Master on COM2 and that the receiver is powered independently by USB-C.

## Serial Configuration

| Interface | Baud | Purpose |
| --- | ---: | --- |
| `Serial` USB | 115200 | binary MIP mirror or ASCII status, depending on `MIRROR_MIP_TO_USB` |
| `Serial1` | 115200 | incoming Septentrio SBF from GNSS |
| `Serial2` | 115200 | outgoing/incoming MIP packets to the MicroStrain INS |

Current defaults:

```cpp
static constexpr bool MIRROR_MIP_TO_USB = true;
static constexpr bool PRINT_STATUS_TO_USB = !MIRROR_MIP_TO_USB;
```

With this setting, USB output is a clean binary MIP capture stream. Set `MIRROR_MIP_TO_USB` to `false` if you want the human-readable status report instead.

## GPS Receiver Output Setup

Use the Septentrio GPS RxControl application to configure the SBF output stream. In RxControl, use `sso` to set:

- Stream: `Stream1`
- Port: `COM1`
- Blocks: `PVTGeodetic+PosCovGeodetic+VelCovGeodetic+AttEuler+AttCovEuler+ReceiverStatus+QualityInd+NTRIPClientStatus`
- Interval: `msec100`

The command form is:

```text
sso, Stream1, COM1, PVTGeodetic+PosCovGeodetic+VelCovGeodetic+AttEuler+AttCovEuler+ReceiverStatus+QualityInd+NTRIPClientStatus, msec100
```

Configure the PPS output with `spps`:

```text
spps, sec1, Low2High, 0.0, GPS, 60, 1.0
```

This PPS configuration means:

- `sec1`: 1 Hz PPS output, one pulse every 1 second.
- `Low2High`: trigger on the rising edge.
- `0.0`: no cable delay compensation.
- `GPS`: use GPS time as the time reference.
- `60`: maximum synchronization age is 60 seconds.
- `1.0`: PPS pulse width is 1.0 ms.

The 1.0 ms pulse width follows the MicroStrain recommendation: the PPS pulse should be at least 100 us wide, and 1 ms or wider is recommended.

After confirming the live output is correct, save the current receiver configuration to onboard boot storage:

```text
exeCopyConfigFile, Current, Boot
```

`msec100` configures a 10 Hz output rate. The bridge uses `PVTGeodetic`, `PosCovGeodetic`, `VelCovGeodetic`, `AttEuler`, and `AttCovEuler`; the receiver status, quality, and NTRIP status blocks are allowed in the stream and ignored by the firmware parser.

## How The Code Works

1. `setup()` starts USB, GNSS UART, and IMU UART at 115200 baud.
2. GPIO pin 2 is configured as PPS input and GPIO pin 3 as PPS output. The interrupt handler mirrors every PPS edge from pin 2 to pin 3 and counts rising edges.
3. The MIP SDK interface is initialized with callbacks that write to and read from `Serial2`.
4. The code sends one-time IMU configuration commands:
   - Aiding Frame Configuration, Aiding command set `0x13`, field `0x01`.
   - Aiding Measurement Control / Enable, Filter command set `0x0D`, field `0x50`, enabling `GNSS_POS_VEL` and `EXTERNAL_HEADING`.
   - Save Device Settings, 3DM command set, so the settings survive reset.
5. `loop()` continuously reads SBF bytes from `Serial1`.
6. The SBF state machine looks for the `$@` sync bytes, validates the SBF CRC, extracts the block ID, and updates the latest GNSS snapshots.
7. `try_send_mip()` waits until position, position covariance, and velocity covariance are valid and have the same GPS TOW. It skips duplicate epochs and invalid PVT modes.
8. For each valid GNSS epoch, the code serializes official MIP aiding command payloads with the MicroStrain SDK and sends them to the IMU over `Serial2`.
9. If dual-antenna heading is available for the same epoch, it also sends true heading aiding.

## Input SBF Blocks

| SBF block | ID | Used for |
| --- | ---: | --- |
| `PVTGeodetic` | 4007 | GPS week/TOW, mode, latitude, longitude, height, velocity N/E/U, satellite count |
| `PosCovGeodetic` | 5906 | position uncertainty, converted from variance to 1-sigma N/E/U |
| `VelCovGeodetic` | 5908 | velocity uncertainty, converted from variance to 1-sigma N/E/U |
| `AttEuler` | 5938 | dual-antenna heading, pitch, roll |
| `AttCovEuler` | 5939 | heading/pitch/roll uncertainty, converted from variance to 1-sigma |

## MIP Output Format

The realtime output is binary MicroStrain MIP, not CSV or ASCII. Each packet uses the official packet wrapper:

| Byte(s) | Meaning | Value in realtime aiding output |
| --- | --- | --- |
| 0 | `SYNC1` | `0x75` |
| 1 | `SYNC2` | `0x65` |
| 2 | Descriptor set | `0x13` for Aiding commands |
| 3 | Payload length | Sum of all field lengths in the packet |
| 4..n | Payload fields | One MIP field per packet in this code |
| Last 2 | Fletcher checksum | Written by `mip_packet_finalize()` |

Each MIP field begins with this field header:

| Field byte | Meaning |
| --- | --- |
| 0 | Field length, including this length byte and the field descriptor byte |
| 1 | Field descriptor |
| 2..n | Field payload defined by that descriptor |

The code sends these realtime MIP aiding fields:

| Descriptor set | Field descriptor | Official command | Payload values from this code |
| --- | --- | --- | --- |
| `0x13` | `0x22` | LLH Position | `time`, `frame_id`, latitude deg, longitude deg, height m, position uncertainty N/E/U m, valid flags `0x0007` |
| `0x13` | `0x29` | NED Velocity | `time`, `frame_id`, velocity North/East/Down m/s, velocity uncertainty N/E/D m/s, valid flags `0x0007` |
| `0x13` | `0x31` | True Heading | `time`, `frame_id`, heading rad, heading uncertainty rad, valid flag `0x0001` |

The velocity conversion is:

```cpp
cmd.velocity[0] = pvt.vn;   // North
cmd.velocity[1] = pvt.ve;   // East
cmd.velocity[2] = -pvt.vu;  // Down = -Up
```

The timestamp currently uses:

```cpp
static constexpr mip_time_timebase AIDING_TIMEBASE =
    MIP_TIME_TIMEBASE_TIME_OF_ARRIVAL;
static constexpr bool USE_GPS_TIME_NANOSECONDS = false;
```

That means the `mip_time.nanoseconds` field is sent as `0`, and the IMU treats message arrival time as the measurement time. If the CV7/GV7 is configured for external time sync from PPS, switch to `MIP_TIME_TIMEBASE_EXTERNAL_TIME` and set `USE_GPS_TIME_NANOSECONDS` to `true`.

## Official MIP Header Compatibility

The output header is produced by the official SDK calls in `mip_send_command_field()`:

```cpp
mip_packet_create(&packet, packet_buffer, sizeof(packet_buffer), MIP_AIDING_CMD_DESC_SET);
mip_packet_add_field(&packet, field_descriptor, payload, payload_length);
mip_packet_finalize(&packet);
```

This guarantees the standard MIP wrapper:

```text
0x75 0x65 <descriptor_set> <payload_length> <field_length> <field_descriptor> <field_payload...> <checksum_msb> <checksum_lsb>
```

Relevant official documentation:

- [MicroStrain MIP Packet Overview](https://s3.amazonaws.com/files.microstrain.com/CV7_INS_Manual/dcp_content/introduction/MIP%20Packet%20Overview.htm)
- [MicroStrain Command Overview](https://s3.amazonaws.com/files.microstrain.com/GV7_INS_Manual/dcp_content/introduction/Command%20Overview.htm)
- [MIP SDK GitHub repository](https://github.com/LORD-MicroStrain/mip_sdk)
- [MIP SDK documentation](https://lord-microstrain.github.io/mip_sdk_documentation/)
- [Aiding Frame Configuration `(0x13,0x01)`](https://s3.amazonaws.com/files.microstrain.com/GV7_INS_Manual/Content/external_content/dcp/Commands/0x13/data/0x01.htm)
- [LLH Position `(0x13,0x22)`](https://s3.amazonaws.com/files.microstrain.com/GV7_INS_Manual/Content/external_content/dcp/Commands/0x13/data/0x22.htm)
- [NED Velocity `(0x13,0x29)`](https://s3.amazonaws.com/files.microstrain.com/GV7_INS_Manual/Content/external_content/dcp/Commands/0x13/data/0x29.htm)
- [True Heading `(0x13,0x31)`](https://s3.amazonaws.com/files.microstrain.com/GV7_User_Manual/external_content/dcp/Commands/0x13/data/0x31.htm)
- [Aiding Measurement Control `(0x0D,0x50)`](https://s3.amazonaws.com/files.microstrain.com/GV7_User_Manual/external_content/dcp/Commands/0x0d/data/0x50.htm)

## Aiding Frame And Lever Arm

The code defines one external aiding frame:

```cpp
static constexpr uint8_t AIDING_FRAME_ID = 1;
static constexpr float ANTENNA_LEVER_ARM[3] = {0.0f, 0.0f, 0.0f};
```

`send_frame_config()` writes this frame to the IMU using Euler format, no tracking, zero rotation, and the lever arm in meters. Update `ANTENNA_LEVER_ARM` to the measured GNSS antenna position relative to the IMU body/vehicle frame before field use.

The same `AIDING_FRAME_ID` is included in every external position, velocity, and heading aiding command, so the frame ID matches the configured MicroStrain aiding frame.

## Capturing Output

When `MIRROR_MIP_TO_USB` is `true`, use the helper script in [`test_recording/record_mip.ps1`](test_recording/record_mip.ps1) to capture and validate the USB MIP stream:

```powershell
.\test_recording\record_mip.ps1 -Port COMx -Baud 115200 -Duration 10
```

The script writes:

- `*.bin`: raw binary MIP packets exactly as mirrored from the Teensy.
- `*.csv`: packet summary with this header:

```csv
wall_time,packet_index,desc_set,payload_len,checksum_ok,checksum_expected,checksum_actual,field_count,fields
```

The `fields` column is formatted as `<field_descriptor>:<payload_byte_count>`, for example `0x22:49` for an LLH Position payload.

## Building With PlatformIO

The project is configured for Teensy 4.1:

```ini
[env:teensy41]
platform = teensy
board = teensy41
framework = arduino
```

Common commands:

```powershell
pio run
pio run -t upload
pio device monitor -b 115200
```

Use `pio device monitor` only when `MIRROR_MIP_TO_USB` is `false`; otherwise the USB stream is binary MIP data intended for capture.
