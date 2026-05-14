# GPS to IMU MIP Bridge

This PlatformIO project runs on a Teensy 4.1 and converts Septentrio SBF messages from a simpleRTK3B Heading / Mosaic-H receiver into MicroStrain MIP External Aiding packets for a CV7-INS.

The firmware entry point is [`src/main.cpp`](src/main.cpp).

## What the Firmware Does

- Reads SBF data from the GPS receiver on `Serial1` at `115200` baud.
- Parses position, covariance, velocity, heading, and heading covariance SBF blocks.
- Builds MIP External Aiding packets on descriptor set `0x13`.
- Sends the MIP stream to the IMU on `Serial2` at `115200` baud.
- Optionally mirrors the generated MIP binary stream to the Teensy USB serial port for capture and debugging.

The current firmware emits these MIP fields:

| MIP field | Description |
| --- | --- |
| `0x13, 0x62` | External GNSS Time |
| `0x13, 0x16` | External Position LLH |
| `0x13, 0x17` | External Velocity NED |
| `0x13, 0x28` | External Heading True, when heading is valid |

## Hardware Connections

GPS receiver to Teensy:

| GPS / simpleRTK3B Heading | Teensy 4.1 |
| --- | --- |
| Arduino rail `TX1` / `COM1` output | Pin `0`, `Serial1 RX` |
| `GND` | `GND` |

Teensy to CV7-INS:

| Teensy 4.1 | CV7-INS |
| --- | --- |
| Pin `8`, `Serial2 TX` | Pin `4`, `RxD` main UART |
| Pin `7`, `Serial2 RX` | Pin `5`, `TxD` |
| `3.3V` | Pin `3`, `Vin` |
| `GND` | Pin `8`, `GND` |

Make sure the simpleRTK3B IOREF switch is set to `3.3V`.

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

After confirming the live output is correct, save the current receiver configuration to onboard boot storage:

```text
exeCopyConfigFile, Current, Boot
```

`msec100` configures a 10 Hz output rate. The bridge uses `PVTGeodetic`, `PosCovGeodetic`, `VelCovGeodetic`, `AttEuler`, and `AttCovEuler`; the receiver status, quality, and NTRIP status blocks are allowed in the stream and ignored by the firmware parser.

## Build and Upload

Install PlatformIO, connect the Teensy 4.1 over USB, then run these commands from this folder:

```powershell
pio run
pio run --target upload
```

The project target is defined in [`platformio.ini`](platformio.ini):

```ini
[env:teensy41]
platform = teensy
board = teensy41
framework = arduino
monitor_speed = 115200
```

## Capturing the MIP Output

By default, `main.cpp` has:

```cpp
static constexpr bool MIRROR_MIP_TO_USB = true;
```

With this enabled, the Teensy USB serial port outputs a clean binary MIP stream instead of text status messages. Capture and verify it with either script:

```powershell
.\record_mip.ps1 -Port COM12 -Duration 10
```

or:

```powershell
python .\record_mip.py --port COM12 --duration 10
```

Replace `COM12` with the Teensy USB serial port. The scripts write a raw `.bin` capture and a `.csv` summary with packet counts, field descriptions, and checksum status.

If you want human-readable status text on USB instead, set `MIRROR_MIP_TO_USB` to `false`, rebuild, and upload again.

## Expected Operation

The bridge sends one MIP packet per GPS epoch after it has matching `PVTGeodetic`, `PosCovGeodetic`, and `VelCovGeodetic` data for the same GPS time-of-week. Heading is included only when valid `AttEuler` and `AttCovEuler` data are available for that epoch.

For normal operation, verify:

- GPS `COM1` is outputting the configured SBF blocks at `msec100`.
- The GPS position mode is valid, preferably RTK Fixed or RTK Float.
- The Teensy receives GPS SBF data on `Serial1`.
- The CV7-INS receives MIP External Aiding packets from Teensy `Serial2`.
- The MIP capture scripts report packets with valid checksums.
