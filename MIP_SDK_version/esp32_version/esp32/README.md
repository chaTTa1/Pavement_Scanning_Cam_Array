# ESP32 GPS SBF Connection Guide

This project receives GPS/GNSS SBF binary data on the ESP32 and later converts it to IMU MIP packets.

## Current Firmware Pin Map

The current firmware in `src/main.cpp` uses this mapping:

| Function | ESP32 UART / GPIO | Connect to |
| --- | --- | --- |
| GPS SBF input | `Serial2 RX`, `GPIO25` | GPS `TX` |
| GPS optional command output | `Serial2 TX`, `GPIO26` | GPS `RX` |
| IMU MIP output | `Serial1 TX`, `GPIO32` | IMU `RX` |
| IMU optional reply input | `Serial1 RX`, `GPIO33` | IMU `TX` |
| GPS PPS input | `GPIO27` | GPS `PPS` |
| IMU PPS output | `GPIO14` | IMU `PPS input`, if used |
| Ground | `GND` | GPS and IMU `GND` |

## Recommended UART Wiring

Use a dedicated ESP32 hardware UART for the GPS receiver. Do not use UART0 pins `GPIO1` / `GPIO3`, because they are shared with USB programming and serial monitor.

Recommended pins for ESP-WROVER-KIT:

| GPS / GNSS receiver | ESP32 pin | Notes |
| --- | --- | --- |
| `TX` | `GPIO25` | ESP32 receives SBF data here |
| `RX` | `GPIO26` | Optional, only needed if ESP32 sends config commands to GPS |
| `GND` | `GND` | Required common ground |
| `VCC` | GPS supply input | Use the voltage required by the GPS receiver |
| `PPS` | `GPIO27` | Optional timing pulse input |

UART crossing rule:

```text
GPS TX  -> ESP32 RX
GPS RX  <- ESP32 TX
GPS GND -> ESP32 GND
```

## Voltage And Interface Notes

ESP32 GPIO pins are **3.3 V only**.

- If the GPS UART is 3.3 V TTL, connect it directly.
- If the GPS UART is 5 V TTL, use a level shifter before connecting to ESP32 RX.
- If the GPS output is RS-232, use a MAX3232 or equivalent RS-232-to-TTL converter.
- If the GPS output is RS-422 / RS-485, use a matching differential receiver/transceiver.

Do not connect RS-232 or 5 V signals directly to ESP32 GPIO pins.

## GPS Receiver Settings

Configure the GPS/GNSS receiver UART to output SBF binary messages.

Suggested starting settings:

| Setting | Value |
| --- | --- |
| Baud rate | `115200` |
| Data bits | `8` |
| Parity | `None` |
| Stop bits | `1` |
| Flow control | `None` |
| Output format | `SBF binary` |

If the GPS receiver also outputs NMEA on the same port, disable NMEA or use a separate port so the ESP32 parser receives clean SBF packets.

## ESP32 Code Pin Example

In Arduino/PlatformIO code, use `Serial2` with explicit pins:

```cpp
static constexpr int GPS_RX_PIN = 25;  // Connect to GPS TX
static constexpr int GPS_TX_PIN = 26;  // Connect to GPS RX, optional
static constexpr uint32_t GPS_BAUD = 115200;

void setup() {
    Serial.begin(115200);
    Serial2.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
}
```

If the ESP32 only listens to GPS data and does not configure the receiver, the GPS RX wire and ESP32 TX pin are optional.

## Output Converted MIP Data To IMU

MIP is a byte-oriented serial protocol, so the converted data should be sent to the IMU through another ESP32 hardware UART. The ESP32 UART TX signal is output on a normal GPIO pin, but it is still a UART signal, not a manually toggled GPIO signal.

Recommended ESP32 pins for IMU output:

| ESP32 pin | IMU / MIP device pin | Notes |
| --- | --- | --- |
| `GPIO32` | `RX` | ESP32 sends converted MIP packets to IMU |
| `GPIO33` | `TX` | Optional, ESP32 receives IMU replies / ACKs |
| `GPIO14` | `PPS input` | Optional forwarded GPS PPS |
| `GND` | `GND` | Required common ground |

UART crossing rule:

```text
ESP32 TX -> IMU RX
ESP32 RX <- IMU TX
ESP32 GND -> IMU GND
```

Example with GPS input on `Serial2` and IMU output on `Serial1`:

```cpp
static constexpr int GPS_RX_PIN = 25;  // Connect to GPS TX
static constexpr int GPS_TX_PIN = 26;  // Optional, connect to GPS RX
static constexpr uint32_t GPS_BAUD = 115200;

static constexpr int IMU_RX_PIN = 33;  // Optional, connect to IMU TX
static constexpr int IMU_TX_PIN = 32;  // Connect to IMU RX
static constexpr int PPS_IN_PIN = 27;  // Optional, connect to GPS PPS
static constexpr int PPS_OUT_IMU_PIN = 14;  // Optional, connect to IMU PPS input
static constexpr uint32_t IMU_BAUD = 115200;

void setup() {
    Serial.begin(115200);
    Serial2.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
    Serial1.begin(IMU_BAUD, SERIAL_8N1, IMU_RX_PIN, IMU_TX_PIN);
}

void sendMipToImu(const uint8_t* packet, size_t length) {
    Serial1.write(packet, length);
    Serial1.flush();
}
```

After converting one GPS SBF message into one complete MIP packet, call:

```cpp
sendMipToImu(mip_packet_buffer, mip_packet_length);
```

The IMU UART port must be configured to the same baud rate, data bits, parity, and stop bits as the ESP32. Start with `115200 8N1` unless the IMU port is configured differently.

## IMU Electrical Interface Notes

Before wiring ESP32 directly to the IMU, check the IMU communication interface:

| IMU interface | Connection method |
| --- | --- |
| 3.3 V TTL UART | Direct GPIO UART connection is OK |
| 5 V TTL UART | Use a level shifter before ESP32 RX |
| RS-232 | Use a MAX3232 or equivalent converter |
| RS-422 / RS-485 | Use a matching differential transceiver |

ESP32 pins are **not 5 V tolerant**. Do not connect RS-232, RS-422, RS-485, or 5 V UART signals directly to ESP32 GPIO.

## Pins To Avoid

Avoid these pins unless you know the board wiring:

| ESP32 pin | Reason |
| --- | --- |
| `GPIO1`, `GPIO3` | USB programming / serial monitor |
| `GPIO6` - `GPIO11` | SPI flash |
| `GPIO16`, `GPIO17` | Often used by PSRAM on WROVER modules |
| `GPIO0`, `GPIO2`, `GPIO12`, `GPIO15` | Boot strapping pins |
| `GPIO34` - `GPIO39` | Input-only pins, not suitable for UART TX |

## Quick Bring-Up Checklist

1. Connect `GND` between GPS and ESP32.
2. Confirm GPS signal voltage is safe for ESP32.
3. Connect `GPS TX` to `ESP32 GPIO25`.
4. Optional: connect `GPS RX` to `ESP32 GPIO26`.
5. Configure GPS UART for SBF at `115200 8N1`.
6. Build and upload the ESP32 firmware.
7. Use the serial monitor at `115200` to check debug output.

PlatformIO commands:

```powershell
platformio run
platformio run --target upload
platformio device monitor -b 115200
```
