# main_MIP_read

ESP32-WROVER-E UART sniffer for the Teensy 4.1 MIP output.

## Wiring

Default RX pin:

```text
Teensy Pin 8 / Serial2 TX  -> ESP32 GPIO34
Teensy Pin 3 / PPS output  -> ESP32 GPIO35
Teensy GND                 -> ESP32 GND
```

Do not connect ESP32 TX back to Teensy for this reader. It only listens.

GPIO34 is used by default because ESP32-WROVER modules commonly reserve GPIO16
and GPIO17 for PSRAM. If your board does not expose GPIO34, change
`MIP_UART_RX_GPIO` in `main/main_MIP_read.c`.

The PPS input is `GPIO35` by default. The ESP32 LED output is `GPIO2` by
default and flashes for 200 ms on each PPS rising edge. Change `PPS_RX_GPIO` or
`PPS_LED_GPIO` in `main/main_MIP_read.c` if your board uses different pins.

## Build

Open an ESP-IDF terminal where `idf.py` is available, then run:

```powershell
cd ESP_read_MIP\main_MIP_read
idf.py set-target esp32
idf.py build
idf.py -p COMx flash monitor
```

## Output

The program uses the official MicroStrain MIP SDK parser and aiding command
decoders from the repository. It prints readable lines for:

- `POS`: Aiding Position LLH, descriptor set `0x13`, field `0x22`
- `VEL`: Aiding Velocity NED, descriptor set `0x13`, field `0x29`
- `HDG`: Aiding Heading True, descriptor set `0x13`, field `0x31`

It also prints one `STATS` line per second with received byte and packet counts.

When the Teensy is in no-GPS-fix mode, position and velocity packets should still
appear, but their flags should be `0x0000`. With a valid GPS fix, position and
velocity flags should normally become `0x0007`.
