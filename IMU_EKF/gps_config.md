# mosaic-H GPS Configuration

This file records the currently working GPS output configuration for the
GPS -> Teensy -> CV7-INS integration.

## Working Configuration

The current working setup is:

- `COM1`: SBF binary output to Teensy
- `USB2`: NMEA output for RxControl / PC monitoring
- `COM2`: reserved for XBee WiFi module
- GNSS attitude: multi-antenna, fixed ambiguity only

## COM1: SBF Output to Teensy

Use the GPS port that is physically wired to Teensy `Serial1 RX` / pin 0.
In the current wiring, this is `COM1`.

```text
scs, COM1, baud115200, bits8, No, bit1, none
sdio, COM1, auto, SBF
sso, Stream1, COM1, AttEuler+PVTGeodetic+PosCovGeodetic+VelCovGeodetic+AttCovEuler, msec100
```

Expected verification:

```text
gso, all
```

Expected relevant output:

```text
SBFOutput, Stream1, COM1, AttEuler+PVTGeodetic+PosCovGeodetic+VelCovGeodetic+AttCovEuler, msec100
```

Expected Teensy status:

```text
SBF rates : PVT=10 PosCov=10 VelCov=10 Att=10 AttCov=10 /s
MIP rate  : 20 /s
```

## USB2: NMEA Output for Monitoring

The currently working NMEA stream is on `USB2`:

```text
sdio, USB2, auto, NMEA
sno, Stream1, USB2, GGA+HDT+ZDA+HRP, msec100
```

Expected verification:

```text
gno, all
```

Expected relevant output:

```text
NMEAOutput, Stream1, USB2, GGA+HDT+ZDA+HRP, msec100
```

Useful one-time NMEA output:

```text
exeNMEAOnce, USB2, GGA+HDT+HRP+ZDA
```

## GNSS Attitude Configuration

Use multi-antenna GNSS attitude with fixed ambiguities:

```text
setGNSSAttitude, MultiAntenna, Fixed
setAntennaLocation, Aux1, auto, 0, 0, 0
setAttitudeOffset, 0, 0
```

Verify:

```text
getGNSSAttitude
getAntennaLocation, Aux1
getAttitudeOffset
```

Expected output:

```text
GNSSAttitude, MultiAntenna, Fixed
AntennaLocation, Aux1, auto, 0.0000, 0.0000, 0.0000
AttitudeOffset, 0.000, 0.000
```

Valid heading evidence in NMEA:

```text
$GPHDT,142.xxx,T
$PSSN,HRP,...,142.xxx,...,mode=2,...
```

For Septentrio HRP, mode `2` means heading/pitch with fixed ambiguities.

## COM2: XBee WiFi Module

If `COM2` is connected to the XBee WiFi module, do not use it for Teensy SBF.
Keep it as a low-rate NMEA stream for XBee.

Recommended XBee configuration:

```text
scs, COM2, baud115200, bits8, No, bit1, none
sdio, COM2, auto, NMEA
sno, Stream2, COM2, GGA, sec1
```

Verify:

```text
gno, all
sdio, COM2
scs, COM2
```

Expected relevant output:

```text
NMEAOutput, Stream2, COM2, GGA, sec1
DataInOut, COM2, auto, NMEA, (on)
COMSettings, COM2, baud115200, bits8, No, bit1, none
```

## Save Configuration

After confirming the configuration is correct, save it to boot:

```text
eccf, Current, Boot
```

After rebooting the GPS receiver, re-check:

```text
gso, all
gno, all
getGNSSAttitude
getAntennaLocation, Aux1
getAttitudeOffset
```
