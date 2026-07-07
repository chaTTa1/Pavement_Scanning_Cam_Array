# Complete XBee and Septentrio mosaic H RTK Configuration Guide

## 1. System Goal

This system is designed to let the XBee WiFi NTRIP Master perform two tasks.

First, the XBee connects to the internet through a Samsung S24 mobile hotspot and receives correction data from the Ohio RTK NTRIP service.

Second, the XBee runs a Socket Server so that a Python program on a computer can receive NMEA data from the mosaic H.

The complete data flow is:

```text
Ohio RTK NTRIP service
        ↓
Samsung S24 mobile hotspot
        ↓
XBee WiFi NTRIP Master
        ↓ RTCMv3
Septentrio mosaic H COM2

Septentrio mosaic H COM2
        ↓ NMEA
XBee Socket Server
        ↓ TCP port 5000
Computer Python program
```

## 2. Confirmed Samsung S24 Hotspot Network

After the computer connected to the Samsung S24 hotspot, Windows reported:

```text
Computer IPv4 address: 10.83.203.177
Subnet mask: 255.255.255.0
Default gateway: 10.83.203.187
Hotspot subnet: 10.83.203.0/24
```

This means devices connected to the hotspot should use addresses in the `10.83.203.x` range.

The earlier XBee address `192.168.0.166` only applied to the previous WiFi network. When the XBee connects to the S24 hotspot, it must use an address in the current hotspot subnet.

The mobile hotspot can assign a different subnet after a restart. Before an important experiment, run:

```cmd
ipconfig
```

Check the following values:

```text
IPv4 Address
Subnet Mask
Default Gateway
```

## 3. XBee WiFi Configuration

### 3.1 Recommended DHCP Configuration

The recommended XBee WiFi settings are:

```text
WiFi: Enabled
SSID: S24wifi
Password: mobile hotspot password
IP config: DHCP
Scan mode: Fast
```

DHCP allows the Samsung S24 to assign:

```text
IP address
Gateway
Subnet
DNS
```

After saving the settings, check the address shown at the top of the XBee page. It may look like:

```text
WiFi S24wifi / 10.83.203.xxx
```

Update the Python program with the actual XBee address:

```python
XBEE_IP = "10.83.203.xxx"
SOCKET_PORT = 5000
```

### 3.2 Optional Static Address

If a static address is required, use settings similar to:

```text
IP address: 10.83.203.166
Gateway: 10.83.203.187
Subnet: /24
DNS: 8.8.8.8
Backup DNS: 1.1.1.1
```

Before assigning `10.83.203.166`, confirm that no other device is using it.

With the XBee turned off, test:

```cmd
ping 10.83.203.166
```

If another device responds, choose a different address.

### 3.3 DNS Purpose

DNS is not important when the computer connects to the XBee directly by IP address.

DNS is important when the XBee connects to an NTRIP caster using a domain name.

Recommended values are:

```text
DNS: 8.8.8.8
Backup DNS: 1.1.1.1
```

## 4. XBee UART Configuration

The XBee UART settings must match mosaic H COM2.

Recommended settings are:

```text
Baud rate: 115200
Data bits: 8
Parity: None
Stop bits: 1
Flow control: None
```

The signal direction is:

```text
mosaic H COM2 TX connects to XBee RX
XBee TX connects to mosaic H COM2 RX
Both devices share GND
```

If the XBee is installed directly in the simpleRTK3B Heading XBee socket, separate TX, RX, and GND wiring is normally not required. The UART baud rate must still match COM2.

## 5. XBee Socket Server Configuration

The Python program actively connects to the XBee. Therefore, the XBee must operate as the Socket Server.

Use:

```text
Socket server: Enabled
TCP port: 5000
UDP port: Not used by the current program
Socket client: Disabled
```

The roles are:

```text
XBee: Socket Server
Computer Python program: TCP Client
```

The Python program must use:

```python
XBEE_IP = "current XBee IP address"
SOCKET_PORT = 5000
```

Test the port from Windows PowerShell:

```powershell
Test-NetConnection 10.83.203.166 -Port 5000
```

A successful result contains:

```text
TcpTestSucceeded : True
```

## 6. XBee NTRIP Client Configuration

If the XBee receives Ohio RTK corrections, its NTRIP Client must be enabled.

Enter the values provided by Ohio RTK:

```text
NTRIP client: Enabled
Caster host: Ohio RTK server address
Caster port: Ohio RTK port
Mount point: Ohio RTK mount point
Username: Ohio RTK username
Password: Ohio RTK password
```

If the Ohio RTK mount point uses VRS, the caster may require the rover position in GGA format. For this reason, mosaic H must output GGA through COM2.

The XBee page should show that the NTRIP connection is active. The received byte counter should continue increasing.

The mosaic H page may still show:

```text
NTRIP disabled
```

This is normal when the external XBee performs the NTRIP connection. That indicator only describes the internal mosaic H NTRIP Client.

## 7. mosaic H COM2 Input and Output Configuration

### 7.1 Query the Current COM2 Configuration

Command:

```text
getDataInOut,COM2
```

Purpose:

```text
Displays the input and output data formats currently allowed on COM2.
```

An earlier result was:

```text
DataInOut, COM2, none, RTCMv3+SBF+NMEA, (on)
```

Meaning:

```text
COM2 input: none
COM2 output: RTCMv3, SBF, and NMEA
COM2 status: enabled
```

This mixed output causes binary data to appear as unreadable characters in a Python text receiver.

### 7.2 Configure RTCMv3 Input and NMEA Output

Command:

```text
setDataInOut,COM2,RTCMv3,NMEA
```

Purpose:

```text
Allows RTCMv3 correction data to enter through COM2.
Allows only NMEA data to leave through COM2.
Stops SBF and RTCMv3 from being transmitted to the XBee.
Keeps the XBee correction input path active.
```

Verify with:

```text
getDataInOut,COM2
```

The correct result is:

```text
DataInOut, COM2, RTCMv3, NMEA, (on)
```

This is the recommended COM2 configuration for the current system.

### 7.3 Optional Configuration Without RTK Input

If the XBee is only used to send position data to the computer and is not used to provide RTK corrections, the command is:

```text
setDataInOut,COM2,none,NMEA
```

Purpose:

```text
Disables COM2 input.
Allows only NMEA output.
```

Do not use this setting when the XBee must send Ohio RTK corrections into the mosaic H.

For the current system, use:

```text
setDataInOut,COM2,RTCMv3,NMEA
```

## 8. mosaic H NMEA Output Configuration

### 8.1 Query All NMEA Streams

Command:

```text
getNMEAOutput,all
```

Purpose:

```text
Lists all configured NMEA output streams.
Shows the output port, enabled messages, and output interval for each stream.
```

### 8.2 Configure GGA, GST, and ZDA on COM2

Command:

```text
setNMEAOutput,Stream1,COM2,GGA+GST+ZDA,sec1
```

Parameter meaning:

```text
Stream1: Name of the NMEA output stream
COM2: Output port
GGA+GST+ZDA: Enabled NMEA messages
sec1: Output once every second
```

Message purposes:

```text
GGA: UTC time, latitude, longitude, altitude, satellite count, and fix quality
GST: Estimated latitude, longitude, and altitude errors
ZDA: UTC time and date
```

GGA is especially important for VRS services because the caster may require the rover position.

### 8.3 Verify the NMEA Stream

Command:

```text
getNMEAOutput,all
```

Expected output should include something similar to:

```text
NMEAOutput, Stream1, COM2, GGA+GST+ZDA, sec1
```

## 9. Configure mosaic H as a Rover

### 9.1 Query the Current PVT Mode

Command:

```text
getPVTMode
```

Purpose:

```text
Shows whether the receiver is configured as Static or Rover.
Shows the permitted positioning solution types.
```

The earlier result was:

```text
PVTMode, Static, StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed, auto
```

This means the receiver was configured in Static mode.

Static mode can cause GGA to report `fix_quality = 7`, keep the output coordinates fixed, and leave GST error fields empty.

### 9.2 Set Rover Mode

Command:

```text
setPVTMode,Rover,StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed,auto
```

Purpose:

```text
Changes the receiver from Static mode to Rover mode.
Allows normal standalone positioning.
Allows SBAS positioning.
Allows DGNSS positioning.
Allows RTK Float.
Allows RTK Fixed.
Lets the receiver automatically select the best available solution.
```

This command is appropriate for a pavement scanning vehicle, mobile platform, or any moving receiver.

### 9.3 Verify Rover Mode

Command:

```text
getPVTMode
```

The correct result is:

```text
PVTMode, Rover, StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed, auto
```

## 10. GGA Fix Quality Values

The Python program reads the `fix_quality` value from GGA.

Common values are:

```text
0: Invalid position
1: Standalone GNSS position
2: DGNSS
4: RTK Fixed
5: RTK Float
7: Manual or Static fixed position state
```

For a moving rover, the desired result is:

```text
fix_quality = 4
```

A value of 5 means that RTK correction data is being used, but the integer ambiguity has not yet been fixed.

A value of 1 means that a normal GNSS position is available, but valid RTK correction data is not being used.

## 11. Empty GST Error Fields

A sentence such as:

```text
$GPGST,193241.00,,,,,,,*75
```

means that GST output is enabled, but no error values are currently available.

The earlier major cause was the Static PVT mode.

After changing to Rover mode and obtaining a valid solution, GST may look like:

```text
$GPGST,193241.00,0.012,0.015,0.009,42.3,0.010,0.011,0.021*XX
```

The Python program would then read:

```text
lat_error_m = 0.010
lon_error_m = 0.011
alt_error_m = 0.021
```

Even in Rover mode, GST fields may remain empty temporarily if the receiver does not yet have a valid solution or has not generated error statistics.

## 12. Save the Configuration

Command:

```text
exeCopyConfigFile,Current,Boot
```

Purpose:

```text
Copies the active configuration to the Boot configuration.
Keeps the current settings after a receiver restart or power cycle.
```

Run this command after all settings have been verified.

## 13. Recommended Complete mosaic H Command Sequence

For the current mobile rover system using Ohio RTK and XBee, use the following sequence:

```text
getDataInOut,COM2
setDataInOut,COM2,RTCMv3,NMEA
getDataInOut,COM2

getNMEAOutput,all
setNMEAOutput,Stream1,COM2,GGA+GST+ZDA,sec1
getNMEAOutput,all

getPVTMode
setPVTMode,Rover,StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed,auto
getPVTMode

exeCopyConfigFile,Current,Boot
```

Detailed command purposes:

```text
getDataInOut,COM2

Reads the current COM2 input and output formats.

setDataInOut,COM2,RTCMv3,NMEA

Allows RTCMv3 correction input from the XBee and allows only NMEA output to the XBee.

getDataInOut,COM2

Confirms that COM2 now uses RTCMv3 input and NMEA output.

getNMEAOutput,all

Displays the current NMEA output streams.

setNMEAOutput,Stream1,COM2,GGA+GST+ZDA,sec1

Configures COM2 to output GGA, GST, and ZDA once per second.

getNMEAOutput,all

Confirms that the NMEA stream is active.

getPVTMode

Displays whether the receiver is currently Static or Rover.

setPVTMode,Rover,StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed,auto

Sets mobile Rover mode and allows all required solution types.

getPVTMode

Confirms that the receiver is now in Rover mode.

exeCopyConfigFile,Current,Boot

Saves the active settings so they remain after restart.
```

## 14. Complete Verification Procedure

### 14.1 Verify WiFi

The XBee page should show:

```text
SSID: S24wifi
Signal strength: acceptable
IP address: 10.83.203.xxx
```

### 14.2 Verify Network Access

With the computer connected to the same S24 hotspot, run:

```cmd
ping actual_XBee_IP
```

Then run:

```powershell
Test-NetConnection actual_XBee_IP -Port 5000
```

### 14.3 Verify the Socket Server

The Python program should display:

```text
[CONNECTED]
```

It should then continue displaying:

```text
[RX]
```

`[CONNECTED]` only confirms the TCP connection.

`[RX]` confirms that the XBee is actually sending data.

### 14.4 Verify NMEA Output

The Python program should receive:

```text
$GPGGA
$GPGST
$GPZDA
```

Large blocks of binary or unreadable text should no longer appear.

### 14.5 Verify RTCMv3 Correction Input

The mosaic H Corrections page should show:

```text
Input port: COM2
Format: RTCMv3
Correction age: continuously updating
Reference station information: available
```

The `XBEE>GPS` indicator on the board should blink when correction data is entering the receiver.

### 14.6 Verify RTK Status

The position solution should progress through states such as:

```text
Standalone
DGNSS
RTK Float
RTK Fixed
```

The desired Python result is:

```text
fix_quality = 4
```

## 15. Final Recommended Configuration

### 15.1 Samsung S24 Hotspot

```text
SSID: S24wifi
Subnet: verify with ipconfig
Observed subnet: 10.83.203.0/24
Observed gateway: 10.83.203.187
```

### 15.2 XBee

```text
WiFi: Enabled
IP config: DHCP
Socket server: Enabled
TCP port: 5000
Socket client: Disabled
NTRIP client: Enabled
UART: 115200, 8N1, no flow control
```

### 15.3 mosaic H

```text
COM2 input: RTCMv3
COM2 output: NMEA
NMEA messages: GGA+GST+ZDA
Output interval: one second
PVT mode: Rover
Allowed solutions: StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed
```

### 15.4 Final Commands

```text
setDataInOut,COM2,RTCMv3,NMEA
setNMEAOutput,Stream1,COM2,GGA+GST+ZDA,sec1
setPVTMode,Rover,StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed,auto
exeCopyConfigFile,Current,Boot
```
