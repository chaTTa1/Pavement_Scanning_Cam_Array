# PPS Camera Synchronization

## 1. Connection

Connect the same PPS signal to physical pin 7 on the AGX and all three Nano devices:

```text
PPS source
  +-- AGX pin 7
  +-- left Nano pin 7
  +-- mid Nano pin 7
  +-- right Nano pin 7
```

The code uses `GPIO.BOARD`, so `PPS_PIN = 7` means physical pin 7.

- Use a 3.3 V-compatible buffered PPS signal. Do not connect 5 V.
- The PPS source, AGX, and all Nano devices must share ground.
- Verify pin 7 on the exact carrier board before connecting it.

Set a different `CAMERA_ID` on each Nano: `left`, `mid`, and `right`.

## 2. Run

### With GPS: PPS + RMC/ZDA UTC

Set these values near the top of `remote_display_raw_bytes.py`:

```python
UTC_MODE = "GPS"
GPS_NMEA_PORT = "/dev/ttyUSB0"
GPS_NMEA_BAUD = 115200
```

Change the serial port and baud rate to match the GPS receiver.

Run:

```bash
python3 remote_display_raw_bytes.py
```

### Without GPS: PPS + Internet NTP

Install and start chrony on the AGX:

```bash
sudo apt install chrony
sudo systemctl enable --now chrony
chronyc tracking
```

After chrony reports a synchronized clock, set:

```python
UTC_MODE = "NO_GPS"
```

Run:

```bash
python3 remote_display_raw_bytes.py
```

Both modes still require PPS to be connected to the AGX and all Nano devices.

## 3. Synchronization

1. The AGX starts the three Nano camera programs and broadcasts
   `START_ON_NEXT_PPS` with a unique `session_id`. Each ready Nano replies
   `ARMED`. If all three replies are not received before the next PPS, the AGX
   retries with a new session.
2. After `left`, `mid`, and `right` are armed, the same next physical PPS edge
   becomes `pps_sequence = 0` on the AGX and every Nano. Each later PPS
   increments the sequence; the measured PPS interval is also used to detect
   skipped pulses.
3. Every device records each PPS edge with its local monotonic nanosecond clock.
   The devices do not need identical system clocks because the shared
   `session_id` and `pps_sequence` identify the same physical pulse.
4. The AGX assigns UTC to every PPS:
   - In `GPS` mode, the RMC/ZDA UTC second received after a PPS is paired with
     that PPS. Serial-message delay is not added to the timestamp.
   - In `NO_GPS` mode, the chrony-synchronized AGX system UTC is sampled at
     every PPS edge.
5. On each Nano, a frame timestamp is taken immediately after Spinnaker
   `GetNextImage()` returns. The Nano sends the raw image together with its
   session, PPS sequence, and local offset:

```text
delta_from_pps_ns = image_monotonic_ns - nano_pps_monotonic_ns
```

6. The AGX matches the frame to its own UTC-labeled PPS using
   `(session_id, pps_sequence)`, then calculates with integer nanoseconds:

```text
image_utc_ns = agx_pps_utc_ns + delta_from_pps_ns
```

7. Before `T` is pressed, the AGX continuously receives and displays all three
   camera streams but discards the save-path copies. After the display shows
   `Handshake=SYNCED` and `UTC pair=PAIRED`, pressing `T` changes the recording
   state to `WAIT_NEXT_PPS` and sets the target to the current PPS sequence plus
   one. Frames received between the key press and that PPS are discarded.
   When the target PPS arrives, the state changes to `RECORDING`; frames from
   that PPS sequence onward can be saved. If the handshake or UTC source is not
   ready, the `T` press is ignored.

Only frames with a valid session, PPS match, UTC reference, and offset are
accepted after recording starts. JPEG EXIF stores UTC date/time, nine
fractional digits, and the exact integer `image_utc_ns`; no coordinates are
written.

The timestamp represents when the frame became available to the Nano through
Spinnaker, not the camera sensor's exposure-start time.
