# Pavement Scanning Cam Array

This repository is a multi-machine pavement imaging system built around three FLIR Blackfly cameras, Jetson devices, a receiver laptop, and optional GPS/IMU support. The current live pipeline is centered on:

- `blkfly_md.py`
- `remote_display_with_dummygps_md.py`

The rest of the repository contains earlier capture/receiver variants, GPS fusion tools, post-processing scripts, and test utilities.

## What The System Does

At a high level, the project tries to do four things:

1. Synchronize capture with a PPS pulse.
2. Acquire grayscale images from Blackfly cameras on Jetson devices.
3. Stream JPEG frames over the network to a receiver machine.
4. Save and optionally georeference the resulting images.

The active `_md` path is mainly a live streaming and saving pipeline. GPS support exists in the codebase, but in the current `_md` scripts the live GPS listener is commented out, so the active runtime flow is primarily:

`PPS -> camera capture -> JPEG encode -> EXIF timing tag -> TCP stream -> laptop receive -> display/save`

## Active Runtime Architecture

### Sender side: `blkfly_md.py`

This script is intended to run on a Jetson attached to one camera.

Its main jobs are:

- wait for a hardware PPS edge on a Jetson GPIO pin
- configure the Blackfly camera through `PySpin`
- capture frames continuously
- compute the time offset from the PPS event
- JPEG-encode frames in a worker thread
- attach EXIF metadata
- stream framed JPEG payloads to the laptop over TCP

Important globals:

- `CAMERA_ID` identifies which camera role this Jetson is serving: `left`, `mid`, or `right`
- `PORT_MAP` maps each role to a dedicated TCP port on the laptop
- `save_q` carries raw captured frames
- `time_q` carries timing offsets
- `stream_q` carries encoded JPEG payloads plus EXIF bytes
- `stop_event` is the shared shutdown signal

The code is split into three logical layers:

1. Networking and queue helpers
2. Camera configuration and acquisition helpers
3. `main()` orchestration

### Receiver side: `remote_display_with_dummygps_md.py`

This script is intended to run on the receiver laptop.

Its main jobs are:

- open three TCP server sockets, one per camera
- SSH into the Jetsons and start `blkfly_md.py`
- accept one image stream per camera
- decode frames for live display
- save the received JPEG bytes to disk
- optionally broadcast dummy GPS packets for testing
- stop the remote scripts during shutdown

Each camera gets:

- its own listening port
- its own receiving thread
- its own save queue
- its own display slot in the `pygame` window

## Main Logic: `blkfly_md.py`

### 1. PPS synchronization

`main()` begins by waiting for a rising edge on a Jetson GPIO input. When the edge arrives, it stores:

- `event_time = get_tow_from_utc()`

This value becomes the timing reference for captured frames. The intention is to measure each frame relative to the PPS pulse.

The PPS signal source in this repo is the ESP32 firmware under `main/hello_world_main.c`, which generates a 1 Hz pulse on GPIO 18.

### 2. Worker threads

After PPS detection, `blkfly_md.py` starts two daemon threads:

- `image_proc_thread(...)`
- `broadcast_thread(...)`

`image_proc_thread(...)`:

- pulls raw frame packets from `save_q`
- pulls the corresponding timing offset from `time_q`
- JPEG-encodes the frame with OpenCV
- builds an EXIF block with `exif_bytes(...)`
- pushes `(frame_id, jpeg_bytes, exif_bytes)` into `stream_q`

`broadcast_thread(...)`:

- opens a TCP connection to the receiver laptop
- reads items from `stream_q`
- inserts EXIF bytes into the JPEG using `piexif.insert(...)`
- sends a 4-byte big-endian payload length
- sends the JPEG payload itself
- reconnects automatically if the socket drops

### 3. Camera discovery and configuration

The camera control flow is:

- `sysScan()` gets the `PySpin` system object and camera list
- `run_single_camera(...)` initializes one camera and calls `acquire_images(...)`
- `cam_configuration(...)` configures frame rate, exposure, gain, black level, and buffer count
- `trigger_configuration(...)` configures trigger mode and stream buffer behavior

The helper functions below that section are thin wrappers around `PySpin` nodes such as:

- `setFrameRate(...)`
- `setExposureTime(...)`
- `setGain(...)`
- `setTriggerMode(...)`
- `setTriggerSource(...)`
- `setStreamBufferHandlingMode(...)`

### 4. Continuous acquisition path

The active path is inside `acquire_images(...)`.

It:

- ensures the camera is configured for continuous acquisition
- begins acquisition if not already streaming
- loops forever calling `capture_image(...)`
- computes `tow - event_time`
- pushes `(packet, offset)` into the processing queues

`capture_image(...)` returns a tuple containing:

- `frame_id`
- a local high-resolution timestamp from `time.perf_counter()`
- the NumPy image array

In practice, this means the current `_md` sender is a continuous live stream producer. Although the file still contains helper code for software and hardware trigger modes, `main()` passes `triggerType = "Off"` and the continuous loop is the real runtime behavior.

### 5. EXIF behavior in the `_md` path

In the current `_md` implementation, EXIF is mainly used for timing metadata:

- camera make/model
- focal length
- `UserComment` containing the frame time offset from the PPS pulse

The file still contains GPS support helpers such as `gps_listener(...)` and `decimal_to_dms_precise(...)`, but the GPS thread is commented out in `main()`, and the active `exif_bytes(...)` builder does not write GPS tags.

## Main Logic: `remote_display_with_dummygps_md.py`

### 1. Static camera map

`CAMERA_CONFIGS` ties together:

- sender IP address
- camera label
- SSH username
- save queue
- target display slot

The script expects:

- left camera at `192.168.1.12`
- mid camera at `192.168.1.11`
- right camera at `192.168.1.13`

### 2. TCP server setup

`main()` creates three server sockets:

- port `5001` for left
- port `5002` for mid
- port `5000` for right

Each receiving thread runs `receiver(server_sock, label)`.

That function:

- waits for a client connection
- identifies the sender by IP
- reads a 4-byte frame length
- reads exactly that many payload bytes
- decodes the JPEG for display using OpenCV
- stores the raw JPEG bytes in the save queue
- updates the latest frame for the `pygame` display loop

### 3. Remote process startup

`init()` launches `blkfly_md.py` remotely over SSH on all three Jetsons with `nohup`.

So the receiver is effectively the session controller for the entire distributed capture run.

### 4. Dummy GPS broadcaster

`gps_dummy_sender()` broadcasts fake GPS data over UDP port `5005`.

The payload shape matches the older sender-side GPS listener format:

- `left`
- `mid`
- `right`

This is useful for testing the older GPS-aware scripts. In the active `_md` sender path, the GPS listener thread is currently commented out, so this dummy broadcaster is more of a compatibility/testing tool than an active dependency.

### 5. Saving and display

`saver()` writes the received JPEG bytes directly to:

- `C:\SFM_IMAGES\left_cam`
- `C:\SFM_IMAGES\mid_cam`
- `C:\SFM_IMAGES\right_cam`

Because the sender inserts EXIF before transmission, the laptop saves the final network payload exactly as received.

The main display loop uses `pygame` to show the latest left, mid, and right frames side by side.

### 6. Shutdown

`de_init()` SSHes back into each Jetson, finds `blkfly_md.py` PIDs with `pgrep`, and sends `SIGINT` twice to stop them cleanly.

## End-To-End `_md` Flow

1. Run `remote_display_with_dummygps_md.py` on the laptop.
2. The laptop opens three TCP listeners.
3. The laptop SSH-launches `blkfly_md.py` on each Jetson.
4. Each Jetson waits for one PPS edge.
5. Each Jetson starts continuous camera capture.
6. Raw frames are queued for processing.
7. Worker threads JPEG-encode and attach EXIF timing metadata.
8. Each Jetson streams length-prefixed JPEGs to its assigned laptop port.
9. The laptop receives, displays, and saves the images.
10. Closing the receiver window triggers remote shutdown.

## Other Important Root-Level Files

### Active support modules

| File | Purpose |
| --- | --- |
| `main/hello_world_main.c` | ESP32 firmware that generates a hardwired 1 Hz PPS pulse on GPIO 18. |
| `Broadcast_GPS.py` | Runs an EKF on IMU + GPS serial data, computes fused position, converts it into per-camera GPS positions, broadcasts them over UDP, and can SSH-launch `blkfly.py` on Jetsons. |
| `cam_array_post_processing.py` | Offline processing utilities for IMU/GPS interpolation, Mosaic text parsing, per-camera GPS conversion, and EXIF tagging of saved images. |
| `Ext_Kalman_filter.py` | Standalone IMU/GPS EKF logger without the remote camera-launch/broadcast orchestration. |
| `gps_socket.py` | Receives NMEA-like GPS streams from two network devices and logs parsed GGA/GST data to CSV. |

### Legacy and transitional capture/receiver scripts

| File | Purpose |
| --- | --- |
| `blkfly.py` | Older Jetson camera script that saves frames locally and writes TOW plus GPS directly into file EXIF rather than streaming them in the current `_md` TCP format. |
| `blkfly2.py` | Transitional network sender that moves toward queue-based processing and EXIF insertion, but uses UDP image transport instead of the newer framed TCP approach. |
| `remote_display_with_dummygps.py` | Older UDP receiver for `blkfly2.py`; launches remote senders, saves incoming JPEGs, displays them, and broadcasts dummy GPS. |
| `remote_display_with_exif.py` | Similar to the older UDP receiver, but without the dummy GPS broadcast helper. |
| `12_17_test.py` | Another receiver-side prototype for streaming/display/saving, functionally close to the older UDP-based display scripts. |

## `Parsers/Testing_Files` Summary

These are mostly standalone analysis or one-off utility scripts rather than part of the live capture loop.

| File | Purpose |
| --- | --- |
| `File_rename.py` | Merges `frame_*.jpg` files from multiple folders into one folder with sequential names. |
| `Geotag_mapping.py` | Reads GPS EXIF from images and plots image positions with Matplotlib. |
| `GPS_accuracy.py` | Reads NMEA from serial, parses GGA/GST, and logs GPS accuracy metrics to CSV. |
| `gps_analysis.py` | Loads logged GPS CSV data, computes traveled distance, converts to local coordinates, and plots trajectory/altitude. |
| `gps_converter.py` | Older standalone implementation of per-camera GPS offset conversion from one GPS + heading stream. |
| `image_gps_plots.py` | Scans image folders, extracts EXIF GPS, and creates a scatter plot of image locations. |
| `Image_resolution.py` | Groups sequential frame IDs and copies the sharpest image from each contiguous group using Laplacian variance. |
| `IMUParser.py` | Merges several exported IMU text files into one consolidated CSV aligned by timestamp. |
| `IMU_quaternion_drift_test.py` | Serial monitor for quaternion packets from the IMU. |
| `IMU_to_csv.py` | Logs multiple IMU packet types to CSV and estimates GPS time-of-week for each row. |
| `Real_Time_Mosaic_Parser.py` | Lightweight serial parser for Mosaic GGA/HDT/RMC output. |
| `Run_Post_processing.py` | Small runner script that calls `cam_array_post_processing.propagate_gps_with_imu(...)`. |
| `TimeTest_receiver.py` | UDP latency test receiver with ACK replies. |
| `TimeTest_sender.py` | UDP latency test sender that measures round-trip time. |

## Data And Output Folders

| Path | Role |
| --- | --- |
| `Data_logs/` | Sample output logs and plots produced by GPS/IMU analysis code. |
| `main/` | ESP-IDF firmware project for the PPS generator. |
| `Parsers/Testing_Files/` | Offline utilities, diagnostics, and experiments. |

## Important Configuration Assumptions

Several values are hard-coded across the repo and should be treated as deployment-specific:

- Jetson usernames and IP addresses
- receiver IP address `192.168.1.1`
- per-camera TCP ports
- Windows and Linux absolute save paths
- Jetson GPIO pin numbers for PPS
- serial COM ports for GPS and IMU tools

There are also a few inconsistencies across older files, so the safest interpretation is:

- use the `_md` pair as the current live path
- use the other scripts as references, older revisions, or offline tools

## Practical Reading Guide

If you want to understand the repository quickly, read files in this order:

1. `remote_display_with_dummygps_md.py`
2. `blkfly_md.py`
3. `main/hello_world_main.c`
4. `Broadcast_GPS.py`
5. `cam_array_post_processing.py`

That sequence explains the live runtime first, then the hardware timing source, then the GPS/post-processing side of the project.


## Calculate camera capture FPS based on driving speed

Though camera can adjust FPS using FLIR PySpin, but the setting effect after several frame/millisecond later. To avoid this, set camera capture time use a time.sleep() function and let camera decided how long the exposre time should be.

How to calculate the FPS:
### 1. Variables and Constants
First, let's define the symbols used in the equations based on your code's inputs:
* $\phi_1, \lambda_1$: Previous latitude and longitude (converted to radians)
* $\phi_2, \lambda_2$: Current latitude and longitude (converted to radians)
* $\Delta\phi, \Delta\lambda$: Difference in latitude and longitude ($\phi_2 - \phi_1$ and $\lambda_2 - \lambda_1$)
* $R$: Radius of the Earth ($6371000$ meters)
* $\Delta t$: Time elapsed between the two GPS readings ($t_{current} - t_{previous}$)
* $D_{target}$: Target distance per frame ($2.0$ meters)
* $FPS_{min}, FPS_{max}$: The minimum and maximum allowed framerates ($1$ and $150$)

### 2. The Haversine Distance ($d$)
The great-circle distance between the two GPS coordinates is calculated using the Haversine formula.

First, we calculate the intermediate term $a$:
$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)$$

Then, we calculate the total distance $d$ in meters:
$$d = 2R \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1-a}\right)$$

### 3. Velocity ($v$)
The real-time speed in meters per second is simply the distance divided by the change in time:
$$v = \frac{d}{\Delta t}$$

### 4. Target Framerate ($f$)
The required framerate is determined by a piecewise function to handle stationary (or near-stationary) scenarios, followed by a clamping function to keep it within the camera's hardware limits.

**Raw Framerate Calculation:**
$$f_{raw} = \begin{cases} FPS_{min}, & \text{if } v < 0.2 \\ \frac{v}{D_{target}}, & \text{otherwise} \end{cases}$$

**Hardware Clamping:**
To ensure the camera doesn't crash or stall, the raw framerate is clamped between your defined minimum and maximum limits:
$$f = \max\left(FPS_{min}, \min(f_{raw}, FPS_{max})\right)$$

### 5. Final Dynamic Delay ($t_{delay}$)
Finally, the framerate is inverted to determine the required `time.sleep()` duration (in seconds) between each frame capture:
$$t_{delay} = \frac{1}{f}$$