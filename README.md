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

## NMEA only

本节记录 2026-07-14 完成的 mosaic-H 双天线 GPS 与 MicroStrain
3DM-CV7-INS 直连方案、SensorConnect 设置、只读采集程序和已解决的问题。

### “NMEA only”的范围

这里的 “NMEA only” 是指 **mosaic-H 到 CV7 的外部 aiding 链路使用 NMEA**，不经过
Teensy。它不表示电脑端只能读取 NMEA：

```text
mosaic-H COM1 TX  ── NMEA ──>  CV7 RX
mosaic-H USB2     ── NMEA ──>  Windows COM16（电脑保存 GPS raw）
CV7 MAIN USB      ── MIP  ──>  Windows COM13（电脑保存 IMU raw 和 EKF）
```

- GPS TX 和 CV7 RX 必须共地，串口波特率当前为 `115200`。
- CV7 的电脑输出继续使用 MIP；不要为了这套接线关闭 MAIN 接口的 MIP。
- 在 SensorConnect 的 Interface Control 中，只在**实际连接 GPS TX 的 CV7 物理接口**勾选
  `Incoming NMEA`。
- `Outgoing NMEA` 对电脑读取 CV7 MIP 没有必要；只有确实需要 CV7 转发 NMEA 时才启用。
- COM 号可能在重新插拔后改变，以上是本次验证时的端口身份。

本次 Windows 11 测试观察到的端口角色：

| Port | Observed role |
| --- | --- |
| `COM13` | CV7 MAIN MIP stream，能检测到 descriptor sets `0x80/0x82` |
| `COM8` | 同一 MicroStrain 设备的另一个 USB serial interface，当时没有可读 MIP packet |
| `COM15` | Septentrio Virtual USB COM Port 1，通常用于命令/控制 |
| `COM16` | Septentrio Virtual USB COM Port 2，输出 USB2 raw NMEA |

同一个 Windows COM port 通常不能被两个程序同时打开。如果 SensorConnect、RxTools 或
另一个 Python process 正在占用端口，自动检测会出现 `PermissionError(13, 'Access is denied')`；
关闭占用该端口的软件后再运行记录器。

### 双天线 heading 方向

- mosaic-H 的 `ANT_1` 是 main antenna，`ANT_2` 是 auxiliary antenna。
- heading 基线方向是 **main antenna 指向 auxiliary antenna**，不是副天线指向主天线；
  方位角以真北为 `0°`，顺时针增加。
- 如果希望 HDT 直接代表车辆前进方向，通常把 main antenna 放在后面、auxiliary antenna
  放在前面；如果实际安装方向不同，需要在 mosaic-H 中配置正确的 heading offset。
- 室内没有足够卫星信号或双天线解算无效时，HDT 可以为空。mosaic-H 双天线 heading
  在室外有效解算时不依赖车辆运动；CV7 的 kinematic heading 初始化则需要运动。

Septentrio 官方硬件手册确认 `ANT_1`/`ANT_2` 分别对应 main/auxiliary antenna：
[mosaic Hardware Manual](https://www.septentrio.com/system/files/support/mosaic_hardware_manual_v1.11.0.pdf)。

### mosaic-H NMEA 查询与当前输出

只读查看 USB2 数据协议和全部 NMEA stream：

```text
getDataInOut,USB2
getNMEAOutput,all
```

本次确认过的主要输出为：

```text
NMEAOutput, Stream1, COM2, GGA, msec100
NMEAOutput, Stream2, USB2, GGA+GST+HDT+RMC, msec100
NMEAOutput, Stream3, COM1, GGA+GSV+HDT+RMC+VTG+ZDA, msec100
```

- `Stream2/USB2` 给电脑记录 GPS raw，每种消息周期为 `100 ms`。
- `Stream3/COM1` 通过 GPS TX 送入 CV7 RX。
- `NMEAOutput,...` 是查询结果，不能直接作为设置命令重新发送；修改时必须使用
  `setNMEAOutput`，例如：

```text
setNMEAOutput,Stream2,USB2,GGA+GST+HDT+RMC,msec100
```

每次修改后用 `getNMEAOutput,all` 复核。更完整的 Septentrio/XBee 命令说明见
[XBee_mosaicH_RTK_setup_guide_EN.md](log_gps/XBee_mosaicH_RTK_setup_guide_EN.md)。

### CV7 初始化与 aiding 设置

MicroStrain 支持建议先在 SensorConnect 中加载 factory defaults，然后将默认 IMU-AHRS
设置保存为 startup settings。当前阶段不要修改 factory-default IMU-AHRS 参数。

外置 GNSS 天线中心与 CV7 物理原点不重合，因此以下两项是正确位置解算所必需的：

1. 配置 Aiding Frame Configuration（MIP `0x13,0x01`）。
2. 按所选 frame 测量并输入 GNSS antenna lever arm；方向、坐标轴和正负号必须依据
   MicroStrain 手册及实际安装尺寸，不能用估计值替代。

MicroStrain 推荐的 Navigation Filter Initialization（MIP `0x0D,0x52`）为：

| Parameter | Value |
| --- | --- |
| Wait For Run Command | `0` |
| Initial Condition Source | `0` (`AUTO_POS_VEL_ATT`) |
| Auto Heading Alignment Selector | `1` (`kinematic`) |
| Reference Frame Selector | `2` (`LLH`) |
| Initial position/velocity/attitude values | 忽略 |

相关官方页面：

- [Antenna offsets](https://s3.amazonaws.com/files.microstrain.com/CV7_INS_Manual/user_manual_content/installation/Antenna.htm)
- [Frame Configuration 0x13,0x01](https://s3.amazonaws.com/files.microstrain.com/CV7_INS_Manual/external_content/dcp/Commands/0x13/data/0x01.htm)
- [Navigation Filter Initialization 0x0D,0x52](https://s3.amazonaws.com/files.microstrain.com/CV7_INS_Manual/external_content/dcp/Commands/0x0d/data/0x52.htm)

### `Enabled`、`Used` 与 heading 初始化

配置查询中的 `aiding_enable` 只表示某类 measurement **允许进入滤波器**，不表示当前
已经收到并采用该 measurement。SensorConnect 的 Aiding Measurements 面板以及 CV7
Filter Status 才表示运行时状态：

- `Enabled`：该 runtime aiding measurement 当前有效并已被滤波器接受。
- `Used`：该 measurement 正在参与当前 EKF 更新。
- 正确运行时，GNSS Position 和 GNSS Velocity 应显示 `Enabled + Used`；有效 HDT/heading
  输入存在时，Heading 也应显示相应状态。

曾读取到的配置包括：

```text
gnss_position_velocity: enabled=true
external_heading: enabled=true
gnss_heading: command 0x0D,0x50 NACK 0x03
```

这不能证明 GNSS 当时已经被 EKF 使用。`gnss_heading` 查询 NACK 也不等同于 HDT 一定
无效；需要结合设备固件支持、实际 NMEA 输入和 Filter Status 判断。SensorConnect 中
全部灰色时，应理解为当前 runtime measurement 没有达到 Enabled/Used，而不是与配置
JSON 矛盾。

如果采用 `AUTO_POS_VEL_ATT + kinematic` 初始化而没有有效双天线 HDT，车辆需要在室外
进行足够的直线运动，使 CV7 从速度方向建立 heading。室内可在 SensorConnect 中设置
manual heading 进行链路测试，并能让设备进入 Full Navigation，但这个假 heading 不能
用于验证真实 GNSS heading，也不应保存为正式道路采集配置。

### Python 3.12 / Anaconda 环境

在激活的 `py312` Conda 环境中安装：

```powershell
python -m pip install python-mscl pyserial matplotlib
```

`python -m pip` 表示让当前 `python` 解释器运行 pip module，可以避免把包安装到另一个
Python 环境。本次验证使用：

```text
Python 3.12
python-mscl 67.1.0.0
```

不要安装无关的同名 `mscl` 包。程序会检查 module 是否真正提供 MicroStrain 的
`Connection` 和 `InertialNode`；`python-mscl` 的正确导入路径为
`from python_mscl import mscl`。

### 只读 CSV 采集程序

主程序为 [CV7_read_nmea.py](IMU_EKF/CV7_read_nmea.py)，详细说明见
[CV7_read_nmea.md](IMU_EKF/CV7_read_nmea.md)。程序参考 MicroStrain 官方 MSCL 接收流程：

```text
Connection.Serial -> InertialNode -> getDataPackets
```

程序不会调用 `setToIdle()`、`resume()`、`setActiveChannelFields()`、
`enableDataStream()`、`saveSettings()`，也不会向 mosaic-H 发送配置命令。

查看端口：

```powershell
python IMU_EKF\CV7_read_nmea.py --list-ports
```

自动识别端口并开始记录：

```powershell
python IMU_EKF\CV7_read_nmea.py
```

明确指定本次验证的端口：

```powershell
python IMU_EKF\CV7_read_nmea.py --cv7-port COM13 --gps-port COM16
```

只保存图片、不打开 plot 窗口：

```powershell
python IMU_EKF\CV7_read_nmea.py --no-show-plot
```

完全关闭绘图：

```powershell
python IMU_EKF\CV7_read_nmea.py --no-plot
```

启动实时 Pangolin/OpenGL 状态和轨迹窗口：

```powershell
python IMU_EKF\CV7_read_nmea.py --gui
```

GUI 状态直接显示 `Full Navigation`、`Stable`、`Enabled and Used`、
`RTK Fixed` 等文字，不使用枚举数字代替。蓝色为 EKF 轨迹，橙色为 raw GPS
GGA 点。Viewer 的 Windows 与 Jetson 构建方法见
[pangolin_viewer/README.md](IMU_EKF/pangolin_viewer/README.md)。GUI 是低频 UDP
旁路，不改变 CV7/GPS 配置，也不影响 CSV 的原生速率采集。

CSV 每一行保存一个原生速率 event，使用 `source` 区分：

| Source | Content |
| --- | --- |
| `CV7_IMU` | CV7 `0x80` 原始 IMU/AHRS fields |
| `CV7_EKF` | CV7 `0x82` filter/EKF fields |
| `CV7_GNSS` | CV7 当前输出的 `0x81` GNSS fields（如果存在） |
| `CV7_OTHER` | 其他 MIP descriptor sets |
| `GPS_NMEA` | mosaic-H USB2 原始和解析后的 NMEA |

记录结束后会生成 raw GPS 与 EKF position 对比图：蓝色为 CV7 EKF，橙色为 GPS raw
GGA。GPS raw 不会被重复填充到每一条 500 Hz IMU 行。

### 已解决的数据问题

1. **EKF 经纬度为 0**：通常表示 filter 尚未得到有效 position 或未完成初始化；heading
   缺失可能阻止 Full Navigation，但不能只凭 `0,0` 判断唯一原因。应同时查看 Filter
   State、GNSS Position/Velocity 的 Enabled/Used 和 heading 状态。
2. **GPS 与 EKF 初始点不同**：可能同时包含天线 lever arm、两个位置参考点不同、EKF
   平滑/延迟、启动旧 NMEA、以及原程序把 double 降为 float32 等因素。应先完成 frame
   和 lever arm，再比较同步后的点。
3. **GPS NMEA 丢行**：旧版线程会被约 500 Hz 的 MSCL/CSV 工作阻塞。现在 GPS 使用独立
   process，并批量读取 Windows 串口 buffer；稳定记录到 GGA/GST/HDT/RMC 各 `10 Hz`。
4. **USB 启动旧数据**：Septentrio 虚拟 COM 端口可能保留有限的旧数据。程序会高速追赶，
   并在 NMEA UTC 接近电脑 UTC 后才写入本次 CSV；没有 UTC 字段时 3 秒后回退开始记录。
5. **MSCL 精度损失**：旧版对所有 field 调用 `as_float()`，使 `estLatitude` 和
   `estLongitude` 从 double 降为 float32。现在按 `storedAs()` 调用 `as_double()`、
   `as_uint64()` 等正确接口。
6. **Ctrl+C 子进程 traceback**：Windows 会把控制台中断发给所有 process。现在 GPS
   process 忽略 `SIGINT`，由主进程设置共享停止事件并统一关闭 CSV/串口。

最后一次硬件只读验证结果：

- GGA/GST/HDT/RMC 均约 `10 Hz`，GGA interval median 为 `0.09990 s`。
- 追到实时后，GPS fix time 与电脑接收时间差稳定约 `-74` 至 `-77 ms`，没有先前约
  `431 s` 的启动旧数据跳跃。
- 3,088 条含 position 的 EKF 记录得到 3,088 个不同经纬度值，double 精度已保留。
- `uint64` reference time 以完整整数写入 CSV。
- `Ctrl+C` 实测可正常输出 `Stopped`、保存 CSV，且不再出现子进程 traceback。
