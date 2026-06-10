# CV7 INS EKF 读取与配置说明

本目录用于读取 MicroStrain 3DM-CV7-INS 的 EKF 融合输出，并配合 Teensy 4.1 把 Septentrio mosaic-H 的 GNSS/SBF 数据转换成 CV7 可接收的 MicroStrain MIP external aiding。

当前数据链路：

```text
Septentrio mosaic-H -> Teensy 4.1 -> 3DM-CV7-INS -> Windows Python CSV
```

CV7 不是直接读取 GPS raw stream。mosaic-H 的 PVT、velocity、attitude 先进入 Teensy，Teensy 再发送 CV7 external aiding commands。如果运行 `CV7_INS_EKF.py` 时加入 `--gps-port`，脚本会同时打开 GPS receiver 的 NMEA 串口，并把 GPS raw rows 写入 `cv7_gps_raw_*.csv`。融合后的坐标在 `cv7_ekf_fused_*.csv`。

## 当前硬件连接

CV7 C-Series Connectivity Board 到 Teensy 当前使用 4 类信号：

| CV7 C-Series 信号 | Teensy 4.1 | 方向 | 作用 |
| --- | --- | --- | --- |
| UUT_RX / RxD | Pin 8, Serial2 TX | Teensy -> CV7 | 向 CV7 发送 MIP external aiding |
| UUT_TX / TxD | Pin 7, Serial2 RX | CV7 -> Teensy | 接收 CV7 ACK/NACK 反馈 |
| GPIO1 | Pin 3 | Teensy -> CV7 | GPS PPS 转发输入 |
| GND | GND | common | 共地 |

Teensy 与 mosaic-H：

| mosaic-H 信号 | Teensy 4.1 | 作用 |
| --- | --- | --- |
| SBF UART TX | Pin 0, Serial1 RX | mosaic-H SBF 数据输入 Teensy |
| 1 Hz PPS output | Pin 2 | PPS 输入 Teensy |
| GND | GND | 共地 |

注意：

- CV7 使用 C-Series USB 供电和连接 Windows。
- Teensy 可以同时由 GPS/外部电源供电并连接 USB 调试，但所有设备必须共地。
- 不要用 Teensy 3.3 V 给 CV7 供电。
- 如果使用 C-Series board 左侧 UART header 给 CV7 发送 TTL serial，J4 jumper 应保持拔掉，避免板上 RS232 transceiver 干扰 UUT_RX。

## INS 与 GPS 安装要求

3DM-CV7-INS 的输出不只取决于串口数据，也取决于实际机械安装。采集数据前需要确认以下项目。

### CV7 INS 安装

- CV7 必须刚性固定在车体/设备主体上。标定或采集开始后，CV7 不应相对 GPS 天线移动。
- 明确车体坐标系。例如路面扫描中可以统一使用 X forward、Y right、Z down，或者使用后处理程序要求的坐标系。
- 检查 CV7 本体坐标轴和车体坐标轴是否一致。如果 CV7 相对车体有旋转，需要在 SensorConnect 的 Installation 设置或设备配置中写入 mounting transform。
- 不要把 CV7 固定在软板、松动支架、线束或容易振动的位置。安装松动会导致 EKF 在每次 GPS aiding 到来时被周期性拉回。
- 固定并记录 IMU 的参考点位置。GNSS lever arm 必须从这个 IMU/body 参考点开始测量。

### GPS 天线安装

- GNSS 天线应有尽可能开阔的天空视野，避免靠近车顶边缘、金属遮挡、高电流线缆、电脑/相机结构等容易产生 multipath 的位置。
- 如果使用 mosaic-H 双天线 heading，两根天线必须和 CV7 固定在同一个刚性车体上。
- 主天线到副天线的 baseline 必须固定并测量。baseline 越稳定，heading 通常越稳定。
- mosaic-H 输出的 heading 是主天线到副天线的基线方向，不一定等于车头方向。如果两根天线左右横向安装，需要在使用 CV7 external heading 前加入正确 heading offset。
- 如果 RxControl Attitude View 中 AttEuler/heading 不是 valid，不要把 CV7 external heading 当作可信的最终航向来源。

### Lever Arm 与 Offset

- 测量 CV7 IMU 参考点到 GNSS 天线相位中心的三维向量，并使用与 CV7 installation 一致的 body-frame 坐标定义。
- 如果使用双天线，需要同时记录主天线 lever arm 和主天线到副天线的 baseline 方向。
- 如果 CV7/SensorConnect 提供 GNSS antenna offset 或 lever arm 设置，应在采集前写入。未设置 lever arm 时，转弯轨迹可能出现周期性横向修正，看起来像一段段小刷子。
- 移动天线、更换车顶安装位置或改变 CV7 安装位置后，需要重新检查 lever arm。

### 时间同步

- GPS 1 Hz PPS 通过当前 Teensy 路径输入到 CV7 GPIO1。
- CV7 应设置为 `PPS Source = GPIO`，`GPIO1 = PPS Input`。
- Teensy 状态中应看到 `PPS rate: 1 /s`。
- Teensy 发给 CV7 的 POS/VEL/heading aiding messages 应使用与 mosaic-H 原始测量一致的 GPS time/TOW。时间戳不一致是 500 Hz EKF 轨迹在 10 Hz GPS 更新附近出现周期性修正的常见原因。

### 最低健康状态

正式信任记录文件前，至少确认：

```text
mosaic-H fix: RTK Float 或 RTK Fixed
mosaic-H AttEuler: valid，如果使用 external heading
Teensy PPS rate: 1 /s
Teensy CV7 ACK POS/VEL: 持续增加
Teensy CV7 NACK total: 0
CV7 filter state: Full Navigation
CV7 GNSS Position/Velocity aiding: enabled and used
CV7 Heading aiding: 只有真实 heading 有效时才 enabled and used
```

## CV7 目标配置

当前目标配置：

```text
PPS Source: GPIO
GPIO1: PPS Input
MAIN - Main USB or UART: MIP parser enabled
GNSS Position and Velocity Aiding: enabled
External Heading Aiding: enabled only when mosaic-H AttEuler heading is valid
```

已知正常状态：

```text
GPIO1 feature = PPS, behavior = input
Main interface incoming_protocols = MIP
GNSS position/velocity aiding = true
```

当前接线使用的是 CV7 C-Series 的 `UUT_RX / UUT_TX` 主串口引脚，不需要把 GPIO2 配成 UART2。GPIO2/UART2 只适用于另一种接法：把外部串口输入接到 GPIO2 pin 9。

## 用命令行配置 CV7

Windows 上同一个 COM 口一次只能被一个程序占用。运行以下命令前请关闭 SensorConnect。

Windows 在项目根目录打开 cmd 或 PowerShell；Linux 在项目根目录打开 Terminal。然后使用 `python` 运行命令。

串口示例：

```text
Windows: COM13
Linux  : /dev/ttyACM0, /dev/ttyUSB0, or /dev/serial/by-id/...
```

基础配置：接收 Teensy 的 POS/VEL external aiding。

```powershell
python IMU_EKF\CV7_config_aiding.py --port COM13 --pps-gpio-pin 1
```

Linux 示例：

```bash
python IMU_EKF/CV7_config_aiding.py --port /dev/ttyACM0 --pps-gpio-pin 1
```

当 mosaic-H 双天线 heading 已有效，并且 Teensy 已能发送 heading 时，再启用 CV7 external heading aiding：

```powershell
python IMU_EKF\CV7_config_aiding.py --port COM13 --pps-gpio-pin 1 --enable-external-heading
```

如果没有有效 heading，但需要让 CV7 在静止状态进入 Full Navigation，可临时写入初始 heading：

```powershell
python IMU_EKF\CV7_INS_EKF.py --port COM13 --init-heading-deg 0 --reset-filter --run-filter
```

这只是运行时初始化，不是真实 GNSS heading。mosaic-H heading 正常后，不应长期依赖固定 `--init-heading-deg 0`。

## 查看 CV7 状态

拉取一次配置和最新 EKF/aiding 状态：

```powershell
python IMU_EKF\CV7_INS_EKF.py --port COM13 --status --configure --rate-hz 10 --status-listen-s 6 --pretty
```

如需静止初始化并查看状态：

```powershell
python IMU_EKF\CV7_INS_EKF.py --port COM13 --status --configure --rate-hz 10 --status-listen-s 6 --init-heading-deg 0 --reset-filter --run-filter --pretty
```

重点检查：

```text
filter_state_name: full_nav
position_valid: 1
velocity_valid: 1
aid_measurement_summary ... aiding_pos_llh ... enabled + used
aid_measurement_summary ... aiding_vel_ned ... enabled + used
```

如果 heading 也正常，heading aiding 应变为 enabled/used；SensorConnect 的 Aiding Measurements 里 Heading 的 Enabled/Used 也会亮。

## 500 Hz EKF CSV 记录

先确认 CV7 已进入 Full Navigation。没有真实 heading 时，可先运行一次：

```powershell
python IMU_EKF\CV7_INS_EKF.py --port COM13 --init-heading-deg 0 --reset-filter --run-filter
```

开始 500 Hz 记录：

```powershell
python IMU_EKF\CV7_INS_EKF.py --port COM13 --configure --rate-hz 500 --stream-preset csv --summary --print-hz 1 --record-csv --skip-check --expected-ekf-hz 500
```

如果要在同一次运行里同时记录 CV7 EKF 和外部 GPS 原始 NMEA 数据，可以让脚本自动检测 GPS receiver 的 NMEA 串口：

```powershell
python IMU_EKF\CV7_INS_EKF.py --list-ports
python IMU_EKF\CV7_INS_EKF.py --port COM13 --gps-port auto --configure --rate-hz 500 --stream-preset csv --summary --print-hz 1 --record-csv --skip-check --expected-ekf-hz 500 --expected-gps-hz 10
```

`--port` 使用 CV7 的串口。`--gps-port auto` 会扫描候选串口，并选择实际输出 NMEA 句子的 GPS 串口。如果使用了 `--record-csv` 但没有写 `--gps-port`，脚本也会尝试自动检测 GPS 串口。

Linux 示例：

```bash
python IMU_EKF/CV7_INS_EKF.py --port /dev/ttyACM0 --configure --rate-hz 500 --stream-preset csv --summary --print-hz 1 --record-csv --skip-check --expected-ekf-hz 500
```

`--stream-preset csv` 会只请求高频记录需要的核心 EKF 字段：

```text
filter timestamp
filter status
position LLH
velocity NED
Euler attitude
position uncertainty
velocity uncertainty
Euler uncertainty
```

输出文件保存在 `IMU_EKF` 目录：

```text
cv7_ekf_fused_YYYYMMDD_HHMMSS.csv
cv7_gps_raw_YYYYMMDD_HHMMSS.csv
cv7_skip_validation_YYYYMMDD_HHMMSS.csv
```

说明：

- `cv7_ekf_fused_*.csv` 是主要文件，包含融合后的 latitude、longitude、height、NED velocity、roll、pitch、yaw 和不确定度。
- `cv7_gps_raw_*.csv` 在使用 `--gps-port` 时会包含外部 GPS NMEA rows。如果没有使用 `--gps-port`，它可能只有表头，因为 GPS 是 Teensy external aiding，不是 CV7 自己输出的 GNSS raw stream。
- `cv7_skip_validation_*.csv` 只记录检测到 gap/skipping 的事件。只有表头表示未检测到 skipping。

## 在同一次运行中记录 GPS 原始数据

当前架构下 CV7 不输出 raw GPS rows。所以 `CV7_INS_EKF.py` 会在读取 CV7 的同时，额外打开 GPS receiver 的 NMEA 串口。外部 GPS rows 会写入同一个带时间戳的 GPS CSV：

```text
cv7_gps_raw_YYYYMMDD_HHMMSS.csv
```

这些 rows 包含 host 接收时间、可用时的 NMEA UTC 时间、原始 NMEA 字符串，以及 GGA、RMC、VTG、GST、HDT、ZDA 的解析字段。

## 绘制 EKF/GPS 输出

记录完成后，可以自动读取最新 CSV，并打开 2D 和 3D EKF/GPS 点图：

```powershell
python IMU_EKF\CV7_plot_ekf_gps.py
```

也可以指定某个文件：

```powershell
python IMU_EKF\CV7_plot_ekf_gps.py --ekf-csv IMU_EKF\cv7_ekf_fused_YYYYMMDD_HHMMSS.csv
```

脚本会打印 EKF 点数量和 GPS raw 点数量，然后打开 matplotlib 图窗。它不会保存图片。EKF 和 GPS samples 都用点表示。

如果匹配的 `cv7_gps_raw_*.csv` 里有外部 GPS rows，绘图脚本会自动使用。也可以手动指定 GPS CSV：

```powershell
python IMU_EKF\CV7_plot_ekf_gps.py --gps-csv IMU_EKF\cv7_gps_raw_YYYYMMDD_HHMMSS.csv
```

停止记录：

```text
PowerShell / cmd: Ctrl+C
Spyder: interrupt/stop current execution
```

## Spyder 使用

可以直接在 Spyder 打开 `CV7_INS_EKF.py` 并运行。文件顶部的 `SPYDER / IDE SETTINGS` 已按 500 Hz CSV 记录倾向设置：

```python
SPYDER_PORT = "COM13"
SPYDER_CONFIGURE = True
SPYDER_RATE_HZ = 500
SPYDER_STREAM_PRESET = "csv"
SPYDER_RECORD_CSV = True
SPYDER_SKIP_CHECK = True
SPYDER_EXPECTED_EKF_HZ = 500.0
```

为避免 Spyder console 卡住，保持：

```python
SPYDER_SUMMARY_OUTPUT = True
SPYDER_PRINT_HZ = 2.0
```

不要在 Spyder console 中每 500 Hz 打印所有 JSON 字段。CSV 会完整记录，console 只低频显示摘要。

也可以在 Spyder Console 手动调用：

```python
run_cv7_status(port="COM13", init_heading_deg=0, reset_filter=True, run_filter=True)
run_cv7_reader(port="COM13", configure=True, rate_hz=500, stream_preset="csv", record_csv=True)
```

## SensorConnect 检查

在 SensorConnect 中：

1. `System -> Interface Control`
   - Main port 保持 MIP。
   - 当前 `UUT_RX / UUT_TX` 接线不需要配置 UART2/GPIO2。
2. `GPIO`
   - PPS Source: GPIO。
   - GPIO1: PPS, Input。
   - GPIO2 对当前接线可以保持 Unused。
3. `Estimation Filter -> Aiding Source Enable`
   - GNSS Position and Velocity Aiding 勾选。
   - External Heading Aiding 只在 Teensy 已发送有效 heading 时勾选。
4. `Status Quickview`
   - State 应为 Full Navigation。
   - GNSS Position/GNSS Velocity 的 Enabled 和 Used 应亮。
   - Heading 若仍灰色，说明 CV7 没有使用 external heading aiding。

## mosaic-H 与 Teensy heading 状态

mosaic-H 是双天线 heading receiver，静止时也可以输出 GNSS attitude。RxControl 的 Attitude View 中应看到：

```text
Heading: valid angle
Mode: GNSS-based ...
```

Teensy 状态中应看到：

```text
SBF rates       : PVT=10 PosCov=10 VelCov=10 Att=10 AttCov=10 /s
AttEuler        : TOW=... mode=... err=0 sv=... valid=yes
Heading         : 142.xx deg ... sigma=...
CV7 ACK TIME/HDG: xxx / nonzero
```

如果出现：

```text
AttEuler : mode=0 err=1 sv=255 valid=no
Heading  : ovf deg
CV7 ACK TIME/HDG: xxx / 0
```

说明 Teensy 收到了 AttEuler block，但 mosaic-H 当前把 attitude 标记为无效，Teensy 不会给 CV7 发送 heading。这不是 CV7 拒收，也不是 Python 读取问题。需要检查：

- RxControl 当前 Att 页面是否有效。
- AttEuler/AttCovEuler 是否输出到 Teensy 正在读取的 UART stream。
- Aux antenna 信号质量是否稳定。
- 两根天线 baseline 是否固定、足够长。
- 天线基线方向是否与车体/IMU yaw 方向一致。

## heading offset 注意事项

mosaic-H 输出的 heading 是主天线到副天线的基线方向，不一定等于车头方向或 CV7 IMU body yaw。如果两根天线是左右横向安装，mosaic-H heading 可能与车辆前向相差约 90 deg。

解决方式二选一：

- 在 mosaic-H/RxControl 中设置 attitude offset。
- 在 Teensy 转发 heading 前加/减固定 offset。

在 offset 未确认前，不建议把 external heading 用作最终车辆航向。

## 常见问题

**为什么两个 COM 口只有 COM13 有数据？**  
CV7 USB 会出现两个虚拟串口。实际 MIP 主数据口通常是其中一个；你这里 COM13 是有数据的主口。

**为什么 SensorConnect 打开时 Python 不能读？**  
Windows 上串口通常独占。关闭 SensorConnect 后再运行 Python。

**为什么 Aiding Measurements 的 GNSS Position/Velocity 亮了，但 Heading 灰色？**  
POS/VEL 已被 CV7 使用，但 Teensy 没有发送有效 `HEADING_TRUE 0x13/0x31`，或 CV7 未启用 External Heading Aiding。

**为什么 CV7 没有 heading 时也能 Full Navigation？**  
可以用 `--init-heading-deg 0` 给 EKF 一个临时初始航向，让滤波器进入 Full Navigation。真实 heading 仍应来自 mosaic-H 或车辆运动/kinematic alignment。

**为什么 500 Hz 记录时 GPS 只有 10 Hz？**  
GPS/mosaic-H aiding 是低频修正；CV7 EKF 可在 IMU propagation 下 500 Hz 输出融合状态。
