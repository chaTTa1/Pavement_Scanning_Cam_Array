# CV7 + mosaic-H 只读单 CSV 记录器

程序：`CV7_read_nmea.py`

这个程序同时记录：

- CV7 `0x80` IMU sensor data；
- CV7 `0x82` EKF 融合结果；
- 如果设备当前有输出，也保留 CV7 `0x81` GNSS data；
- mosaic-H USB2 当前已经输出的原始 NMEA 数据，包括 GGA、HDT、ZDA、HRP，
  以及配置中存在时的 RMC、VTG、GST；
- MSCL 返回的其他 MIP descriptor set，不会静默丢弃。

所有事件写入同一个 `cv7_all_readonly_YYYYMMDD_HHMMSS.csv`。每行保持原始
接收频率，并使用 `source` 区分 `CV7_IMU`、`CV7_EKF`、`CV7_GNSS`、
`CV7_OTHER` 和 `GPS_NMEA`。程序不会把低频 GPS 数据复制到每一行 500 Hz
IMU 数据中。

## 只读保证

CV7 读取使用 MicroStrain 官方 MSCL Python 接收方式：

```text
Connection.Serial -> InertialNode -> getDataPackets
```

程序不会调用以下配置或控制函数：

```text
setToIdle
setActiveChannelFields
enableDataStream
resume
saveSettings
```

GPS 串口由独立进程批量读取串口驱动中已经到达的字节，不会发送 Septentrio 命令。
独立进程避免 MSCL 的高频原生调用和 CSV 写入阻塞 10 Hz NMEA；启动时会先追到接近
电脑 UTC 的实时 NMEA，再把数据写入本次 CSV，避免 Septentrio 虚拟 COM 端口中残留的
旧数据形成时间跳跃。如果 NMEA 中没有任何 UTC 字段，则等待 3 秒后开始记录。程序依赖
CV7 和 mosaic-H 在运行前已经配置好输出数据。如果某个字段没有被设备当前配置输出，
程序不会为了获得该字段而修改设备配置。

## 运行

建议为项目创建独立虚拟环境，再安装 MSCL 和 `pyserial`：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install python-mscl pyserial
```

`python-mscl` 是非官方的 PyPI 打包器，但其中封装的是 MicroStrain MSCL binary。
程序同时兼容官方安装提供的顶层 `mscl` module 和 pip 包提供的
`from python_mscl import mscl`。CV7-INS 需要 MSCL 66.0.0 或更新版本。

安装后验证：

```powershell
python -c "from python_mscl import mscl; print(mscl.__file__)"
python IMU_EKF\CV7_read_nmea.py --help
```

程序默认自动检测当前操作系统，并自动寻找 CV7 MIP 主端口和 GPS NMEA 端口：

```powershell
python IMU_EKF\CV7_read_nmea.py
```

自动检测会：

1. 使用 `platform.system()` 识别 Windows、Linux 或其他当前系统；
2. 按 `CV7_INS_EKF.py` 的设备描述、制造商、VID/PID 和设备路径规则给串口排序；
3. 只调用 MSCL `getDataPackets()`，选择真正输出 `0x80/0x82/0xA0` 的 CV7 端口；
4. 在排除 CV7 端口后，只读探测 NMEA 句子并选择 mosaic-H GPS 端口。

查看操作系统和所有串口身份：

```powershell
python IMU_EKF\CV7_read_nmea.py --list-ports
```

也可以明确填写两个端口，跳过对应的自动扫描：

```powershell
python IMU_EKF\CV7_read_nmea.py --cv7-port COM13 --gps-port COM3
```

固定 CV7 端口、只自动寻找 GPS NMEA 端口：

```powershell
python IMU_EKF\CV7_read_nmea.py --cv7-port COM13 --gps-port auto
```

记录 60 秒并指定输出文件：

```powershell
python IMU_EKF\CV7_read_nmea.py --cv7-port COM13 --gps-port COM3 --duration 60 --output IMU_EKF\test_all.csv
```

如果只想读取 CV7，不打开 GPS USB 监视端口：

```powershell
python IMU_EKF\CV7_read_nmea.py --cv7-port COM13 --gps-port none
```

`Ctrl+C` 由主进程统一处理：主进程通知 GPS 读取进程退出，再关闭 CSV 和串口；
不会向 CV7 发送 idle、resume 或其他配置命令。

## 记录结束后的轨迹图

程序结束记录后默认自动生成并打开一张二维位置对比图：

- 蓝色：CV7 IMU/EKF 融合后的经纬度轨迹；
- 橙色：GPS raw GGA 位置点；
- 坐标轴：相对于第一个有效位置的 East/North 米制坐标。

PNG 自动保存在 CSV 旁边：

```text
cv7_all_readonly_YYYYMMDD_HHMMSS_gps_vs_ekf.png
```

仅保存图片、不打开窗口：

```powershell
python IMU_EKF\CV7_read_nmea.py --no-show-plot
```

完全关闭绘图：

```powershell
python IMU_EKF\CV7_read_nmea.py --no-plot
```

指定图片路径：

```powershell
python IMU_EKF\CV7_read_nmea.py --plot-output IMU_EKF\gps_vs_ekf.png
```

如果当前 EKF 尚未输出有效经纬度，例如 `estLatitude=0`、`estLongitude=0`，
图片仍会显示 GPS raw 点，并在图中注明没有有效 EKF position，避免把 `(0,0)`
错误地画成真实轨迹。绘图需要 `matplotlib`：

```powershell
python -m pip install matplotlib
```

## CSV 结构

- `host_unix_ns`、`host_utc_iso`、`elapsed_s`：主机接收时间；
- `source`：这一行的数据来源；
- `descriptor_set`：CV7 MIP descriptor set；
- `mip_*`：在启动后的 schema discovery 阶段发现的 MSCL channel；
- MSCL 标量按设备报告的原始类型读取：`double` 不降为 `float32`，`uint64` 不转浮点；
- `mip_fields_json`：该 MIP packet 的全部 channel，防止后来出现的新字段丢失；
- `gps_*`：GPS NMEA 解析字段；
- `gps_raw_nmea`：未经修改的完整 NMEA 句子；
- `gps_raw_fields_json`：未经单位变换的 NMEA 字段数组。

在当前接线中，GPS NMEA 通过 `mosaic-H TX -> CV7 RX` 直接进入 CV7，不经过
Teensy。CV7 不会自动把 RX 收到的原始 NMEA 再转发给电脑；要同时保存 GPS 原始
数据，mosaic-H USB2 NMEA 虚拟串口必须同时连接电脑并保持当前输出配置。

## Pangolin 实时 GUI

构建 C++ Pangolin Viewer 后，可以在采集过程中同时显示设备状态和实时轨迹：

```powershell
python IMU_EKF\CV7_read_nmea.py --gui
```

状态枚举不会显示为 `4`、`3`、`5` 之类的数字，而会显示为明确文字，例如：

- Filter：`Full Navigation`、`Stable`、`No warnings`；
- Aiding：`Enabled and Used`、`Partially used`、`Waiting for GGA`、`Waiting for HDT`；
- GNSS：`RTK Fixed`、`RTK Float`、`Autonomous GNSS`；
- Connection：`Connected`、`Receiving NMEA`、`No EKF data`。

速率、坐标、卫星数、HDOP、航向、速度和精度属于测量值，因此仍以数值显示。
右侧蓝线为 CV7 EKF 轨迹，橙点为 mosaic-H raw GGA。GUI 只接收默认 10 Hz 的
UDP 状态副本；CSV 仍按原生数据率记录，不会因 GUI 丢包或退出而丢失数据。

完整的 Windows、Jetson 构建和远程显示方法见
[`pangolin_viewer/README.md`](pangolin_viewer/README.md)。
