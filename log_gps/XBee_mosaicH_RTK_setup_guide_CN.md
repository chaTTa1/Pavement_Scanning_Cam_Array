# XBee 与 Septentrio mosaic H 的完整配置指南

## 1. 系统目标

本系统的目标是让 XBee WiFi NTRIP Master 同时完成两件事。

第一，XBee 通过 Samsung S24 手机热点访问互联网，并连接 Ohio RTK 的 NTRIP 服务。

第二，XBee 通过 Socket Server 把 mosaic H 输出的 NMEA 数据发送给电脑上的 Python 程序。

完整数据方向如下。

```text
Ohio RTK NTRIP 服务
        ↓
Samsung S24 手机热点
        ↓
XBee WiFi NTRIP Master
        ↓ RTCMv3
Septentrio mosaic H 的 COM2

Septentrio mosaic H 的 COM2
        ↓ NMEA
XBee Socket Server
        ↓ TCP 5000
电脑 Python 程序
```

## 2. 当前确认的网络参数

电脑连接 Samsung S24 热点后，Windows 显示了以下参数。

```text
电脑 IPv4 地址：10.83.203.177
子网掩码：255.255.255.0
默认网关：10.83.203.187
热点网段：10.83.203.0/24
```

这说明连接 S24 热点的所有设备都应使用 `10.83.203.x` 网段。

原来使用的 `192.168.0.166` 只适用于原来的 WiFi 网络。连接 S24 热点后，XBee 地址必须改为 `10.83.203.x`。

手机热点重新启动后，网关或分配地址可能变化，因此建议每次重要实验前再次运行：

```cmd
ipconfig
```

确认电脑当前的 IPv4 地址、子网掩码和默认网关。

## 3. XBee WiFi 设置

### 3.1 推荐使用 DHCP

最推荐的设置是：

```text
WiFi：开启
SSID：S24wifi
Password：手机热点密码
IP config：DHCP
Scan mode：Fast
```

DHCP 会让 Samsung S24 自动给 XBee 分配以下内容：

```text
IP address
Gateway
Subnet
DNS
```

保存设置后，查看 XBee 页面顶部显示的实际地址，例如：

```text
WiFi S24wifi / 10.83.203.xxx
```

然后把 Python 程序中的地址修改为这个实际地址。

```python
XBEE_IP = "10.83.203.xxx"
SOCKET_PORT = 5000
```

### 3.2 必须使用静态地址时

可以参考以下设置：

```text
IP address：10.83.203.166
Gateway：10.83.203.187
Subnet：/24
DNS：8.8.8.8
Backup DNS：1.1.1.1
```

使用静态地址前必须确认 `10.83.203.166` 没有被其他设备占用。

可以先在电脑中测试：

```cmd
ping 10.83.203.166
```

如果在 XBee 关闭时仍然有回复，说明该地址已被其他设备使用，不应分配给 XBee。

### 3.3 DNS 的作用

如果电脑只是通过局域网 IP 连接 XBee，DNS 不重要。

如果 XBee 需要使用域名连接 Ohio RTK 的 NTRIP Caster，DNS 必须正常工作。

推荐：

```text
DNS：8.8.8.8
Backup DNS：1.1.1.1
```

## 4. XBee UART 设置

XBee UART 必须与 mosaic H 的 COM2 完全一致。

推荐设置：

```text
Baud rate：115200
Data bits：8
Parity：None
Stop bits：1
Flow control：None
```

数据方向为：

```text
mosaic H COM2 TX 连接 XBee RX
XBee TX 连接 mosaic H COM2 RX
两者 GND 相连
```

如果 XBee 是直接插在 simpleRTK3B Heading 的 XBee 插槽中，通常不需要额外连接 TX、RX 和 GND，但仍需确保 UART 波特率一致。

## 5. XBee Socket Server 设置

Python 程序主动连接 XBee，因此 XBee 必须作为 Socket Server。

设置如下：

```text
Socket server：开启
TCP port：5000
UDP port：可以保留默认值，但当前程序不使用
Socket client：关闭
```

角色关系如下：

```text
XBee：Socket Server
电脑 Python：TCP Client
```

Python 中应设置：

```python
XBEE_IP = "XBee 当前实际 IP"
SOCKET_PORT = 5000
```

电脑端测试命令：

```powershell
TestNetConnection 10.83.203.166 Port 5000
```

如果工具显示 TCP 测试成功，说明端口可以连接。

## 6. XBee NTRIP Client 设置

如果 XBee 负责从 Ohio RTK 下载修正数据，必须开启 XBee 自己的 NTRIP Client。

设置内容包括：

```text
NTRIP client：开启
Caster host：Ohio RTK 提供的服务器地址
Caster port：Ohio RTK 提供的端口
Mount point：Ohio RTK 提供的挂载点
Username：Ohio RTK 用户名
Password：Ohio RTK 密码
```

如果 Ohio RTK 使用 VRS，XBee 需要把流动站的 GGA 位置发送给 Caster，因此 mosaic H 必须通过 COM2 输出 GGA。

XBee 页面应显示 NTRIP 已连接，接收字节数应持续增加。

mosaic H 网页中的 `NTRIP disabled` 可以继续显示。这个图标表示 mosaic H 内部的 NTRIP Client 没有启用。当前 NTRIP 连接由外部 XBee 完成，所以该图标不能用来判断 XBee NTRIP 是否正常。

## 7. mosaic H COM2 输入输出设置

### 7.1 查询当前 COM2 设置

命令：

```text
getDataInOut,COM2
```

作用：

```text
查询 COM2 当前允许输入和输出的数据格式。
```

你之前得到的结果是：

```text
DataInOut, COM2, none, RTCMv3+SBF+NMEA, (on)
```

含义是：

```text
COM2 输入：none
COM2 输出：RTCMv3、SBF、NMEA
COM2 状态：开启
```

这个设置会让 COM2 输出混合的文本和二进制数据，Python 终端会出现乱码。

### 7.2 让 XBee 输入 RTCMv3，同时让 mosaic 输出 NMEA

命令：

```text
setDataInOut,COM2,RTCMv3,NMEA
```

作用：

```text
把 COM2 输入设置为 RTCMv3。
把 COM2 输出设置为 NMEA。
禁止 COM2 输出 SBF 和 RTCMv3。
保留 XBee 向 mosaic 输入 RTK 修正数据的能力。
```

执行后再次检查：

```text
getDataInOut,COM2
```

正确结果应为：

```text
DataInOut, COM2, RTCMv3, NMEA, (on)
```

这个配置最适合当前系统。

### 7.3 仅记录位置而不使用 XBee RTK 时的可选设置

命令：

```text
setDataInOut,COM2,none,NMEA
```

作用：

```text
禁止 COM2 接收任何输入。
只允许 COM2 输出 NMEA。
```

如果 XBee 需要向 mosaic 输入 Ohio RTK 修正，则不能使用这一设置。

当前系统应使用：

```text
setDataInOut,COM2,RTCMv3,NMEA
```

## 8. mosaic H NMEA 输出设置

### 8.1 查询所有 NMEA 输出流

命令：

```text
getNMEAOutput,all
```

作用：

```text
列出接收机当前配置的全部 NMEA 输出流。
显示每个输出流使用的端口、消息类型和输出周期。
```

### 8.2 设置 COM2 每秒输出 GGA、GST 和 ZDA

命令：

```text
setNMEAOutput,Stream1,COM2,GGA+GST+ZDA,sec1
```

各参数含义如下。

```text
Stream1：输出流名称
COM2：输出端口
GGA+GST+ZDA：需要输出的 NMEA 消息
sec1：每秒输出一次
```

消息用途如下。

```text
GGA：时间、经纬度、高度、卫星数量和定位质量
GST：纬度、经度和高度方向的误差统计
ZDA：UTC 时间和日期
```

GGA 对 VRS NTRIP 服务尤其重要，因为 Caster 可能需要流动站当前位置。

### 8.3 验证 NMEA 输出

再次输入：

```text
getNMEAOutput,all
```

应看到类似：

```text
NMEAOutput, Stream1, COM2, GGA+GST+ZDA, sec1
```

## 9. 把 mosaic H 设置为 Rover 模式

### 9.1 查询当前 PVT 模式

命令：

```text
getPVTMode
```

作用：

```text
查询接收机当前是 Static 模式还是 Rover 模式。
查询允许使用的定位解类型。
```

你之前得到的结果是：

```text
PVTMode, Static, StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed, auto
```

这表示接收机处于 Static 固定模式。

Static 模式会导致 GGA 中可能出现 `fix_quality = 7`，坐标保持固定，而且 GST 误差字段可能为空。

### 9.2 设置为 Rover 模式

命令：

```text
setPVTMode,Rover,StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed,auto
```

作用：

```text
把接收机从 Static 模式改为 Rover 模式。
允许普通单点定位。
允许 SBAS。
允许 DGNSS。
允许 RTK Float。
允许 RTK Fixed。
让接收机自动选择当前可用的最佳定位解。
```

这条命令适用于安装在扫描车、移动平台或其他运动设备上的接收机。

### 9.3 验证 Rover 模式

命令：

```text
getPVTMode
```

正确结果应为：

```text
PVTMode, Rover, StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed, auto
```

## 10. GGA 定位质量值

Python 程序读取 GGA 中的 `fix_quality`。

常见值如下。

```text
0：无有效定位
1：普通 GNSS 定位
2：DGNSS
4：RTK Fixed
5：RTK Float
7：Manual 或 Static 固定位置状态
```

对于移动 Rover，理想状态是：

```text
fix_quality = 4
```

如果为 5，表示已经获得 RTK 修正，但整数模糊度还没有固定。

如果为 1，表示接收机有普通定位，但没有使用有效 RTK 修正。

## 11. GST 误差字段为空的原因

下面的 GST：

```text
$GPGST,193241.00,,,,,,,*75
```

说明 GST 消息已经开启，但误差字段没有数据。

之前的主要原因是接收机处于 Static 模式。

切换到 Rover 模式并获得有效定位后，GST 可能变为：

```text
$GPGST,193241.00,0.012,0.015,0.009,42.3,0.010,0.011,0.021*XX
```

Python 会解析出：

```text
lat_error_m = 0.010
lon_error_m = 0.011
alt_error_m = 0.021
```

即使已经设置为 Rover，接收机在没有有效解或没有生成误差统计时，GST 字段仍可能暂时为空。

## 12. 保存配置到启动配置

命令：

```text
exeCopyConfigFile,Current,Boot
```

作用：

```text
把当前正在使用的配置复制到 Boot 配置。
确保接收机断电或重启后仍保留当前设置。
```

应在全部设置和验证完成后执行。

## 13. 推荐的完整 mosaic H 命令顺序

对于当前使用 Ohio RTK、XBee 和移动 Rover 的系统，推荐按以下顺序执行。

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

每条命令的目的如下。

```text
getDataInOut,COM2
读取 COM2 当前输入输出格式。

setDataInOut,COM2,RTCMv3,NMEA
允许 XBee 向接收机输入 RTCMv3，同时让接收机只向 XBee 输出 NMEA。

getDataInOut,COM2
验证 COM2 已变为 RTCMv3 输入和 NMEA 输出。

getNMEAOutput,all
检查当前 NMEA 输出流。

setNMEAOutput,Stream1,COM2,GGA+GST+ZDA,sec1
设置 COM2 每秒输出 GGA、GST 和 ZDA。

getNMEAOutput,all
确认 NMEA 输出流已经生效。

getPVTMode
查询当前 Static 或 Rover 状态。

setPVTMode,Rover,StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed,auto
把接收机设置为移动 Rover，并允许所有需要的定位解类型。

getPVTMode
确认接收机已经进入 Rover 模式。

exeCopyConfigFile,Current,Boot
保存设置，防止重启后丢失。
```

## 14. 完整验证流程

### 14.1 验证 WiFi

XBee 页面顶部应显示：

```text
SSID：S24wifi
信号强度：正常
IP：10.83.203.xxx
```

### 14.2 验证网络

电脑连接同一个 S24 热点后运行：

```cmd
ping XBee实际IP
```

然后使用 Windows 网络测试工具检查 TCP 5000 端口。

### 14.3 验证 Socket Server

Python 应显示：

```text
[CONNECTED]
```

随后持续显示：

```text
[RX]
```

`[CONNECTED]` 只证明 TCP 已连接。

`[RX]` 才证明 XBee 实际发送了数据。

### 14.4 验证 NMEA

Python 应看到：

```text
$GPGGA
$GPGST
$GPZDA
```

不应再出现大量二进制乱码。

### 14.5 验证 NTRIP 修正输入

mosaic H 的 Corrections 页面应显示：

```text
输入端口：COM2
格式：RTCMv3
修正年龄：持续更新
参考站信息：有效
```

板卡上的 `XBEE>GPS` 指示灯应在接收修正时闪烁。

### 14.6 验证 RTK

定位状态应逐步变化：

```text
普通定位
DGNSS
RTK Float
RTK Fixed
```

Python 中理想结果：

```text
fix_quality = 4
```

## 15. 最终推荐配置汇总

### 15.1 Samsung S24 热点

```text
SSID：S24wifi
网段：以 ipconfig 实际结果为准
本次网段：10.83.203.0/24
本次网关：10.83.203.187
```

### 15.2 XBee

```text
WiFi：开启
IP config：DHCP
Socket server：开启
TCP port：5000
Socket client：关闭
NTRIP client：开启
UART：115200，8N1，无流控
```

### 15.3 mosaic H

```text
COM2 输入：RTCMv3
COM2 输出：NMEA
NMEA 消息：GGA+GST+ZDA
输出周期：1 秒
PVT 模式：Rover
定位解：StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed
```

### 15.4 最终命令

```text
setDataInOut,COM2,RTCMv3,NMEA
setNMEAOutput,Stream1,COM2,GGA+GST+ZDA,sec1
setPVTMode,Rover,StandAlone+SBAS+DGNSS+RTKFloat+RTKFixed,auto
exeCopyConfigFile,Current,Boot
```
