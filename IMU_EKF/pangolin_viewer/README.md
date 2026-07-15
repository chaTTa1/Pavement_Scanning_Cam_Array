# CV7 Pangolin 实时状态窗口

这个窗口使用与 ORB-SLAM3 Viewer 相同类型的 C++ Pangolin/OpenGL 界面。Python
采集程序仍然负责设备只读采集和 CSV 写入，GUI 只通过 UDP 接收低频状态快照，
不会向 CV7 或 mosaic-H 发送配置命令。

## 显示内容

左侧状态栏中的设备和滤波器状态全部是文字，不用枚举数字代替：

- `Full Navigation`、`Initialization`、`AHRS` 等滤波器状态；
- `Stable`、`Converging`、`No warnings` 等滤波器健康状态；
- `Enabled and Used`、`Partially used`、`Waiting for GGA`、`Waiting for HDT`
  等 aiding 状态；
- `RTK Fixed`、`RTK Float`、`Autonomous GNSS` 等定位状态；
- `Connected`、`Receiving NMEA`、`No EKF data` 等连接状态。

速率、卫星数、HDOP、航向、速度、精度及 IMU 测量本身仍显示数值，因为这些不是
枚举状态。右侧蓝色轨迹是 CV7 EKF 位置，橙色点是 mosaic-H 原始 GGA 位置。

`GPS connection` 表示电脑正在从 mosaic-H USB NMEA 监视端口收到数据；它本身不能
证明 CV7 RX 正在接收。`Aiding summary` 和 Filter 状态来自 CV7 的 MIP 输出，用于判断
滤波器是否实际使用 aiding。两项应结合查看。MSCL 当前把多个 aiding summary 字段返回为
相同名称，因此当它们的状态不一致时，GUI 显示 `Partially used`，而不会猜测哪一个重复字段
一定是 position、velocity 或 heading。

## Windows 构建

仓库已经包含可直接运行的 Windows x64 发布包：

```text
release/windows-x64/cv7_pangolin_viewer.exe
release/windows-x64/glew32.dll
release/windows-x64/README.txt
```

另一台 Windows x64 电脑 clone 仓库并安装 Python 采集依赖后，可以直接运行：

```powershell
python IMU_EKF\CV7_read_nmea.py --gui
```

Python 会优先自动找到发布目录中的 EXE，不需要在另一台电脑重新编译 Pangolin。
如果要修改 Viewer 源码或重新生成 EXE，再使用下面的构建步骤。

安装 Visual Studio C++ Build Tools、Git 和 CMake。推荐通过 vcpkg 安装 C++
Pangolin；这里不使用 Windows 上不可用的 `pypangolin` wheel。

```powershell
git clone https://github.com/microsoft/vcpkg "$env:USERPROFILE\vcpkg"
& "$env:USERPROFILE\vcpkg\bootstrap-vcpkg.bat" -disableMetrics
& "$env:USERPROFILE\vcpkg\vcpkg.exe" install "pangolin[core]:x64-windows" --disable-metrics

cmake -S IMU_EKF\pangolin_viewer `
      -B IMU_EKF\pangolin_viewer\build `
      -A x64 `
      "-DCMAKE_TOOLCHAIN_FILE=$env:USERPROFILE\vcpkg\scripts\buildsystems\vcpkg.cmake"
cmake --build IMU_EKF\pangolin_viewer\build --config Release
```

生成文件通常位于：

```text
IMU_EKF\pangolin_viewer\build\Release\cv7_pangolin_viewer.exe
```

该路径会被 Python 程序自动找到。

## Jetson AGX Nano / Ubuntu 构建

先按 Pangolin 官方方式安装依赖并构建 Pangolin，然后构建本窗口：

```bash
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
./scripts/install_prerequisites.sh recommended
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cmake --install build

cd /path/to/Pavement_Scanning_Cam_Array
cmake -S IMU_EKF/pangolin_viewer \
      -B IMU_EKF/pangolin_viewer/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build IMU_EKF/pangolin_viewer/build -j$(nproc)
```

如果 CMake 找不到 Pangolin，可在配置窗口时增加：

```bash
-DPangolin_DIR=/path/to/Pangolin/build
```

## 运行

构建完成后，一条命令同时启动采集和 GUI：

```powershell
python IMU_EKF\CV7_read_nmea.py --gui
```

程序仍会自动识别系统和串口。窗口中的 `Stop recording` 会请求 Python 安全停止，
刷新并关闭 CSV；`Quit viewer` 只关闭窗口，采集仍会继续。

也可以把 Viewer 和采集器分开启动：

```powershell
IMU_EKF\pangolin_viewer\build\Release\cv7_pangolin_viewer.exe --port 5600
python IMU_EKF\CV7_read_nmea.py --gui-no-launch --gui-port 5600
```

部署到另一台机器显示时，Viewer 监听 `0.0.0.0:5600`，采集程序使用
`--gui-host VIEWER_IP --gui-no-launch`。防火墙需要允许 UDP 5600；若要从 Viewer
远程停止记录，还需允许采集端 UDP 5601。

## 数据通道

```text
CV7 MIP ----> Python recorder ----> CSV (原始频率、可靠保存)
                   |
mosaic-H USB NMEA -+----> UDP status snapshot (默认 10 Hz) ----> Pangolin GUI
```

UDP 只用于显示。GUI 未启动、网络丢包或窗口退出都不会造成 CSV 数据丢失。
