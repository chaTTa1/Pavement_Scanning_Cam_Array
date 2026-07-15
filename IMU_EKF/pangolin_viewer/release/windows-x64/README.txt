CV7 INS / mosaic-H Pangolin Viewer - Windows x64
================================================

Package files
-------------
cv7_pangolin_viewer.exe   Real-time Pangolin/OpenGL viewer
glew32.dll                Required OpenGL helper library; keep beside the EXE
README.txt                This file

Recommended use with this repository
------------------------------------
1. Clone or download the Pavement_Scanning_Cam_Array repository.
2. Keep all three files in this release/windows-x64 directory.
3. Activate the Python 3.12 Conda environment used for CV7 acquisition.
4. Install the Python dependencies if necessary:

   python -m pip install python-mscl pyserial matplotlib

5. From the repository root, run:

   python IMU_EKF\CV7_read_nmea.py --gui

CV7_read_nmea.py automatically finds the EXE in this directory. The viewer
does not configure the CV7 or mosaic-H. The Python recorder remains read-only.

Run the viewer separately
-------------------------
From this directory:

   cv7_pangolin_viewer.exe --port 5600

Then, from the repository root:

   python IMU_EKF\CV7_read_nmea.py --gui-no-launch --gui-port 5600

System requirement
------------------
Windows x64 and the Microsoft Visual C++ 2015-2022 x64 Redistributable are
required. If Windows reports that MSVCP140.dll, MSVCP140_CODECVT_IDS.dll,
VCRUNTIME140.dll, or VCRUNTIME140_1.dll is missing, install that Microsoft
runtime. OPENGL32.dll and the other Windows system DLLs are supplied by Windows.

This Windows EXE cannot run on Jetson/Linux ARM64. Build the same C++ source
natively on the Jetson by following IMU_EKF/pangolin_viewer/README.md.

