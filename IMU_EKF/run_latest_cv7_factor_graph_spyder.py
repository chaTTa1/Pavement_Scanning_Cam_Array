from pathlib import Path
import importlib.util
import runpy
import sys


# Spyder-friendly runner:
# 1. Convert latest IMU_EKF CV7/Teensy CSV group into recorder-format input.
# 2. Run the existing factor_graph_gps_imu_own_data.py code on that input.

sys.dont_write_bytecode = True

if "__file__" in globals():
    THIS_DIR = Path(__file__).resolve().parent
else:
    THIS_DIR = Path.cwd() / "IMU_EKF" if (Path.cwd() / "IMU_EKF").exists() else Path.cwd()

REPO_DIR = THIS_DIR.parent
FACTOR_GRAPH_SCRIPT = REPO_DIR / "test_code" / "recording_code" / "factor_graph_gps_imu_own_data.py"
CONVERTER_SCRIPT = THIS_DIR / "prepare_cv7_factor_graph_input.py"

converter_globals = runpy.run_path(str(CONVERTER_SCRIPT))
stamp = converter_globals["STAMP"]
input_dir = converter_globals["output_dir"]
output_dir = THIS_DIR / f"factor_graph_output_{stamp}"

spec = importlib.util.spec_from_file_location("factor_graph_gps_imu_own_data", FACTOR_GRAPH_SCRIPT)
fg = importlib.util.module_from_spec(spec)
sys.modules["factor_graph_gps_imu_own_data"] = fg
spec.loader.exec_module(fg)

old_argv = sys.argv[:]
try:
    sys.argv = [
        str(FACTOR_GRAPH_SCRIPT),
        "--data-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--duration-s",
        "120",
    ]
    fg.main()
finally:
    sys.argv = old_argv
