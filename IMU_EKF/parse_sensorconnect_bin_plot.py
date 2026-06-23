from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Spyder-friendly settings. Edit these two paths or run the whole file directly.
DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "IMU_EKF"
BIN_PATH = DATA_DIR / "3DM-CV7-INS_6291.228332_20260616T204052Z.bin"
JSON_PATH = DATA_DIR / "3DM-CV7-INS_6291.228332_20260616T204052Z.json"
SAVE_FIGURES = False
SHOW_FIGURES = True



if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from CV7_INS_EKF import (  # noqa: E402
    DESC_FILTER,
    DESC_GNSS_LEGACY,
    DESC_SENSOR,
    DESC_SYSTEM_DATA,
    MipError,
    MipPacket,
    checksum,
    decoded_packet_fields,
)


SYNC = b"\x75\x65"
GNSS_MODULE_DESCRIPTORS = set(range(0x91, 0x96))


def iter_mip_packets_from_file(path):
    data = path.read_bytes()
    pos = 0
    packet_index = 0
    bad_checksum = 0
    skipped_bytes = 0

    while True:
        sync_index = data.find(SYNC, pos)
        if sync_index < 0:
            break
        skipped_bytes += max(0, sync_index - pos)
        pos = sync_index
        if pos + 6 > len(data):
            break
        payload_len = data[pos + 3]
        total_len = 4 + payload_len + 2
        if pos + total_len > len(data):
            break
        raw = data[pos : pos + total_len]
        pos += total_len
        if checksum(raw[:-2]) != raw[-2:]:
            bad_checksum += 1
            continue
        yield packet_index, MipPacket(raw[2], raw[4:-2], float(packet_index))
        packet_index += 1

    print(
        json.dumps(
            {
                "event": "bin_scan_done",
                "path": str(path),
                "packets": packet_index,
                "bad_checksum": bad_checksum,
                "skipped_bytes": skipped_bytes,
                "bytes": len(data),
            },
            sort_keys=True,
        )
    )


def update_row_from_item(row, item):
    field = item.get("field", "")
    row.setdefault("fields", []).append(field)
    for key, value in item.items():
        if key in {
            "source",
            "descriptor_set",
            "field",
            "field_descriptor",
            "host_time_unix_s",
            "raw_hex",
            "indicator_names",
        }:
            continue
        if key == "valid_flags":
            row[f"{field}_valid_flags"] = value
        elif key == "valid_flags_hex":
            row[f"{field}_valid_flags_hex"] = value
        else:
            row[key] = value


def parse_bin(bin_path):
    ekf_rows = []
    imu_rows = []
    gnss_rows = []
    system_rows = []
    aid_rows = []
    unknown_packets = 0
    decode_errors = 0

    for packet_index, packet in iter_mip_packets_from_file(bin_path):
        row = {
            "packet_index": packet_index,
            "descriptor_set": f"0x{packet.descriptor_set:02X}",
        }
        try:
            items = decoded_packet_fields(packet, debug=True)
        except MipError as exc:
            decode_errors += 1
            print(f"decode error packet={packet_index}: {exc}")
            continue

        for item in items:
            if item.get("field") in ("aid_measurement_summary", "aiding_measurement_summary"):
                aid_rows.append(
                    {
                        "packet_index": packet_index,
                        "gps_tow_s": item.get("gps_tow_s"),
                        "measurement_source": item.get("measurement_source"),
                        "measurement_type": item.get("measurement_type"),
                        "measurement_type_name": item.get("measurement_type_name"),
                        "indicator": item.get("indicator"),
                        "indicator_hex": item.get("indicator_hex"),
                        "enabled": item.get("enabled"),
                        "used": item.get("used"),
                        "sample_time_warning": item.get("sample_time_warning"),
                        "configuration_error": item.get("configuration_error"),
                    }
                )
                continue
            update_row_from_item(row, item)

        row["fields"] = ",".join(row.get("fields", []))
        if packet.descriptor_set == DESC_FILTER:
            ekf_rows.append(row)
        elif packet.descriptor_set == DESC_SENSOR:
            imu_rows.append(row)
        elif packet.descriptor_set == DESC_GNSS_LEGACY or packet.descriptor_set in GNSS_MODULE_DESCRIPTORS:
            gnss_rows.append(row)
        elif packet.descriptor_set == DESC_SYSTEM_DATA:
            system_rows.append(row)
        else:
            unknown_packets += 1

    print(
        json.dumps(
            {
                "event": "parse_done",
                "ekf_rows": len(ekf_rows),
                "imu_rows": len(imu_rows),
                "gnss_rows": len(gnss_rows),
                "system_rows": len(system_rows),
                "aid_summary_rows": len(aid_rows),
                "unknown_packets": unknown_packets,
                "decode_errors": decode_errors,
            },
            sort_keys=True,
        )
    )
    return (
        pd.DataFrame(ekf_rows),
        pd.DataFrame(imu_rows),
        pd.DataFrame(gnss_rows),
        pd.DataFrame(system_rows),
        pd.DataFrame(aid_rows),
    )


def to_numeric(df):
    for col in df.columns:
        if col in {"fields", "descriptor_set", "filter_state_name", "status_flags_hex", "measurement_type_name"}:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
    return df


def add_plot_time(df):
    if df.empty:
        return df
    if "gps_tow_s" in df.columns and pd.to_numeric(df["gps_tow_s"], errors="coerce").notna().any():
        df["plot_time_s"] = pd.to_numeric(df["gps_tow_s"], errors="coerce")
    elif "reference_time_s" in df.columns:
        df["plot_time_s"] = pd.to_numeric(df["reference_time_s"], errors="coerce")
    else:
        df["plot_time_s"] = pd.to_numeric(df["packet_index"], errors="coerce")
    first = df["plot_time_s"].replace([np.inf, -np.inf], np.nan).dropna()
    df["x_time_s"] = df["plot_time_s"] - first.iloc[0] if len(first) else np.nan
    return df


def add_local_xy(frames):
    lat_values = []
    lon_values = []
    for df in frames:
        if df.empty or "latitude_deg" not in df.columns or "longitude_deg" not in df.columns:
            continue
        lat_values.append(pd.to_numeric(df["latitude_deg"], errors="coerce").dropna())
        lon_values.append(pd.to_numeric(df["longitude_deg"], errors="coerce").dropna())
    if not lat_values:
        return None, None
    lat0 = float(pd.concat(lat_values).median())
    lon0 = float(pd.concat(lon_values).median())
    radius_m = 6378137.0
    for df in frames:
        if df.empty or "latitude_deg" not in df.columns or "longitude_deg" not in df.columns:
            continue
        df["east_m"] = np.deg2rad(pd.to_numeric(df["longitude_deg"], errors="coerce") - lon0) * radius_m * np.cos(np.deg2rad(lat0))
        df["north_m"] = np.deg2rad(pd.to_numeric(df["latitude_deg"], errors="coerce") - lat0) * radius_m
    print(f"Local ENU origin lat0={lat0:.10f}, lon0={lon0:.10f}")
    return lat0, lon0


def save_fig(fig, output_dir, name):
    if not SAVE_FIGURES:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print("Saved:", path)


def plot_series(ax, df, x_col, y_col, label, color, marker=".", every=1):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if every > 1 and len(x) > every:
        x = x.iloc[::every]
        y = y.iloc[::every]
    if len(x):
        ax.scatter(x, y, s=9, color=color, marker=marker, label=label, alpha=0.85)


def plot_outputs(ekf, imu, gnss, aid, output_dir, stamp):
    add_local_xy([ekf, gnss])

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    plot_series(ax, gnss, "east_m", "north_m", "GNSS stream", "blue")
    plot_series(ax, ekf, "east_m", "north_m", "EKF position", "red", every=2)
    ax.set_title("SensorConnect position overlay, equal meter scale")
    ax.set_xlabel("east from shared origin (m)")
    ax.set_ylabel("north from shared origin (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    save_fig(fig, output_dir, f"sensorconnect_position_overlay_{stamp}.png")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    plot_series(axes[0], gnss, "x_time_s", "north_m", "GNSS north", "blue")
    plot_series(axes[0], ekf, "x_time_s", "north_m", "EKF north", "red", every=2)
    axes[0].set_ylabel("north (m)")
    axes[0].set_title("North position")
    plot_series(axes[1], gnss, "x_time_s", "east_m", "GNSS east", "blue")
    plot_series(axes[1], ekf, "x_time_s", "east_m", "EKF east", "red", every=2)
    axes[1].set_ylabel("east (m)")
    axes[1].set_title("East position")
    plot_series(axes[2], gnss, "x_time_s", "ellipsoid_height_m", "GNSS height", "blue")
    plot_series(axes[2], ekf, "x_time_s", "ellipsoid_height_m", "EKF height", "red", every=2)
    axes[2].set_ylabel("height (m)")
    axes[2].set_title("Height")
    axes[2].set_xlabel("time from first sample (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    save_fig(fig, output_dir, f"sensorconnect_position_time_{stamp}.png")

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    plot_series(axes[0], gnss, "x_time_s", "heading_deg", "GNSS course/heading", "blue")
    plot_series(axes[0], ekf, "x_time_s", "yaw_deg", "EKF yaw", "red", every=2)
    axes[0].set_ylabel("deg")
    axes[0].set_title("Heading / yaw")
    plot_series(axes[1], ekf, "x_time_s", "roll_deg", "EKF roll", "red", every=2)
    axes[1].set_ylabel("deg")
    axes[1].set_title("Roll")
    plot_series(axes[2], ekf, "x_time_s", "pitch_deg", "EKF pitch", "red", every=2)
    axes[2].set_ylabel("deg")
    axes[2].set_title("Pitch")
    plot_series(axes[3], ekf, "x_time_s", "yaw_uncert_deg", "EKF yaw uncertainty", "purple", every=2)
    axes[3].set_ylabel("deg")
    axes[3].set_title("Yaw uncertainty")
    axes[3].set_xlabel("time from first sample (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    save_fig(fig, output_dir, f"sensorconnect_attitude_{stamp}.png")

    if not aid.empty:
        aid_plot = aid.copy()
        aid_plot["x_time_s"] = pd.to_numeric(aid_plot["gps_tow_s"], errors="coerce")
        first = aid_plot["x_time_s"].dropna()
        if len(first):
            aid_plot["x_time_s"] -= first.iloc[0]
        aid_plot["used_int"] = pd.to_numeric(aid_plot["used"], errors="coerce").fillna(0).astype(int)
        aid_plot["enabled_int"] = pd.to_numeric(aid_plot["enabled"], errors="coerce").fillna(0).astype(int)
        fig, ax = plt.subplots(1, 1, figsize=(13, 5))
        names = list(aid_plot["measurement_type_name"].dropna().unique())
        for idx, name in enumerate(names):
            part = aid_plot[aid_plot["measurement_type_name"] == name]
            y = np.full(len(part), idx, dtype=float) + part["used_int"].to_numpy(dtype=float) * 0.35
            ax.scatter(part["x_time_s"], y, s=14, label=f"{name} used offset", alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("time from first aiding summary sample (s)")
        ax.set_title("Aiding measurement summary: y+0.35 means used=True")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        save_fig(fig, output_dir, f"sensorconnect_aiding_summary_{stamp}.png")


def print_json_config_summary(json_path):
    if not json_path.exists():
        return
    data = json.loads(json_path.read_text(encoding="utf-8"))
    state = data.get("state", {})
    print("SensorConnect configuration summary:")
    for key in ["0D50", "0D13", "0D56", "1301", "0C31", "0C41"]:
        block = state.get(key)
        if not block:
            continue
        print(f"  {key} {block.get('name')}:")
        for cmd in block.get("commands", []):
            print(f"    {cmd}")


def main():
    stamp = BIN_PATH.stem.replace("3DM-CV7-INS_6291.228332_", "")
    output_dir = DATA_DIR / f"sensorconnect_parse_{stamp}"
    print_json_config_summary(JSON_PATH)
    ekf, imu, gnss, system, aid = parse_bin(BIN_PATH)
    for df in [ekf, imu, gnss, system, aid]:
        to_numeric(df)
        add_plot_time(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    ekf.to_csv(output_dir / f"sensorconnect_ekf_{stamp}.csv", index=False)
    imu.to_csv(output_dir / f"sensorconnect_imu_{stamp}.csv", index=False)
    gnss.to_csv(output_dir / f"sensorconnect_gnss_{stamp}.csv", index=False)
    system.to_csv(output_dir / f"sensorconnect_system_{stamp}.csv", index=False)
    aid.to_csv(output_dir / f"sensorconnect_aid_summary_{stamp}.csv", index=False)

    print("Rows:", {"ekf": len(ekf), "imu": len(imu), "gnss": len(gnss), "system": len(system), "aid": len(aid)})
    if not ekf.empty and "filter_state_name" in ekf.columns:
        print("EKF states:")
        print(ekf["filter_state_name"].value_counts(dropna=False).to_string())
    if not aid.empty:
        print("Aiding summary used counts:")
        print(aid.groupby(["measurement_type_name", "used"]).size().to_string())

    plot_outputs(ekf, imu, gnss, aid, output_dir, stamp)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


main()
