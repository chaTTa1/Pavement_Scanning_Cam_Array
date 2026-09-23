import json
import gc
import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from pyspin import PySpin


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "captures"
MAPPING_FILE = Path(__file__).resolve().parent / "camera_mapping.json"

PREVIEW_WINDOW_NAME = "Camera Identification"
STREAM_WINDOW_NAME = "Three Camera Capture"

ROLES = ["LEFT", "MIDDLE", "RIGHT"]

# Keep these short so the OpenCV window stays responsive.
PREVIEW_TIMEOUT_MS = 100
STREAM_TIMEOUT_MS = 100

# Display size only. Saved images remain at the camera's full resolution.
DISPLAY_HEIGHT = 480

IMAGE_PROCESSOR = PySpin.ImageProcessor()


# ============================================================
# GENERAL HELPERS
# ============================================================

def put_label(
    image,
    text,
    position,
    scale=0.7,
    thickness=2,
    color=(255, 255, 255),
):
    cv2.putText(
        image,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def resize_to_height(image, target_height=DISPLAY_HEIGHT):
    if image is None:
        return None

    h, w = image.shape[:2]
    if h == target_height:
        return image

    scale = target_height / float(h)
    new_width = max(1, int(round(w * scale)))
    return cv2.resize(
        image,
        (new_width, target_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )


def make_placeholder(height=DISPLAY_HEIGHT, width=640, text="Waiting for frame..."):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    put_label(image, text, (30, height // 2), 0.8, 2)
    return image


# ============================================================
# CAMERA INFORMATION
# ============================================================

def get_camera_info(cam):
    nodemap = cam.GetTLDeviceNodeMap()

    def get_string(name, default="Unknown"):
        try:
            node = PySpin.CStringPtr(nodemap.GetNode(name))
            if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
                return node.GetValue()
        except PySpin.SpinnakerException:
            pass
        return default

    return {
        "vendor": get_string("DeviceVendorName"),
        "model": get_string("DeviceModelName"),
        "serial": get_string("DeviceSerialNumber"),
    }


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

def configure_camera_for_continuous_acquisition(cam):
    cam.Init()

    nodemap = cam.GetNodeMap()

    acquisition_mode = PySpin.CEnumerationPtr(
        nodemap.GetNode("AcquisitionMode")
    )

    if not PySpin.IsAvailable(acquisition_mode) or not PySpin.IsWritable(
        acquisition_mode
    ):
        raise RuntimeError("AcquisitionMode is not writable.")

    continuous = acquisition_mode.GetEntryByName("Continuous")

    if not PySpin.IsAvailable(continuous) or not PySpin.IsReadable(continuous):
        raise RuntimeError("Continuous acquisition mode is unavailable.")

    acquisition_mode.SetIntValue(continuous.GetValue())


def start_acquisition(cameras):
    started = []
    try:
        for cam in cameras:
            cam.BeginAcquisition()
            started.append(cam)
    except Exception:
        for cam in reversed(started):
            try:
                cam.EndAcquisition()
            except Exception:
                pass
        raise


def stop_acquisition(cameras):
    if cameras is None:
        return

    for cam in cameras:
        try:
            cam.EndAcquisition()
        except Exception:
            pass


def deinitialize_cameras(cameras):
    if cameras is None:
        return

    for cam in cameras:
        try:
            cam.DeInit()
        except Exception:
            pass


# ============================================================
# IMAGE ACQUISITION
# ============================================================

def acquire_bgr_frame(cam, timeout_ms=100):
    image_result = None
    image_converted = None

    try:
        image_result = cam.GetNextImage(timeout_ms)

        if image_result.IsIncomplete():
            return None

        image_converted = IMAGE_PROCESSOR.Convert(
            image_result,
            PySpin.PixelFormat_BGR8,
        )

        # Copy because the Spinnaker image object is released below.
        return image_converted.GetNDArray().copy()

    except PySpin.SpinnakerException as exc:
        # A short timeout is expected while keeping the GUI responsive.
        # Avoid printing every timeout because that can itself make Spyder
        # appear to freeze.
        if "timeout" not in str(exc).lower():
            print(f"Camera acquisition error: {exc}")
        return None

    finally:
        if image_converted is not None:
            try:
                image_converted.Release()
            except Exception:
                pass

        if image_result is not None:
            try:
                image_result.Release()
            except Exception:
                pass


# ============================================================
# MAPPING FILE
# ============================================================

def load_saved_mapping() -> Optional[Dict[str, str]]:
    if not MAPPING_FILE.exists():
        return None

    try:
        with MAPPING_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return None

        mapping = data.get("roles")
        if not isinstance(mapping, dict):
            return None

        if not all(role in mapping for role in ROLES):
            return None

        return {
            role: str(mapping[role])
            for role in ROLES
        }

    except (OSError, json.JSONDecodeError, TypeError):
        return None


def save_mapping(camera_infos, role_to_index):
    data = {
        "roles": {
            role: camera_infos[role_to_index[role]]["serial"]
            for role in ROLES
        }
    }

    with MAPPING_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def saved_serial_mapping_to_indices(camera_infos):
    saved = load_saved_mapping()

    if not saved:
        return {}

    serial_to_index = {
        info["serial"]: index
        for index, info in enumerate(camera_infos)
    }

    result = {}

    for role in ROLES:
        serial = saved.get(role)
        if serial not in serial_to_index:
            return {}

        result[role] = serial_to_index[serial]

    # Do not accept a corrupted mapping where two roles point at
    # the same physical camera.
    if len(set(result.values())) != len(ROLES):
        return {}

    return result


# ============================================================
# CAMERA IDENTIFICATION PREVIEW
# ============================================================

def preview_mouse_callback(event, x, y, flags, state):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for x0, x1, camera_index in state["bounds"]:
        if x0 <= x < x1:
            state["selected_index"] = camera_index
            return


def remove_camera_from_assignments(assignments, camera_index):
    for role in list(assignments.keys()):
        if assignments[role] == camera_index:
            del assignments[role]


def assign_role(assignments, role, camera_index):
    # Remove the selected camera from any old role.
    remove_camera_from_assignments(assignments, camera_index)

    # If this role was already assigned to another camera,
    # remove that old assignment.
    if role in assignments:
        del assignments[role]

    assignments[role] = camera_index


def assignment_for_camera(assignments, camera_index):
    for role, index in assignments.items():
        if index == camera_index:
            return role
    return None


def build_identification_display(frames, camera_infos, state):
    assignments = state["assignments"]
    selected_index = state["selected_index"]

    panels = []
    bounds = []
    x_cursor = 0

    for index, info in enumerate(camera_infos):
        frame = frames.get(index)

        if frame is None:
            panel = make_placeholder(text="Waiting for frame...")
        else:
            panel = resize_to_height(frame)

        panel = panel.copy()

        role = assignment_for_camera(assignments, index)

        # Camera information
        put_label(
            panel,
            f"CAMERA {index}",
            (15, 30),
            scale=0.8,
            thickness=2,
        )

        put_label(
            panel,
            f"Serial: {info['serial']}",
            (15, 60),
            scale=0.65,
            thickness=2,
        )

        if role:
            put_label(
                panel,
                f"ASSIGNED: {role}",
                (15, 92),
                scale=0.75,
                thickness=2,
                color=(0, 255, 0),
            )
        else:
            put_label(
                panel,
                "UNASSIGNED",
                (15, 92),
                scale=0.75,
                thickness=2,
                color=(0, 255, 255),
            )

        if selected_index == index:
            # Thick border tells the user which camera will receive
            # the next L/M/R assignment.
            cv2.rectangle(
                panel,
                (3, 3),
                (panel.shape[1] - 4, panel.shape[0] - 4),
                (0, 255, 0),
                6,
            )
            put_label(
                panel,
                "SELECTED",
                (15, panel.shape[0] - 20),
                scale=0.7,
                thickness=2,
                color=(0, 255, 0),
            )

        bounds.append(
            (x_cursor, x_cursor + panel.shape[1], index)
        )
        x_cursor += panel.shape[1]
        panels.append(panel)

    if not panels:
        return np.zeros((DISPLAY_HEIGHT + 70, 640, 3), dtype=np.uint8)

    display = np.hstack(panels)

    footer_height = 70
    footer = np.zeros(
        (footer_height, display.shape[1], 3),
        dtype=np.uint8,
    )

    put_label(
        footer,
        "CLICK camera  |  L = LEFT   M = MIDDLE   R = RIGHT",
        (15, 27),
        scale=0.65,
        thickness=2,
    )

    put_label(
        footer,
        "ENTER = accept mapping   |   U = clear selected   |   ESC = cancel",
        (15, 55),
        scale=0.55,
        thickness=2,
    )

    display = np.vstack([display, footer])
    state["bounds"] = bounds

    return display


def identify_cameras_interactively(
    cameras,
    camera_infos,
    initial_mapping=None,
):
    """
    Shows all three cameras live.

    Mouse:
        Click a camera panel to select it.

    Keyboard:
        L -> assign selected camera to LEFT
        M -> assign selected camera to MIDDLE
        R -> assign selected camera to RIGHT
        U -> clear selected camera's assignment
        ENTER -> accept when all three roles are assigned
        ESC -> cancel

    If a saved mapping exists, it is shown automatically.
    The user can simply press ENTER to accept it.
    """

    assignments = dict(initial_mapping or {})

    state = {
        "selected_index": 0 if cameras else None,
        "assignments": assignments,
        "bounds": [],
    }

    last_good_frames = {}
    accepted = False

    cv2.namedWindow(
        PREVIEW_WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        PREVIEW_WINDOW_NAME,
        1500,
        620,
    )

    cv2.setMouseCallback(
        PREVIEW_WINDOW_NAME,
        preview_mouse_callback,
        state,
    )

    start_acquisition(cameras)

    try:
        while True:
            frames = {}

            for index, cam in enumerate(cameras):
                frame = acquire_bgr_frame(
                    cam,
                    PREVIEW_TIMEOUT_MS,
                )

                if frame is not None:
                    last_good_frames[index] = frame

                frames[index] = last_good_frames.get(index)

            display = build_identification_display(
                frames,
                camera_infos,
                state,
            )

            cv2.imshow(
                PREVIEW_WINDOW_NAME,
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("l"), ord("L")):
                if state["selected_index"] is not None:
                    assign_role(
                        state["assignments"],
                        "LEFT",
                        state["selected_index"],
                    )

            elif key in (ord("m"), ord("M")):
                if state["selected_index"] is not None:
                    assign_role(
                        state["assignments"],
                        "MIDDLE",
                        state["selected_index"],
                    )

            elif key in (ord("r"), ord("R")):
                if state["selected_index"] is not None:
                    assign_role(
                        state["assignments"],
                        "RIGHT",
                        state["selected_index"],
                    )

            elif key in (ord("u"), ord("U"), 8, 127):
                if state["selected_index"] is not None:
                    remove_camera_from_assignments(
                        state["assignments"],
                        state["selected_index"],
                    )

            elif key in (ord("0"), ord("1"), ord("2")):
                index = key - ord("0")
                if index < len(cameras):
                    state["selected_index"] = index

            elif key in (10, 13):
                if (
                    set(state["assignments"].keys()) == set(ROLES)
                    and len(set(state["assignments"].values())) == len(ROLES)
                ):
                    accepted = True
                    return dict(state["assignments"])
                else:
                    # Do not block the OpenCV event loop with input().
                    # The user can continue clicking/assigning.
                    print(
                        "Please assign LEFT, MIDDLE, and RIGHT "
                        "to three different cameras before pressing Enter."
                    )

            elif key == 27:
                return None

    finally:
        cv2.destroyWindow(PREVIEW_WINDOW_NAME)

        # If the user accepted the mapping, acquisition remains running
        # so the main streaming loop can continue without restarting it.
        if not accepted:
            stop_acquisition(cameras)


# ============================================================
# NORMAL THREE-CAMERA DISPLAY
# ============================================================

def build_stream_display(
    frames,
    camera_infos,
    role_to_index,
):
    panels = []

    for role in ROLES:
        index = role_to_index[role]
        frame = frames.get(index)

        if frame is None:
            panel = make_placeholder(text=f"{role}: waiting...")
        else:
            panel = resize_to_height(frame)

        panel = panel.copy()

        put_label(
            panel,
            role,
            (15, 35),
            scale=1.0,
            thickness=3,
            color=(0, 255, 0),
        )

        put_label(
            panel,
            f"Serial: {camera_infos[index]['serial']}",
            (15, 68),
            scale=0.55,
            thickness=2,
        )

        panels.append(panel)

    return np.hstack(panels)


def get_next_target_number(output_dir: Path) -> int:
    """Return the next shared target number for LEFT/MIDDLE/RIGHT captures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    max_number = 0
    pattern = re.compile(
        r"^(?:left|mid|right)_target_(\d{3})\.jpg$",
        re.IGNORECASE,
    )

    for path in output_dir.iterdir():
        if path.is_file():
            match = pattern.match(path.name)
            if match:
                max_number = max(max_number, int(match.group(1)))

    return max_number + 1


def save_target_set(last_good_frames, role_to_index):
    """
    Save one synchronized set using a shared sequence number.

    Example:
        left_target_001.jpg
        mid_target_001.jpg
        right_target_001.jpg
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_number = get_next_target_number(OUTPUT_DIR)

    filenames = {
        "LEFT": OUTPUT_DIR / f"left_target_{target_number:03d}.jpg",
        "MIDDLE": OUTPUT_DIR / f"mid_target_{target_number:03d}.jpg",
        "RIGHT": OUTPUT_DIR / f"right_target_{target_number:03d}.jpg",
    }

    for role in ROLES:
        camera_index = role_to_index[role]
        frame = last_good_frames.get(camera_index)

        if frame is None:
            print(f"  {role}: no frame available; not saved.")
            continue

        filename = filenames[role]

        if not cv2.imwrite(
    str(filename),
    frame,
    [cv2.IMWRITE_JPEG_QUALITY, 95],
):
            raise RuntimeError(f"Could not save image: {filename}")

        print(f"  {role}: {filename.name}")

    return target_number




# ============================================================
# MAIN
# ============================================================

def main():
    system = None
    cam_list = None
    cameras: Optional[List] = None

    camera_infos = []
    role_to_index = {}
    cam = None

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Getting Spinnaker system...")
        system = PySpin.System.GetInstance()

        cam_list = system.GetCameras()
        camera_count = cam_list.GetSize()

        print(f"Detected {camera_count} camera(s).")

        if camera_count != 3:
            raise RuntimeError(
                f"Expected exactly 3 cameras, but found {camera_count}."
            )

        # Keep exactly one Python reference to each camera.
        cameras = [
            cam_list.GetByIndex(index)
            for index in range(camera_count)
        ]

        print("\nDetected cameras:")

        for index, cam in enumerate(cameras):
            info = get_camera_info(cam)
            camera_infos.append(info)

            print(
                f"  Camera {index}: "
                f"{info['model']} | Serial {info['serial']}"
            )

        # Initialize/configure all cameras before preview.
        for index, cam in enumerate(cameras):
            print(
                f"Initializing camera {index} "
                f"(serial {camera_infos[index]['serial']})..."
            )
            configure_camera_for_continuous_acquisition(cam)

        # Load saved serial mapping if it still matches the connected
        # cameras. It is displayed in the live preview so the user can
        # visually confirm it before pressing Enter.
        saved_mapping = saved_serial_mapping_to_indices(
            camera_infos
        )

        if saved_mapping:
            print("\nSaved mapping found:")
            for role in ROLES:
                index = saved_mapping[role]
                print(
                    f"  {role}: Camera {index} "
                    f"(serial {camera_infos[index]['serial']})"
                )
        else:
            print("\nNo valid saved mapping found.")

        print("\nOpening live camera identification window...")
        print(
            "Click a camera, then press L/M/R to assign it. "
            "Press Enter when finished."
        )

        role_to_index = identify_cameras_interactively(
            cameras,
            camera_infos,
            initial_mapping=saved_mapping,
        )

        if role_to_index is None:
            print("Camera identification cancelled.")
            return

        save_mapping(
            camera_infos,
            role_to_index,
        )

        print("\nFinal camera mapping:")

        for role in ROLES:
            index = role_to_index[role]
            print(
                f"  {role}: Camera {index} "
                f"(serial {camera_infos[index]['serial']})"
            )

        print(f"\nMapping saved to: {MAPPING_FILE}")
        print(f"Captures will be saved to: {OUTPUT_DIR}")

        # Acquisition is already running because the identification
        # preview left it running after the user accepted the mapping.
        last_good_frames = {}

        print("\nLive stream started.")
        print("Press Q to save one frame from each camera.")
        print("Press ESC to quit without saving.")

        cv2.namedWindow(
            STREAM_WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            STREAM_WINDOW_NAME,
            1500,
            620,
        )

        while True:
            frames = {}

            for role in ROLES:
                index = role_to_index[role]
                cam = cameras[index]

                frame = acquire_bgr_frame(
                    cam,
                    STREAM_TIMEOUT_MS,
                )

                if frame is not None:
                    last_good_frames[index] = frame

                frames[index] = last_good_frames.get(index)

            display = build_stream_display(
                frames,
                camera_infos,
                role_to_index,
            )

            cv2.imshow(
                STREAM_WINDOW_NAME,
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                print("\nSaving target set...")

                target_number = save_target_set(
                    last_good_frames,
                    role_to_index,
                )

                print(
                    f"Capture complete: target {target_number:03d}. "
                    "Continuing live view."
                )

            if key == 27:
                print("\nESC pressed. Exiting.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    except Exception as exc:
        print(f"\nERROR: {exc}")

    finally:
        print("\nCleaning up Spinnaker...")

        # Stop acquisition first.
        if cameras is not None:
            stop_acquisition(cameras)

        # Deinitialize cameras while we still own the references.
        if cameras is not None:
            deinitialize_cameras(cameras)

        # Close OpenCV windows.
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass

        # The main loop leaves 'cam' pointing to one camera.
        # Delete it explicitly so ReleaseInstance() can succeed.
        try:
            del cam
        except Exception:
            pass

        # Release the only Python references we intentionally keep to
        # the camera objects.
        if cameras is not None:
            cameras.clear()

        cameras = None

        gc.collect()

        # Now release the Spinnaker camera list.
        if cam_list is not None:
            try:
                cam_list.Clear()
            except Exception as exc:
                print(f"Warning clearing camera list: {exc}")

        cam_list = None

        gc.collect()

        # Finally release the Spinnaker system.
        if system is not None:
            try:
                system.ReleaseInstance()
            except Exception as exc:
                print(f"Warning releasing Spinnaker system: {exc}")

        system = None

        gc.collect()

        print("Cleanup complete.")


if __name__ == "__main__":
    main()
