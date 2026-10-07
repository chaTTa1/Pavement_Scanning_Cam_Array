"""Review single-circle targets, then calibrate three fixed cameras from GPS points.

Image coordinates: u=column, v=row, origin at the top-left pixel center.
World coordinates: local East, North, relative GGA altitude, in meters.
Extrinsics: X_camera = R @ X_world + t; camera axes are right, down, forward.
Normalized DLT initializes each camera. Joint bundle adjustment then refines
all cameras and shared 3D targets with robust, soft GPS constraints. Skew=0;
Brown-Conrady distortion uses OpenCV order [k1, k2, p1, p2, k3].
Image-only residuals, original-GPS residuals and held-out predictions are
reported separately. A small image residual is not absolute 3D accuracy.
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import least_squares


data_dir = Path(r"E:\SynologyDrive\paper\road_defect_detection_data\DLT_test_Sept24\DLT_test_Sept24")
gps_file = data_dir / "GPS_points.csv"
output_dir = Path(__file__).resolve().parent / "DLT_results_Sept24"
camera_prefixes = {"camera_1": "left", "camera_2": "mid", "camera_3": "right"}

# CSV data row 1 corresponds to target_001, independently of camera.
# The GPS coordinates measure the antenna, NOT the circle center.
# Set [dEast, dNorth, dUp] in meters FROM antenna reference point TO circle.
# A constant vector is valid only if its direction stays fixed in WORLD axes.
# Alternatively supply an N by 3 array, one measured world-frame offset per row.
# None allows image review but deliberately prevents an incorrect calibration.
# User assumption: the circle stays vertically 4 cm below the antenna point.
antenna_to_circle_enu_m = [0.0, 0.0, -0.04]

# All frames are reviewed by default. Set >0 to auto-advance unambiguous
# automatic detections after this many milliseconds; failed/manual frames pause.
preview_delay_ms = 0
preview_scale = 1.0
reuse_reviewed_centers = True
review_saved_centers = False  # True to inspect already accepted frames again.
save_annotated_images = True

# Detection settings for these 720 x 540 images; all areas are ORIGINAL pixels.
blob_min_area = 2.0
blob_max_area = 600.0
minimum_dot_contrast = 25.0
minimum_white_ring = 180.0
candidate_score_ratio = 1.4
static_position_radius_px = 3.0
static_minimum_frames = 10
calibration_max_evaluations = 3000

# GPS is an antenna measurement, not an exact circle coordinate. This is a
# regularization weight, NOT a measured GPS standard deviation: a 0.05 m
# correction contributes the same residual magnitude as a 1 px image error.
# For this dataset, initial three-view triangulation differs from GPS by
# about 0.053 m at the median. Sensitivity checks at 0.01/0.03/0.05 m and
# held-out prediction motivated this setting. Raw GPS is never overwritten.
gps_regularization_m = 0.05
joint_robust_loss_scale = 1.0  # soft_l1 transition in the normalized objective.
validation_folds = 5
validation_seed = 20261006
bootstrap_repetitions = 100  # Set 0 to skip uncertainty resampling.
bootstrap_seed = 20261007

# Physical output size: insert at this width in Word to retain the point sizes.
plot_width_cm = 16.0
plot_height_cm = 14.0
plot_font_size = 10
plot_parameter_font_size = 9
plot_export_dpi = 600


def load_gps_points(path):
    """Preserve CSV order; form a small-area metric frame without inventing geoid height."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 8:
        raise ValueError("At least 8 noncoplanar points are needed for the 15-parameter fit.")
    geodetic = np.array([[float(row["latitude"]), float(row["longitude"]),
                          float(row["altitude_m"])] for row in rows])
    if not np.isfinite(geodetic).all():
        raise ValueError("GPS coordinates contain NaN or infinity; no rows were removed.")
    if np.any(np.abs(geodetic[:, 0]) > 90) or np.any(np.abs(geodetic[:, 1]) > 180):
        raise ValueError("GPS latitude/longitude must be degrees.")

    # gps_socket_oneButton.py stores GGA field 9 (orthometric/MSL altitude).
    # With no geoid separation, do not treat that altitude as ellipsoid height.
    # This first-order local frame uses WGS84 surface curvature for E/N and
    # measured altitude differences for U. It is an approximation for this site,
    # not an absolute ECEF transform or a geoid-corrected topocentric frame.
    a = 6378137.0
    flattening = 1.0 / 298.257223563
    e2 = flattening * (2.0 - flattening)
    lat0, lon0 = np.radians(geodetic[0, :2])
    denominator = 1.0 - e2 * np.sin(lat0) ** 2
    radius_north = a * (1.0 - e2) / denominator ** 1.5
    radius_east = a / np.sqrt(denominator)
    delta_lat = np.radians(geodetic[:, 0]) - lat0
    delta_lon = np.radians(geodetic[:, 1]) - lon0
    east = radius_east * np.cos(lat0) * delta_lon
    north = radius_north * delta_lat
    up = geodetic[:, 2] - geodetic[0, 2]
    antenna_points = np.column_stack([east, north, up])
    metadata = {
        "gps_file": str(path.resolve()),
        "reference_latitude_deg": float(geodetic[0, 0]),
        "reference_longitude_deg": float(geodetic[0, 1]),
        "reference_gga_altitude_m": float(geodetic[0, 2]),
        "world_axes": "East, North, relative GGA altitude; meters",
        "world_frame_method": "WGS84 local first-order horizontal approximation; relative MSL height",
        "geoid_separation": "Unknown; no ellipsoid-height or absolute ECEF conversion applied",
        "gps_row_mapping": "1-based CSV data row equals numeric image suffix",
        "camera_extrinsics": "X_camera = R @ X_world + t; right/down/forward camera axes",
    }
    quality = [row.get("fix_quality", "unknown") for row in rows]
    print(f"GPS: {len(rows)} rows; fix qualities: {sorted(set(quality))}")
    print("Local antenna coordinate range [E, N, U], meters:", np.ptp(antenna_points, axis=0))
    print("Using relative GGA altitude, not ellipsoid height. GPS row order is unchanged.")
    return rows, antenna_points, metadata


def find_camera_images(folder, prefix, point_count):
    pattern = re.compile(rf"^{prefix}_target_(\d+)\.(jpg|jpeg|png|bmp|tif|tiff)$", re.I)
    files = {}
    for path in folder.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            point_id = int(match.group(1))
            if point_id in files:
                raise ValueError(f"Duplicate target number {point_id} in {folder}")
            files[point_id] = path
    if set(files) != set(range(1, point_count + 1)):
        raise ValueError(f"{folder}: image numbers must match GPS rows 1..{point_count} exactly.")
    return [files[point_id] for point_id in range(1, point_count + 1)]


def refine_circle(gray, rectangle):
    """Fit an ellipse to subpixel intensity crossings around an isolated dark dot."""
    x1, y1, x2, y2 = rectangle
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    padding = 4
    left = max(0, int(np.floor(x1)) - padding)
    top = max(0, int(np.floor(y1)) - padding)
    right = min(gray.shape[1], int(np.ceil(x2)) + padding + 1)
    bottom = min(gray.shape[0], int(np.ceil(y2)) + padding + 1)
    patch = gray[top:bottom, left:right].astype(float)
    yy, xx = np.indices(patch.shape)
    distance = np.hypot(xx - (center_x - left), yy - (center_y - top))
    core_radius = max(2.0, min(x2 - x1, y2 - y1) / 4.0)
    dark = float(np.min(patch[distance <= core_radius]))
    light = float(np.percentile(patch, 95))
    if light - dark < minimum_dot_contrast:
        return None
    threshold = (dark + light) / 2.0
    mask = np.uint8(patch < threshold)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    target = None
    best_distance = max(3.0, min(x2 - x1, y2 - y1) / 2.0)
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        cx, cy = centroids[index]
        if x == 0 or y == 0 or x + width >= patch.shape[1] or y + height >= patch.shape[0]:
            continue
        if area < 2 or area > blob_max_area:
            continue
        offset = np.hypot(cx + left - center_x, cy + top - center_y)
        if offset < best_distance:
            best_distance = offset
            target = index
    if target is None:
        return None

    component = labels == target
    crossings = []
    for y, x in np.argwhere(component[:, 1:] != component[:, :-1]):
        fraction = (threshold - patch[y, x]) / (patch[y, x + 1] - patch[y, x])
        crossings.append([left + x + fraction, top + y])
    for y, x in np.argwhere(component[1:, :] != component[:-1, :]):
        fraction = (threshold - patch[y, x]) / (patch[y + 1, x] - patch[y, x])
        crossings.append([left + x, top + y + fraction])
    if len(crossings) < 5:
        return None
    ellipse = cv2.fitEllipse(np.array(crossings, dtype=np.float32))
    (cx, cy), (width, height), angle = ellipse
    if min(width, height) <= 0 or max(width, height) > 2 * max(x2 - x1, y2 - y1):
        return None
    if not (x1 <= cx <= x2 and y1 <= cy <= y2):
        return None
    radius = np.sqrt(np.count_nonzero(component) / np.pi)
    distance = np.hypot(xx - (cx - left), yy - (cy - top))
    ring = patch[(distance > 1.2 * radius) & (distance < 1.7 * radius)]
    ring_gray = float(np.median(ring)) if ring.size else 0.0
    return {"u": float(cx), "v": float(cy), "axes": [float(width), float(height)],
            "angle": float(angle), "contrast": light - dark, "ring_gray": ring_gray,
            "method": "intensity_ellipse"}


def detect_camera_targets(files, preview_window=None):
    # Adapted from the supplied SimpleBlobDetector example for ONE small black
    # circle, not a 4 x 11 grid. No findCirclesGrid or image rescaling is used.
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 255
    params.thresholdStep = 5
    params.minRepeatability = 2
    params.minDistBetweenBlobs = 2
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = blob_min_area
    params.maxArea = blob_max_area
    params.filterByCircularity = True
    params.minCircularity = 0.25
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.15
    detector = cv2.SimpleBlobDetector_create(params)
    all_candidates = []
    image_size = None
    for image_index, path in enumerate(files):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Cannot read {path}")
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is not None and current_size != image_size:
            raise ValueError(f"Image size changed in {path}")
        image_size = current_size
        if preview_window is not None:
            progress = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.putText(progress, f"Finding candidates: {image_index + 1}/{len(files)} | Q: quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(preview_window, progress)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(preview_window, cv2.WND_PROP_VISIBLE) < 1:
                return None, image_size
        candidates = []
        for point in detector.detect(gray):
            x, y = point.pt
            radius = max(3.0, point.size / 2.0 + 2.0)
            candidate = refine_circle(gray, (x - radius, y - radius, x + radius, y + radius))
            if candidate is None or candidate["ring_gray"] < minimum_white_ring:
                continue
            ratio = min(candidate["axes"]) / max(candidate["axes"])
            candidate["score"] = candidate["contrast"] * ratio
            candidates.append(candidate)
        all_candidates.append(candidates)

    detections = []
    for candidates in all_candidates:
        usable = []
        for candidate in candidates:
            repeated = 0
            for other_frame in all_candidates:
                for other in other_frame:
                    if np.hypot(candidate["u"] - other["u"], candidate["v"] - other["v"]) < static_position_radius_px:
                        repeated += 1
                        break
            # A parked vehicle/background dot repeats across many target poses.
            # This rejects automatic candidates only; manual selection stays available.
            if repeated < static_minimum_frames:
                usable.append(candidate)
        usable.sort(key=lambda item: item["score"], reverse=True)
        if not usable:
            detections.append((None, "Detection failed. Drag an ellipse around the dot."))
        elif len(usable) > 1 and usable[0]["score"] < candidate_score_ratio * usable[1]["score"]:
            detections.append((None, "Several candidates. Drag around the correct dot."))
        else:
            detections.append((usable[0], "Automatic ellipse: check center, then Enter."))
    return detections, image_size


def mouse_select_circle(event, x, y, flags, state):
    # WINDOW_AUTOSIZE ensures display-to-original coordinates use this exact scale.
    scale = state["scale"]
    height, width = state["gray"].shape
    if event == cv2.EVENT_LBUTTONDOWN:
        if not (0 <= x < width * scale and 0 <= y < height * scale):
            return
        state["start"] = (x / scale, y / scale)
        state["end"] = state["start"]
        state["dragging"] = True
        state["manual"] = True
        state["target"] = None
    elif state["dragging"] and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
        point = (float(np.clip(x / scale, 0, width - 1)),
                 float(np.clip(y / scale, 0, height - 1)))
        state["end"] = point
        if event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
            x1, x2 = sorted([state["start"][0], point[0]])
            y1, y2 = sorted([state["start"][1], point[1]])
            if x2 - x1 < 2 or y2 - y1 < 2:
                state["message"] = "Selection too small. Drag again around the complete circle."
                return
            target = refine_circle(state["gray"], (x1, y1, x2, y2))
            if target is None:
                # Explicit user-drawn ellipse fallback, never an invented auto-center.
                target = {"u": (x1 + x2) / 2, "v": (y1 + y2) / 2,
                          "axes": [x2 - x1, y2 - y1], "angle": 0.0,
                          "method": "manual_drawn_ellipse"}
                state["message"] = "Manual ellipse center. Check/adjust the drag, then Enter."
            else:
                target["method"] = "manual_roi_intensity_ellipse"
                state["message"] = "Circle refined inside selection. Check, then Enter."
            state["target"] = target


def draw_target_preview(image, state, title):
    scale = state["scale"]
    height, width = image.shape[:2]
    display = cv2.resize(image, (round(width * scale), round(height * scale)))
    target = state["target"]
    if target is not None:
        center = (target["u"] * scale, target["v"] * scale)
        axes = (target["axes"][0] * scale, target["axes"][1] * scale)
        color = (0, 180, 255) if state["manual"] else (0, 255, 0)
        cv2.ellipse(display, (center, axes, target["angle"]), color, 1, cv2.LINE_AA)
        pixel = (round(center[0]), round(center[1]))
        cv2.drawMarker(display, pixel, (0, 0, 255), cv2.MARKER_CROSS, 9, 1)
        position = (max(0, min(pixel[0] + 9, display.shape[1] - 175)),
                    max(16, min(pixel[1] - 9, display.shape[0] - 5)))
        cv2.putText(display, f"({target['u']:.2f}, {target['v']:.2f})", position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if state["dragging"]:
        x1, y1 = state["start"]
        x2, y2 = state["end"]
        center = ((x1 + x2) * scale / 2, (y1 + y2) * scale / 2)
        axes = (max(1.0, abs(x2 - x1) * scale), max(1.0, abs(y2 - y1) * scale))
        cv2.ellipse(display, (center, axes, 0.0), (0, 180, 255), 1, cv2.LINE_AA)
    canvas = cv2.copyMakeBorder(display, 0, 155, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    lines = [title, state["message"], "Drag: select/reselect circle | Enter/Space: accept",
             "R: clear | A: auto detect | B: previous | Q/Esc: save and quit"]
    if target is not None:
        lines.append(f"u={target['u']:.4f}, v={target['v']:.4f} px | {target['method']}")
        x, y = round(target["u"]), round(target["v"])
        crop = image[max(0, y - 12):min(height, y + 13), max(0, x - 12):min(width, x + 13)]
        zoom = cv2.resize(crop, (140, 140), interpolation=cv2.INTER_NEAREST)
        cx = (target["u"] - max(0, x - 12) + 0.5) * 140 / crop.shape[1] - 0.5
        cy = (target["v"] - max(0, y - 12) + 0.5) * 140 / crop.shape[0] - 0.5
        cv2.drawMarker(zoom, (round(cx), round(cy)), (0, 0, 255), cv2.MARKER_CROSS, 11, 1)
        canvas[display.shape[0] + 7:display.shape[0] + 147, -147:-7] = zoom
    for index, text in enumerate(lines):
        available = canvas.shape[1] - 164
        font_scale = min(0.45, available / max(1, cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]))
        cv2.putText(canvas, text, (8, display.shape[0] + 23 + 25 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def review_circle_centers(camera_files, cache_file, run_dir):
    if preview_scale <= 0:
        raise ValueError("preview_scale must be positive.")
    cache = {"data_dir": str(data_dir.resolve()), "image_coordinates": "original zero-based u,v pixels", "cameras": {}}
    if reuse_reviewed_centers and cache_file.exists():
        with cache_file.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("data_dir") == cache["data_dir"]:
            cache = saved
    window = "DLT calibration - circle review"
    image_points = {}
    image_sizes = {}
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        for camera, files in camera_files.items():
            print(f"Detecting {camera}: {len(files)} images...")
            detections, image_sizes[camera] = detect_camera_targets(files, window)
            if detections is None:
                print("Detection stopped. Previously accepted centers remain in", cache_file)
                return None, None, cache
            reviewed = cache["cameras"].setdefault(camera, {})
            index = 0
            revisit = False
            while index < len(files):
                path = files[index]
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Cannot read {path}")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                automatic, message = detections[index]
                target = None if automatic is None else automatic.copy()
                cached = reviewed.get(path.name)
                stat = path.stat()
                if cached and cached.get("size_bytes") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
                    if not review_saved_centers and not revisit:
                        index += 1
                        continue
                    target = cached["target"].copy()
                    message = "Previously reviewed center. Check, then Enter."
                state = {"gray": gray, "target": target, "scale": preview_scale,
                         "start": None, "end": None, "dragging": False,
                         "manual": cached is not None, "message": message}
                cv2.setMouseCallback(window, mouse_select_circle, state)
                elapsed_ms = 0
                while True:
                    title = f"{camera} | {path.name} | GPS row {index + 1}/{len(files)}"
                    display = draw_target_preview(image, state, title)
                    cv2.imshow(window, display)
                    key = cv2.waitKey(20) & 0xFF
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (27, ord("q")):
                        print("Review stopped. Accepted centers remain saved in", cache_file)
                        return None, None, cache
                    if key == ord("b") and index > 0:
                        index -= 1
                        revisit = True
                        break
                    if key == ord("r"):
                        state["target"] = None
                        state["manual"] = True
                        state["message"] = "Drag an ellipse around the target circle."
                    elif key == ord("a"):
                        state["target"] = None if automatic is None else automatic.copy()
                        state["manual"] = False
                        state["message"] = message
                        elapsed_ms = 0
                    elapsed_ms += 20
                    auto_accept = (preview_delay_ms > 0 and elapsed_ms >= preview_delay_ms
                                   and not state["manual"] and not state["dragging"])
                    if (key in (10, 13, 32) or auto_accept) and state["target"] is not None and not state["dragging"]:
                        reviewed[path.name] = {"point_id": index + 1, "target": state["target"],
                                               "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                        temporary = cache_file.with_suffix(".tmp")
                        temporary.write_text(json.dumps(cache, indent=2), encoding="utf-8")
                        temporary.replace(cache_file)
                        if save_annotated_images:
                            destination = run_dir / camera
                            destination.mkdir(exist_ok=True)
                            if not cv2.imwrite(str(destination / (path.stem + "_review.png")), display):
                                raise OSError(f"Could not save preview for {path}")
                        index += 1
                        revisit = False
                        break
            image_points[camera] = np.array([[reviewed[path.name]["target"]["u"],
                                              reviewed[path.name]["target"]["v"]] for path in files])
    finally:
        cv2.destroyAllWindows()
    return image_points, image_sizes, cache


def normalized_dlt(world_points, image_points):
    """Estimate P from N by 3 metric coordinates and N by 2 original pixels."""
    world_points = np.asarray(world_points, dtype=float)
    image_points = np.asarray(image_points, dtype=float)
    if world_points.ndim != 2 or world_points.shape[1] != 3 or image_points.shape != (len(world_points), 2):
        raise ValueError("DLT expects corresponding N by 3 and N by 2 arrays.")
    if len(world_points) < 6 or not np.isfinite(world_points).all() or not np.isfinite(image_points).all():
        raise ValueError("DLT needs at least 6 finite correspondences.")
    center_3d = world_points.mean(axis=0)
    centered_3d = world_points - center_3d
    geometry_singular_values = np.linalg.svd(centered_3d, compute_uv=False)
    if np.linalg.matrix_rank(centered_3d) < 3:
        raise ValueError("Control points are coplanar/collinear: full 3-D DLT is not identifiable.")
    center_2d = image_points.mean(axis=0)
    centered_2d = image_points - center_2d
    if np.linalg.matrix_rank(centered_2d) < 2:
        raise ValueError("Image points are collinear.")
    scale_3d = np.sqrt(3) / np.mean(np.linalg.norm(centered_3d, axis=1))
    scale_2d = np.sqrt(2) / np.mean(np.linalg.norm(centered_2d, axis=1))
    T3 = np.eye(4)
    T3[:3, :3] *= scale_3d
    T3[:3, 3] = -scale_3d * center_3d
    T2 = np.eye(3)
    T2[:2, :2] *= scale_2d
    T2[:2, 2] = -scale_2d * center_2d
    xyz = np.column_stack([centered_3d * scale_3d, np.ones(len(world_points))])
    uv = centered_2d * scale_2d
    A = np.zeros((2 * len(world_points), 12))
    for index in range(len(world_points)):
        A[2 * index, :4] = xyz[index]
        A[2 * index, 8:] = -uv[index, 0] * xyz[index]
        A[2 * index + 1, 4:8] = xyz[index]
        A[2 * index + 1, 8:] = -uv[index, 1] * xyz[index]
    _, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    tolerance = singular_values[0] * max(A.shape) * np.finfo(float).eps
    if singular_values[-2] <= tolerance:
        raise ValueError("DLT design matrix is rank deficient beyond its expected scale ambiguity.")
    P = np.linalg.solve(T2, Vt[-1].reshape(3, 4)) @ T3
    if np.linalg.det(P[:, :3]) < 0:
        P = -P
    P /= np.linalg.norm(P[2, :3])
    return P, geometry_singular_values, singular_values


def calibration_residuals(parameters, world_points, image_points):
    fx, fy, cx, cy = parameters[:4]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    distortion = parameters[10:15] if len(parameters) == 15 else np.zeros(5)
    projected, _ = cv2.projectPoints(world_points, parameters[4:7], parameters[7:10], K, distortion)
    return (projected.reshape(-1, 2) - image_points).ravel()


def calibrate_camera(world_points, image_points):
    world_points = np.asarray(world_points, dtype=float)
    image_points = np.asarray(image_points, dtype=float)
    if len(world_points) < 8:
        raise ValueError("The full zero-skew camera plus five distortion coefficients needs at least 8 points.")
    P, geometry_singular_values, design_singular_values = normalized_dlt(world_points, image_points)
    K_dlt, R_dlt, center_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    K_dlt /= K_dlt[2, 2]
    camera_center = (center_h[:3] / center_h[3]).reshape(3)
    t_dlt = -R_dlt @ camera_center
    if K_dlt[0, 0] <= 0 or K_dlt[1, 1] <= 0 or np.linalg.det(R_dlt) < 0:
        raise ValueError("DLT decomposition did not give positive focal lengths and a proper rotation.")
    depth_dlt = (R_dlt @ world_points.T + t_dlt.reshape(3, 1))[2]
    if np.any(depth_dlt <= 0):
        raise ValueError("DLT puts control points behind the camera. Check GPS/image pairing and antenna offsets.")
    rvec_dlt, _ = cv2.Rodrigues(R_dlt)
    initial = np.concatenate([[K_dlt[0, 0], K_dlt[1, 1], K_dlt[0, 2], K_dlt[1, 2]],
                              rvec_dlt.ravel(), t_dlt])
    lower = np.full(10, -np.inf)
    upper = np.full(10, np.inf)
    lower[:2] = np.finfo(float).eps
    # Refit all pinhole parameters after imposing skew=0, then release k1..k3,p1,p2.
    pinhole = least_squares(calibration_residuals, initial, args=(world_points, image_points),
                           bounds=(lower, upper), x_scale="jac", loss="linear",
                           max_nfev=calibration_max_evaluations)
    initial_distorted = np.concatenate([pinhole.x, np.zeros(5)])
    fit = least_squares(calibration_residuals, initial_distorted, args=(world_points, image_points),
                        bounds=(np.r_[lower, np.full(5, -np.inf)], np.r_[upper, np.full(5, np.inf)]),
                        x_scale="jac", loss="linear", max_nfev=calibration_max_evaluations,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    fx, fy, cx, cy = fit.x[:4]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    R, _ = cv2.Rodrigues(fit.x[4:7])
    t = fit.x[7:10]
    distortion = fit.x[10:15]
    residuals = fit.fun.reshape(-1, 2)
    camera_points = R @ world_points.T + t.reshape(3, 1)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    ideal_P = K @ H[:3]
    warnings = []
    if not pinhole.success or not fit.success:
        warnings.append("An optimizer did not converge; inspect its termination message.")
    if np.any(camera_points[2] <= 0):
        warnings.append("Some final camera depths are nonpositive; do not use this calibration.")
    if geometry_singular_values[-1] / geometry_singular_values[0] < 1e-3:
        warnings.append("Control geometry is nearly planar by the configured diagnostic ratio 1e-3.")
    column_norm = np.linalg.norm(fit.jac, axis=0)
    scaled_jacobian = fit.jac / np.maximum(column_norm, np.finfo(float).eps)
    jacobian_singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    radius_squared = np.sum((camera_points[:2] / camera_points[2]) ** 2, axis=0)
    r2 = np.linspace(0, np.max(radius_squared), 200)
    k1, k2, p1, p2, k3 = distortion
    radial_derivative = 1 + 3 * k1 * r2 + 5 * k2 * r2 ** 2 + 7 * k3 * r2 ** 3
    if np.any(radial_derivative <= 0):
        warnings.append("Radial mapping folds within the sampled control-point radius; distortion is unreliable.")
    result = {
        "K": K.tolist(), "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "distortion": distortion.tolist(),
        "distortion_named": {"k1": float(k1), "k2": float(k2), "k3": float(k3), "p1": float(p1), "p2": float(p2)},
        "R_world_to_camera": R.tolist(), "rvec_world_to_camera": fit.x[4:7].tolist(),
        "t_world_to_camera_m": t.tolist(), "camera_center_world_m": (-R.T @ t).tolist(),
        "H_camera_from_world": H.tolist(), "P_ideal_undistorted": ideal_P.tolist(),
        "P_dlt_raw_pixels": P.tolist(), "K_dlt_unconstrained": K_dlt.tolist(),
        "dlt_skew_px": float(K_dlt[0, 1]), "optimized_skew_px": 0.0,
        "R_dlt": R_dlt.tolist(), "t_dlt_m": t_dlt.tolist(),
        "pinhole_rmse_px": float(np.sqrt(np.mean(np.sum(pinhole.fun.reshape(-1, 2) ** 2, axis=1)))),
        "reprojection_rmse_px": float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))),
        "maximum_reprojection_error_px": float(np.max(np.linalg.norm(residuals, axis=1))),
        "point_count": len(world_points), "positive_depth_count": int(np.sum(camera_points[2] > 0)),
        "optimizer_converged": bool(fit.success), "optimizer_message": str(fit.message),
        "pinhole_optimizer_converged": bool(pinhole.success),
        "world_geometry_singular_values_m": geometry_singular_values.tolist(),
        "dlt_design_singular_values": design_singular_values.tolist(),
        "column_scaled_jacobian_singular_values": jacobian_singular_values.tolist(),
        "warnings": warnings,
        "fit_notes": "All control points retained with equal weight; residuals are in-sample, not independent accuracy. P_ideal excludes distortion.",
    }
    return result, residuals


def camera_parameter_vector(result):
    """Pack fx,fy,cx,cy, Rodrigues rotation, translation, k1,k2,p1,p2,k3."""
    K = np.asarray(result["K"])
    return np.concatenate(([K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                           result["rvec_world_to_camera"], result["t_world_to_camera_m"],
                           result["distortion"]))


def project_camera_points(camera, world_points):
    K = np.array([[camera[0], 0.0, camera[2]], [0.0, camera[1], camera[3]], [0.0, 0.0, 1.0]])
    projected, jacobian = cv2.projectPoints(world_points, camera[4:7], camera[7:10], K, camera[10:15])
    return projected.reshape(-1, 2), jacobian


def joint_calibration_residuals(parameters, image_points, gps_points):
    camera_count, point_count = image_points.shape[:2]
    cameras = parameters[:15 * camera_count].reshape(camera_count, 15)
    world_points = parameters[15 * camera_count:].reshape(point_count, 3)
    residuals = []
    for index, camera in enumerate(cameras):
        projected, _ = project_camera_points(camera, world_points)
        residuals.extend((projected - image_points[index]).ravel())
    residuals.extend(((world_points - gps_points) / gps_regularization_m).ravel())
    return np.asarray(residuals)


def joint_calibration_jacobian(parameters, image_points, gps_points):
    camera_count, point_count = image_points.shape[:2]
    camera_parameter_count = 15 * camera_count
    cameras = parameters[:camera_parameter_count].reshape(camera_count, 15)
    world_points = parameters[camera_parameter_count:].reshape(point_count, 3)
    image_row_count = 2 * camera_count * point_count
    jacobian = np.zeros((image_row_count + 3 * point_count, len(parameters)))
    # OpenCV columns: rvec, tvec, fx, fy, cx, cy, distortion.
    camera_column_order = [6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14]
    for camera_index, camera in enumerate(cameras):
        _, projection_jacobian = project_camera_points(camera, world_points)
        R, _ = cv2.Rodrigues(camera[4:7])
        # d(pixel)/d(world) = d(pixel)/d(camera XYZ) @ R.
        point_jacobian = projection_jacobian[:, 3:6] @ R
        row_start = 2 * camera_index * point_count
        column_start = 15 * camera_index
        jacobian[row_start:row_start + 2 * point_count, column_start:column_start + 15] = projection_jacobian[:, camera_column_order]
        for point in range(point_count):
            row = row_start + 2 * point
            column = camera_parameter_count + 3 * point
            jacobian[row:row + 2, column:column + 3] = point_jacobian[2 * point:2 * point + 2]
    jacobian[image_row_count:, camera_parameter_count:] = np.eye(3 * point_count) / gps_regularization_m
    return jacobian


def fit_camera_array(initial_cameras, gps_points, image_points, initial_points=None):
    """GPS-anchored bundle adjustment; no observations or 3D priors are deleted."""
    if gps_regularization_m <= 0 or joint_robust_loss_scale <= 0:
        raise ValueError("GPS regularization and robust loss scales must be positive.")
    camera_count = len(initial_cameras)
    if initial_points is None:
        initial_points = gps_points
    initial = np.concatenate((initial_cameras.ravel(), initial_points.ravel()))
    lower = np.full(len(initial), -np.inf)
    for camera in range(camera_count):
        lower[15 * camera:15 * camera + 2] = np.finfo(float).eps
    fit = least_squares(joint_calibration_residuals, initial, jac=joint_calibration_jacobian,
                        args=(image_points, gps_points), bounds=(lower, np.inf),
                        x_scale="jac", loss="soft_l1", f_scale=joint_robust_loss_scale,
                        max_nfev=calibration_max_evaluations, ftol=1e-9, xtol=1e-10, gtol=1e-8)
    cameras = fit.x[:15 * camera_count].reshape(camera_count, 15)
    world_points = fit.x[15 * camera_count:].reshape(-1, 3)
    # Always report ordinary pixel residuals, not robust costs or weighted RMS.
    image_residuals = fit.fun[:image_points.size].reshape(image_points.shape)
    return cameras, world_points, image_residuals, fit


def triangulation_residuals(world_point, cameras, observations):
    residuals = []
    for camera, observation in zip(cameras, observations):
        projected, _ = project_camera_points(camera, world_point.reshape(1, 3))
        residuals.extend(projected[0] - observation)
    return np.asarray(residuals)


def triangulate_camera_points(cameras, image_points):
    """CxNx2 original distorted pixels -> Nx3 world meters; no GPS used."""
    camera_count, point_count = image_points.shape[:2]
    if camera_count < 2 or camera_count != len(cameras):
        raise ValueError("Triangulation needs matching observations from at least two cameras.")
    normalized = []
    projections = []
    for camera, observations in zip(cameras, image_points):
        K = np.array([[camera[0], 0.0, camera[2]], [0.0, camera[1], camera[3]], [0.0, 0.0, 1.0]])
        xy = cv2.undistortPointsIter(observations.reshape(-1, 1, 2), K, camera[10:15], None, None,
                                     (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12))
        normalized.append(xy.reshape(-1, 2))
        R, _ = cv2.Rodrigues(camera[4:7])
        projections.append(np.column_stack((R, camera[7:10])))
    world_points = []
    for point in range(point_count):
        A = []
        for camera in range(camera_count):
            P = projections[camera]
            x, y = normalized[camera][point]
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        _, _, Vt = np.linalg.svd(A)
        homogeneous_point = Vt[-1]
        if abs(homogeneous_point[3]) < np.finfo(float).eps:
            raise ValueError("Triangulated point is at infinity; check camera baseline and correspondences.")
        initial = homogeneous_point[:3] / homogeneous_point[3]
        fit = least_squares(triangulation_residuals, initial, args=(cameras, image_points[:, point]),
                            x_scale="jac", max_nfev=100)
        if not fit.success:
            raise ValueError(f"Triangulation did not converge for point {point + 1}: {fit.message}")
        for P in projections:
            if (P[:, :3] @ fit.x + P[:, 3])[2] <= 0:
                raise ValueError(f"Triangulated point {point + 1} lies behind a camera.")
        world_points.append(fit.x)
    return np.asarray(world_points)


def validate_camera_array(gps_points, image_points):
    """Exclude entire target positions, fit cameras, predict one view from the other two."""
    camera_count, point_count = image_points.shape[:2]
    if camera_count != 3 or validation_folds < 2:
        raise ValueError("This validation requires three cameras and at least two folds.")
    rng = np.random.default_rng(validation_seed)
    folds = np.array_split(rng.permutation(point_count), validation_folds)
    residuals = np.zeros_like(image_points)
    world_errors = np.zeros_like(gps_points)
    fold_ids = np.zeros(point_count, dtype=int)
    fold_results = []
    for fold_index, test in enumerate(folds):
        train = np.setdiff1d(np.arange(point_count), test)
        initial_cameras = []
        for observations in image_points:
            initial_result, _ = calibrate_camera(gps_points[train], observations[train])
            initial_cameras.append(camera_parameter_vector(initial_result))
        cameras, _, _, fit = fit_camera_array(np.asarray(initial_cameras), gps_points[train], image_points[:, train])
        # Neither the test pixels nor their GPS coordinates enter camera fitting.
        reconstructed = triangulate_camera_points(cameras, image_points[:, test])
        world_errors[test] = reconstructed - gps_points[test]
        for camera in range(camera_count):
            other_cameras = np.array([index for index in range(camera_count) if index != camera])
            reconstructed = triangulate_camera_points(cameras[other_cameras], image_points[other_cameras][:, test])
            predicted, _ = project_camera_points(cameras[camera], reconstructed)
            residuals[camera, test] = predicted - image_points[camera, test]
        fold_ids[test] = fold_index + 1
        fold_results.append({"fold": fold_index + 1, "test_point_ids": (test + 1).tolist(),
                             "optimizer_converged": bool(fit.success), "optimizer_message": fit.message,
                             "camera_parameters": cameras.tolist()})
        print(f"Held-out validation: fold {fold_index + 1}/{validation_folds}; converged={fit.success}", flush=True)
    summary = {
        "method": "Position-wise cross-validation: fit on training positions only; triangulate test points from two cameras, predict the third without its observation or GPS.",
        "seed": validation_seed, "folds": fold_results,
        "all_folds_converged": all(fold["optimizer_converged"] for fold in fold_results),
        "per_camera_rmse_px": np.sqrt(np.mean(np.sum(residuals ** 2, axis=2), axis=1)).tolist(),
        "per_camera_maximum_error_px": np.max(np.linalg.norm(residuals, axis=2), axis=1).tolist(),
        "gps_comparison_3d_rms_m": float(np.sqrt(np.mean(np.sum(world_errors ** 2, axis=1)))),
        "gps_comparison_3d_median_m": float(np.median(np.linalg.norm(world_errors, axis=1))),
        "gps_comparison_3d_maximum_m": float(np.max(np.linalg.norm(world_errors, axis=1))),
        "limitations": "Internal validation of this capture volume, not independent metric ground truth. GPS comparison includes GPS/antenna/timing error. Settings were explored on this dataset.",
    }
    return summary, residuals, world_errors, fold_ids


def bootstrap_camera_array(cameras, adjusted_points, gps_points, image_points, run_dir):
    """Resample complete target positions, retaining all three views in each draw."""
    if bootstrap_repetitions == 0:
        return {"status": "disabled", "repetitions": 0}
    if bootstrap_repetitions < 2:
        raise ValueError("Use at least two bootstrap repetitions, or 0 to disable.")
    rng = np.random.default_rng(bootstrap_seed)
    camera_draws = []
    point_draws = []
    sample_indices = []
    converged = []
    for repetition in range(bootstrap_repetitions):
        indices = rng.integers(0, len(gps_points), len(gps_points))
        sampled_cameras, _, _, fit = fit_camera_array(cameras, gps_points[indices],
                                                      image_points[:, indices], adjusted_points[indices])
        # Reconstruct fixed original observations; their GPS is not used here.
        reconstructed = triangulate_camera_points(sampled_cameras, image_points)
        camera_draws.append(sampled_cameras)
        point_draws.append(reconstructed)
        sample_indices.append(indices)
        converged.append(bool(fit.success))
        if (repetition + 1) % 10 == 0 or repetition + 1 == bootstrap_repetitions:
            print(f"Bootstrap: {repetition + 1}/{bootstrap_repetitions}", flush=True)
    camera_draws = np.asarray(camera_draws)
    point_draws = np.asarray(point_draws)
    # Save every draw, including any nonconverged fits; never silently discard.
    np.savez_compressed(run_dir / "bootstrap_samples.npz", camera_parameters=camera_draws,
                        reconstructed_world_points_m=point_draws,
                        sampled_point_ids=np.asarray(sample_indices) + 1, converged=converged)
    quantiles = np.quantile(camera_draws, [0.025, 0.5, 0.975], axis=0)
    point_std = np.std(point_draws, axis=0, ddof=1)
    with (run_dir / "bootstrap_reconstruction_spread.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "std_E_m", "std_N_m", "std_U_m", "std_vector_norm_m"])
        for point, std in enumerate(point_std):
            writer.writerow([point + 1, *std, np.linalg.norm(std)])
    return {
        "repetitions": bootstrap_repetitions, "seed": bootstrap_seed,
        "converged_count": sum(converged), "all_converged": all(converged),
        "parameter_order": ["fx", "fy", "cx", "cy", "rx", "ry", "rz", "tx_m", "ty_m", "tz_m", "k1", "k2", "p1", "p2", "k3"],
        "percentiles": [2.5, 50, 97.5], "camera_parameter_percentiles": quantiles.tolist(),
        "median_reconstruction_std_norm_m": float(np.median(np.linalg.norm(point_std, axis=1))),
        "maximum_reconstruction_std_norm_m": float(np.max(np.linalg.norm(point_std, axis=1))),
        "limitations": "Calibration resampling stability conditional on this dataset and regularization, not absolute accuracy. Systematic GPS errors, target attitude, image localization noise on future points and extrapolation are not included. Percentiles include all draws; nonconvergence invalidates their interpretation.",
    }


def camera_distortion_diagnostics(camera, image_size):
    width, height = image_size
    K = np.array([[camera[0], 0.0, camera[2]], [0.0, camera[1], camera[3]], [0.0, 0.0, 1.0]])
    grid = np.stack(np.meshgrid(np.linspace(0, width - 1, 25), np.linspace(0, height - 1, 19)), axis=-1).reshape(-1, 2)
    xy = cv2.undistortPointsIter(grid.reshape(-1, 1, 2), K, camera[10:15], None, None,
                                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)).reshape(-1, 2)
    k1, k2, p1, p2, k3 = camera[10:15]
    radius_squared = np.linspace(0, np.max(np.sum(xy ** 2, axis=1)), 1000)
    radial_derivative = 1 + 3 * k1 * radius_squared + 5 * k2 * radius_squared ** 2 + 7 * k3 * radius_squared ** 3
    x, y = xy.T
    r2 = x ** 2 + y ** 2
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    gradient = k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2
    jxx = radial + 2 * x ** 2 * gradient + 2 * p1 * y + 6 * p2 * x
    jyy = radial + 2 * y ** 2 * gradient + 6 * p1 * y + 2 * p2 * x
    jxy = 2 * x * y * gradient + 2 * p1 * x + 2 * p2 * y
    determinants = jxx * jyy - jxy ** 2
    rays = np.column_stack((xy, np.ones(len(xy))))
    projected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, camera[10:15])
    roundtrip_error = np.linalg.norm(projected.reshape(-1, 2) - grid, axis=1)
    return {"minimum_radial_derivative": float(np.min(radial_derivative)),
            "minimum_distortion_jacobian_determinant": float(np.min(determinants)),
            "image_grid_roundtrip_max_error_px": float(np.max(roundtrip_error)),
            "passed": bool(np.all(radial_derivative > 0) and np.all(determinants > 0)
                           and np.all(roundtrip_error < 1e-5)),
            "note": "Numerical grid check, not evidence of accuracy outside the control-point coverage."}


def enable_residual_plot_scaling(fig, residual_ax, parameter_ax):
    """Scale all plot elements from their original sizes when the window changes."""
    base_size = fig.get_size_inches().copy()
    texts = list(fig.texts) + list(residual_ax.texts) + list(parameter_ax.texts)
    texts.extend([residual_ax.title, residual_ax.xaxis.label, residual_ax.yaxis.label,
                  residual_ax.xaxis.get_offset_text(), residual_ax.yaxis.get_offset_text()])
    text_sizes = [(text, text.get_fontsize()) for text in texts]
    line_widths = []
    for line in list(residual_ax.lines) + list(residual_ax.spines.values()):
        line_widths.append((line, line.get_linewidth()))
    scatter_sizes = []
    for scatter in residual_ax.collections:
        scatter_sizes.append((scatter, scatter.get_sizes().copy(), scatter.get_linewidths().copy()))
    tick_sizes = []
    for axis in (residual_ax.xaxis, residual_ax.yaxis):
        tick = axis.get_major_ticks()[0]
        tick_sizes.append((axis, tick.label1.get_fontsize(), tick.tick1line.get_markersize(),
                           tick.tick1line.get_markeredgewidth(), tick.get_pad(), axis.labelpad))
    grid_width = residual_ax.get_xgridlines()[0].get_linewidth()
    title_size = residual_ax.title.get_fontsize()
    title_pad = plt.rcParams["axes.titlepad"]
    legend = residual_ax.get_legend()
    legend_size = None if legend is None else legend.get_texts()[0].get_fontsize()
    legend_width = None if legend is None else legend.get_frame().get_linewidth()
    layout = fig.get_layout_engine()
    layout_pads = layout.get().copy()

    def resize_contents(event):
        if event.width <= 0 or event.height <= 0 or fig.canvas.is_saving():
            return
        # Use the limiting dimension to preserve proportions in wide/tall windows.
        # Always refer to the original sizes, so repeated resizing cannot compound.
        current_size = fig.get_size_inches()
        scale = min(current_size[0] / base_size[0], current_size[1] / base_size[1])
        for text, size in text_sizes:
            text.set_fontsize(size * scale)
        for line, width in line_widths:
            line.set_linewidth(width * scale)
        for scatter, sizes, widths in scatter_sizes:
            # Matplotlib scatter sizes are areas in points squared.
            scatter.set_sizes(sizes * scale ** 2)
            scatter.set_linewidths(widths * scale)
        for axis, size, length, width, pad, labelpad in tick_sizes:
            axis.set_tick_params(which="major", labelsize=size * scale, length=length * scale,
                                 width=width * scale, pad=pad * scale)
            axis.labelpad = labelpad * scale
        residual_ax.grid(True, linewidth=grid_width * scale)
        residual_ax.set_title(residual_ax.get_title(), fontsize=title_size * scale, pad=title_pad * scale)
        if legend_size is not None:
            # Rebuilding also scales legend spacing and marker handles correctly.
            resized_legend = residual_ax.legend(loc="upper left", fontsize=legend_size * scale,
                                                 handlelength=1, handletextpad=0.4, borderpad=0.4)
            resized_legend.get_frame().set_linewidth(legend_width * scale)
        layout.set(w_pad=layout_pads["w_pad"] * scale, h_pad=layout_pads["h_pad"] * scale)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("resize_event", resize_contents)


def plot_camera_residuals(camera, result, residuals, run_dir, axis_limit, validation_residuals=None):
    """Plot signed projected-minus-observed residuals and final camera parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(plot_width_cm / 2.54, plot_height_cm / 2.54),
                             gridspec_kw={"width_ratios": [1, 1.1]},
                             constrained_layout=True)
    fig.canvas.manager.set_window_title(f"{camera} - joint calibration and validation")
    fig.suptitle(f"{camera}: calibration residuals", fontsize=plot_font_size + 3)
    residual_ax, parameter_ax = axes
    residual_ax.scatter(residuals[:, 0], residuals[:, 1], s=18,
                        color="tab:blue", alpha=0.8, label="BA: adjusted 3D points")
    if validation_residuals is not None:
        residual_ax.scatter(validation_residuals[:, 0], validation_residuals[:, 1], s=18,
                            marker="^", color="tab:orange", alpha=0.7, label="Held-out prediction")
        residual_ax.legend(loc="upper left", fontsize=plot_parameter_font_size,
                           handlelength=1, handletextpad=0.4, borderpad=0.4)
    residual_ax.axhline(0, color="0.35", linewidth=1)
    residual_ax.axvline(0, color="0.35", linewidth=1)
    residual_ax.set_xlim(-axis_limit, axis_limit)
    residual_ax.set_ylim(-axis_limit, axis_limit)
    residual_ax.set_aspect("equal", adjustable="box")
    residual_ax.xaxis.set_major_formatter(FormatStrFormatter("%+g"))
    residual_ax.yaxis.set_major_formatter(FormatStrFormatter("%+g"))
    residual_ax.set_xlabel("Horizontal residual, du (px)", fontsize=plot_font_size)
    residual_ax.set_ylabel("Vertical residual, dv (px)", fontsize=plot_font_size)
    residual_ax.tick_params(axis="both", labelsize=plot_font_size)
    residual_ax.grid(True, alpha=0.25)
    residual_ax.set_axisbelow(True)
    mean_residual = residuals.mean(axis=0)
    title = f"N = {len(residuals)} | BA RMSE = {result['reprojection_rmse_px']:.3f} px\n"
    if validation_residuals is None:
        title += f"Mean (du, dv): ({mean_residual[0]:+.3f}, {mean_residual[1]:+.3f}) px"
    else:
        validation_rmse = np.sqrt(np.mean(np.sum(validation_residuals ** 2, axis=1)))
        title += f"Held-out RMSE = {validation_rmse:.3f} px"
    residual_ax.set_title(title, fontsize=plot_font_size)
    fig.supxlabel("Residual = projected - observed (original pixels)\n"
                  "+du: right; +dv: down in image. All points retained; GPS soft constraint.",
                  fontsize=plot_parameter_font_size)

    # Display the refined K, R, t, not the unconstrained DLT initialization.
    K = np.asarray(result["K"])
    R = np.asarray(result["R_world_to_camera"])
    t = result["t_world_to_camera_m"]
    distortion = result["distortion_named"]
    # Round display text only; JSON/CSV retain full numerical precision.
    lines = ["Final optimized parameters", "", "K (focal / principal point: px)"]
    for row in K:
        lines.append(f"[{row[0]:9.3f} {row[1]:9.3f} {row[2]:9.3f}]")
    lines.extend([f"skew = {K[0, 1]:.3f} px (fixed)", "",
                  "Extrinsics: X_cam = R X_world + t", "R ="])
    for row in R:
        lines.append(f"[{row[0]:+9.5f} {row[1]:+9.5f} {row[2]:+9.5f}]")
    lines.extend(["t (m) =", f"[{t[0]:+9.5f} {t[1]:+9.5f} {t[2]:+9.5f}]", "",
                  "Distortion (dimensionless)"])
    for name in ("k1", "k2", "k3", "p1", "p2"):
        lines.append(f"{name} = {distortion[name]:+.6g}")
    lines.extend(["", "World (m): East, North, rel. alt.",
                  "Camera: right, down, forward"])
    parameter_ax.set_axis_off()
    parameter_ax.text(0.02, 0.97, "\n".join(lines), transform=parameter_ax.transAxes,
                      va="top", ha="left", fontfamily="monospace", fontsize=plot_parameter_font_size,
                      linespacing=1.2)
    # Keep exact page dimensions; a tight bounding box would change Word scaling.
    fig.savefig(run_dir / f"{camera}_residual_scatter.png", dpi=plot_export_dpi, facecolor="white")
    with plt.rc_context({"svg.fonttype": "path"}):
        fig.savefig(run_dir / f"{camera}_residual_scatter.svg", facecolor="white")
    # Save the standard Word files first; interactive resizing affects the window.
    enable_residual_plot_scaling(fig, residual_ax, parameter_ax)
    return fig


def main():
    gps_rows, antenna_points, metadata = load_gps_points(gps_file)
    camera_files = {}
    for camera, prefix in camera_prefixes.items():
        camera_files[camera] = find_camera_images(data_dir / camera, prefix, len(gps_rows))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir()
    points, sizes, cache = review_circle_centers(camera_files, output_dir / "reviewed_centers.json", run_dir)
    if points is None:
        return
    with (run_dir / "image_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["camera", "point_id", "filename", "u_px", "v_px", "method"])
        for camera, files in camera_files.items():
            for index, path in enumerate(files):
                target = cache["cameras"][camera][path.name]["target"]
                writer.writerow([camera, index + 1, path.name, *points[camera][index], target["method"]])
    if antenna_to_circle_enu_m is None:
        metadata["status"] = "Image review complete; calibration awaits measured antenna-to-circle offsets."
        (run_dir / "pending_calibration.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("\nImage centers saved:", run_dir / "image_points.csv")
        print("Calibration not run: GPS points are antenna locations. Set antenna_to_circle_enu_m first.")
        return
    offsets = np.asarray(antenna_to_circle_enu_m, dtype=float)
    if offsets.shape == (3,):
        offsets = np.tile(offsets, (len(antenna_points), 1))
    if offsets.shape != antenna_points.shape or not np.isfinite(offsets).all():
        raise ValueError("antenna_to_circle_enu_m must be a finite 3-vector or N by 3 array in WORLD E/N/U meters.")
    world_points = antenna_points + offsets
    metadata["antenna_to_circle_enu_m"] = offsets.tolist()
    metadata["offset_assumption"] = "User-configured antenna-to-circle displacement in local WORLD axes; no target-attitude correction applied."
    metadata["circle_center_perspective_bias"] = "Image ellipse centers are used; circle eccentricity is not corrected."
    with (run_dir / "world_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "save_timestamp", "utc_time", "fix_quality", "latitude_deg", "longitude_deg", "gga_altitude_m",
                         "antenna_E_m", "antenna_N_m", "antenna_U_m", "circle_E_m", "circle_N_m", "circle_U_m"])
        for index, row in enumerate(gps_rows):
            writer.writerow([index + 1, row.get("save_timestamp", ""), row.get("utc_time", ""),
                             row.get("fix_quality", ""), row["latitude"], row["longitude"], row["altitude_m"],
                             *antenna_points[index], *world_points[index]])
    initial_results = {}
    initial_cameras = []
    original_residuals = []
    for camera in camera_files:
        result, residuals = calibrate_camera(world_points, points[camera])
        result["image_size_width_height"] = list(sizes[camera])
        initial_results[camera] = result
        initial_cameras.append(camera_parameter_vector(result))
        original_residuals.append(residuals)
        print(f"{camera}: fixed-GPS initialization RMSE = {result['reprojection_rmse_px']:.4f} px", flush=True)
    (run_dir / "fixed_gps_calibration_results.json").write_text(
        json.dumps({"metadata": metadata, "cameras": initial_results}, indent=2), encoding="utf-8")

    observations = np.asarray([points[camera] for camera in camera_files])
    initial_cameras = np.asarray(initial_cameras)
    original_residuals = np.asarray(original_residuals)
    print("\nJoint camera/target optimization with soft GPS constraints...", flush=True)
    cameras, adjusted_points, all_residuals, fit = fit_camera_array(initial_cameras, world_points, observations)
    if not fit.success:
        raise RuntimeError(f"Joint calibration did not converge: {fit.message}")
    gps_corrections = adjusted_points - world_points
    gps_distances = np.linalg.norm(gps_corrections, axis=1)
    with (run_dir / "adjusted_world_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "gps_circle_E_m", "gps_circle_N_m", "gps_circle_U_m",
                         "adjusted_circle_E_m", "adjusted_circle_N_m", "adjusted_circle_U_m",
                         "delta_E_m", "delta_N_m", "delta_U_m", "correction_norm_m"])
        for index in range(len(world_points)):
            writer.writerow([index + 1, *world_points[index], *adjusted_points[index],
                             *gps_corrections[index], gps_distances[index]])

    metadata["calibration_method"] = "DLT initialization followed by joint camera/3D-target bundle adjustment with soft GPS constraints"
    metadata["gps_regularization_m"] = gps_regularization_m
    metadata["gps_weight_meaning"] = "Algorithmic regularization: this many meters has the weight of 1 pixel. Not a measured GPS standard deviation."
    metadata["loss"] = "soft_l1; all image observations and all GPS priors retained"
    metadata["loss_scale"] = joint_robust_loss_scale
    metadata["reprojection_world_points_file"] = "adjusted_world_points.csv"
    metadata["raw_gps_prior_file"] = "world_points.csv"
    results = {"metadata": metadata, "cameras": {}}
    diagnostics = {
        "camera_order": list(camera_files), "optimizer_converged": bool(fit.success),
        "optimizer_message": fit.message, "optimizer_evaluations": fit.nfev,
        "residual_correlation_u_before_bundle": np.corrcoef(original_residuals[:, :, 0]).tolist(),
        "residual_correlation_v_before_bundle": np.corrcoef(original_residuals[:, :, 1]).tolist(),
        "gps_correction_median_m": float(np.median(gps_distances)),
        "gps_correction_rms_m": float(np.sqrt(np.mean(gps_distances ** 2))),
        "gps_correction_maximum_m": float(np.max(gps_distances)),
        "largest_gps_correction_point_ids": (np.argsort(gps_distances)[-10:][::-1] + 1).tolist(),
        "metric_accuracy_verified": False,
        "interpretation": "Shared image residuals indicate disagreement between measured antenna-based GPS targets and image-derived target positions. This does not identify whether timing, motion, antenna offset, or GPS error is responsible. A constant 4 cm world offset is absorbed by translation and cannot remove point-dependent errors.",
        "scope": "Assumes fixed intrinsics/extrinsics and the same stationary target position across all three views of each ID. No synchronized exposure timestamps or independent metric checkpoints are available.",
    }
    scaled_jacobian = fit.jac / np.maximum(np.linalg.norm(fit.jac, axis=0), np.finfo(float).eps)
    diagnostics["joint_scaled_jacobian_singular_values"] = np.linalg.svd(scaled_jacobian, compute_uv=False).tolist()
    residuals_by_camera = {}
    for camera_index, camera in enumerate(camera_files):
        parameters = cameras[camera_index]
        residuals = all_residuals[camera_index]
        K = np.array([[parameters[0], 0.0, parameters[2]], [0.0, parameters[1], parameters[3]], [0.0, 0.0, 1.0]])
        R, _ = cv2.Rodrigues(parameters[4:7])
        t = parameters[7:10]
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t
        depths = (R @ adjusted_points.T + t.reshape(3, 1))[2]
        gps_projected, _ = project_camera_points(parameters, world_points)
        gps_pixel_residuals = gps_projected - points[camera]
        distortion_diagnostics = camera_distortion_diagnostics(parameters, sizes[camera])
        hull = cv2.convexHull(points[camera].astype(np.float32)).reshape(-1, 2)
        warnings = []
        if np.any(depths <= 0):
            warnings.append("Some adjusted control points lie behind the camera; do not use this result.")
        if not distortion_diagnostics["passed"]:
            warnings.append("Full-image distortion invertibility check failed; do not extrapolate this model.")
        warnings.append("Small image residuals do not establish absolute 3D accuracy; use validation and bootstrap diagnostics.")
        result = {
            "K": K.tolist(), "R_world_to_camera": R.tolist(),
            "rvec_world_to_camera": parameters[4:7].tolist(), "t_world_to_camera_m": t.tolist(),
            "camera_center_world_m": (-R.T @ t).tolist(), "H_camera_from_world": H.tolist(),
            "P_ideal_undistorted": (K @ H[:3]).tolist(),
            "distortion": parameters[10:15].tolist(), "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
            "distortion_named": {"k1": parameters[10], "k2": parameters[11], "p1": parameters[12],
                                 "p2": parameters[13], "k3": parameters[14]},
            "optimized_skew_px": 0.0, "dlt_skew_px": initial_results[camera]["dlt_skew_px"],
            "P_dlt_raw_pixels": initial_results[camera]["P_dlt_raw_pixels"],
            "K_dlt_unconstrained": initial_results[camera]["K_dlt_unconstrained"],
            "image_size_width_height": list(sizes[camera]),
            "reprojection_rmse_px": float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))),
            "maximum_reprojection_error_px": float(np.max(np.linalg.norm(residuals, axis=1))),
            "fixed_gps_rmse_px_before_bundle": initial_results[camera]["reprojection_rmse_px"],
            "original_gps_rmse_px_after_bundle": float(np.sqrt(np.mean(np.sum(gps_pixel_residuals ** 2, axis=1)))),
            "point_count": len(world_points), "positive_depth_count": int(np.sum(depths > 0)),
            "optimizer_converged": bool(fit.success), "optimizer_message": fit.message,
            "distortion_diagnostics": distortion_diagnostics, "control_image_hull_px": hull.tolist(),
            "warnings": warnings,
            "fit_notes": "Ordinary unweighted pixel RMS on adjusted shared 3D targets, using all observations. Raw GPS remains a soft prior. The original GPS RMS is reported separately. P_ideal excludes distortion.",
        }
        results["cameras"][camera] = result
        residuals_by_camera[camera] = residuals
        with (run_dir / f"{camera}_residuals.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["point_id", "u_observed_px", "v_observed_px", "u_projected_px", "v_projected_px",
                             "du_projected_minus_observed_px", "dv_projected_minus_observed_px", "error_px",
                             "du_original_gps_px", "dv_original_gps_px", "original_gps_error_px"])
            projected = points[camera] + residuals
            for index in range(len(world_points)):
                writer.writerow([index + 1, *points[camera][index], *projected[index], *residuals[index],
                                 np.linalg.norm(residuals[index]), *gps_pixel_residuals[index],
                                 np.linalg.norm(gps_pixel_residuals[index])])
        print(f"\n{camera}: joint BA RMSE={result['reprojection_rmse_px']:.4f} px; final skew=0 px")
        print(f"Original GPS prior projected with final camera: RMSE={result['original_gps_rmse_px_after_bundle']:.4f} px")
        print("K =\n", np.array(result["K"]))
        print("[k1, k2, p1, p2, k3] =", result["distortion"])
        print("R =\n", np.array(result["R_world_to_camera"]))
        print("t (m) =", result["t_world_to_camera_m"])
        for warning in result["warnings"]:
            print("CHECK:", warning)
    (run_dir / "calibration_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "calibration_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print("\nValidating on held-out positions...", flush=True)
    validation, validation_residuals, validation_world_errors, fold_ids = validate_camera_array(world_points, observations)
    diagnostics["cross_validation"] = validation
    with (run_dir / "held_out_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["camera", "point_id", "fold", "du_predicted_minus_observed_px", "dv_predicted_minus_observed_px",
                         "error_px", "three_view_minus_gps_E_m", "three_view_minus_gps_N_m", "three_view_minus_gps_U_m"])
        for camera_index, camera in enumerate(camera_files):
            results["cameras"][camera]["held_out_prediction_rmse_px"] = validation["per_camera_rmse_px"][camera_index]
            for point in range(len(world_points)):
                residual = validation_residuals[camera_index, point]
                writer.writerow([camera, point + 1, fold_ids[point], *residual, np.linalg.norm(residual),
                                 *validation_world_errors[point]])
    diagnostics["all_camera_validation_rmse_below_1px"] = bool(
        validation["all_folds_converged"] and np.all(np.asarray(validation["per_camera_rmse_px"]) < 1.0))
    (run_dir / "calibration_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "calibration_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    diagnostics["bootstrap"] = bootstrap_camera_array(cameras, adjusted_points, world_points, observations, run_dir)
    (run_dir / "calibration_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print("\nHeld-out prediction RMSE (px):", validation["per_camera_rmse_px"])
    print(f"GPS correction: median={np.median(gps_distances):.4f} m; maximum={np.max(gps_distances):.4f} m")
    print("3D absolute accuracy is not verified by these image residuals.")
    print("\nCalibration output:", run_dir, flush=True)
    # A common symmetric scale makes the three signed residual clouds comparable.
    axis_limit = 0.0
    for residuals in residuals_by_camera.values():
        axis_limit = max(axis_limit, float(np.max(np.abs(residuals))))
    axis_limit = max(axis_limit, float(np.max(np.abs(validation_residuals))))
    axis_limit = max(1.0, 1.1 * axis_limit)
    for camera_index, (camera, residuals) in enumerate(residuals_by_camera.items()):
        plot_camera_residuals(camera, results["cameras"][camera], residuals, run_dir, axis_limit,
                              validation_residuals[camera_index])
    plt.show()


if __name__ == "__main__":
    main()
