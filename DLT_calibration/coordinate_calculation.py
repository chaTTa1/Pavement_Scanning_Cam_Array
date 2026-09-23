"""Calculate GPS, IMU, and camera coordinate transformations using manual 2D pattern points.

Transform names follow ``H_target_from_source``. Homogeneous column points are
transformed as

    point_target = H_target_from_source @ point_source

GPS coordinates are converted to a local East-North-Up (ENU) frame before any
camera calibration. A complete GPS-to-IMU transform requires the IMU attitude
and the physical lever arm from the IMU origin to the auxiliary GPS antenna.
Two stationary GPS antennas alone determine only a baseline, not a full 3-D
rotation.

Manual Pattern Points Input Format:
-----------------------------------
The target CSV file (``camera_antenna_point.csv``) contains 3D geodetic target 
positions (latitude, longitude, ellipsoid height). Corresponding 2D image coordinates
are loaded from the target JSON file (``manual_2d_coordinates.json``) for each camera.

One camera target row supplies one 3-D GPS point and the corresponding manual 2-D pixel center.
Cameras must remain fixed during calibration, and collected points must not all be coplanar.
Distortion uses the Brown-Conrady model [k1, k2, p1, p2, k3].
"""

import csv
import json
import math
from pathlib import Path
import re
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# -----------------------------------------------------------------------------
# Input and output paths
# -----------------------------------------------------------------------------
def read_manual_camera_target_json(json_path, camera_names):
    """Load manual 2D pixel coordinates from a JSON file into NumPy arrays."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    def natural_sort_key(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else filename

    camera_points_dict = {}
    for camera_name in camera_names:
        if camera_name not in data:
            raise KeyError(f"Camera '{camera_name}' not found in JSON file.")
        
        # Sort using natural index order to handle mix of .jpg, .jpeg, and padding
        target_dict = data[camera_name]
        sorted_targets = sorted(target_dict.keys(), key=natural_sort_key)
        points = [target_dict[target] for target in sorted_targets]
        
        camera_points_dict[camera_name] = np.asarray(points, dtype=float)
        
    return camera_points_dict


script_directory = Path(__file__).resolve().parent

gps_points = script_directory / "DLT_test_Sept10" / "gps_main.csv"
imu_points = script_directory / "DLT_test_Sept10" / "gps_imu_antenna.csv"
camera_antenna_point = script_directory / "DLT_test_Sept10" / "camera_antenna_point.csv"

# Relative Path to JSON file
manual_json_path = script_directory / "DLT_test_Sept10" / "manual_2d_coordinates.json"

camera_names = ["camera_1", "camera_2", "camera_3"]

output_json = script_directory / "coordinate_transforms.json"


# A complete GPS-to-IMU transform cannot be recovered from two stationary GPS
# positions alone. Replace these two values using the measured installation.
# rotation_imu_from_gps maps ENU vectors into the IMU coordinate axes.

imu_origin_to_auxiliary_antenna_imu = [0.27, 0.0, -0.06]
r_imu = R.from_euler('zyx', [0.16239, -0.00816, -0.01963], degrees=False)
R_ned_from_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
rotation_imu_from_gps = r_imu.as_matrix() @ R_ned_from_enu

# Calibration and bootstrap settings.
bootstrap_iterations = 200
gps_bootstrap_block_length = 1
random_seed = 20260827
confidence_level = 0.95

# Set a value only when both GPS files contain timestamps and the platform was
# moving. For static calibration, None independently averages the recordings.
gps_maximum_time_difference_seconds = None


# Nonlinear refinement settings used after the linear DLT initialization.
calibration_maximum_iterations = 100
calibration_initial_damping = 1e-3
calibration_finite_difference_step = 1e-6
calibration_convergence_tolerance = 1e-9


WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_ECCENTRICITY_SQUARED = (
    WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
)


def validate_point_array(points, dimension, name, minimum_count=1):
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != dimension:
        raise ValueError(
            f"{name} must have shape N by {dimension}; received {points.shape}"
        )

    if points.shape[0] < minimum_count:
        raise ValueError(
            f"{name} needs at least {minimum_count} rows; "
            f"received {points.shape[0]}"
        )

    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains NaN or infinite values")

    return points


def geodetic_to_ecef(geodetic_points):
    """Convert latitude, longitude, ellipsoid height to WGS84 ECEF meters."""
    geodetic_points = validate_point_array(
        geodetic_points,
        3,
        "geodetic_points",
    )

    latitude_deg = geodetic_points[:, 0]
    longitude_deg = geodetic_points[:, 1]
    height_m = geodetic_points[:, 2]

    if np.any(latitude_deg < -90.0) or np.any(latitude_deg > 90.0):
        raise ValueError("latitude must be between -90 and 90 degrees")

    if np.any(longitude_deg < -180.0) or np.any(longitude_deg > 180.0):
        raise ValueError("longitude must be between -180 and 180 degrees")

    latitude_rad = np.radians(latitude_deg)
    longitude_rad = np.radians(longitude_deg)

    sin_latitude = np.sin(latitude_rad)
    cos_latitude = np.cos(latitude_rad)
    sin_longitude = np.sin(longitude_rad)
    cos_longitude = np.cos(longitude_rad)

    prime_vertical_radius = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - WGS84_ECCENTRICITY_SQUARED * sin_latitude ** 2
    )

    x = (prime_vertical_radius + height_m) * cos_latitude * cos_longitude
    y = (prime_vertical_radius + height_m) * cos_latitude * sin_longitude
    z = (
        prime_vertical_radius * (1.0 - WGS84_ECCENTRICITY_SQUARED)
        + height_m
    ) * sin_latitude

    return np.column_stack([x, y, z])


def ecef_to_geodetic(ecef_points, maximum_iterations=20, tolerance_rad=1e-13):
    """Convert WGS84 ECEF meters to latitude, longitude, ellipsoid height."""
    ecef_points = validate_point_array(ecef_points, 3, "ecef_points")

    x = ecef_points[:, 0]
    y = ecef_points[:, 1]
    z = ecef_points[:, 2]
    horizontal_distance = np.sqrt(x ** 2 + y ** 2)

    if np.any(horizontal_distance < 1e-9):
        raise ValueError("ECEF points on the polar axis are not supported")

    longitude_rad = np.arctan2(y, x)
    latitude_rad = np.arctan2(
        z,
        horizontal_distance * (1.0 - WGS84_ECCENTRICITY_SQUARED),
    )

    for _ in range(maximum_iterations):
        sin_latitude = np.sin(latitude_rad)
        prime_vertical_radius = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(
            1.0 - WGS84_ECCENTRICITY_SQUARED * sin_latitude ** 2
        )

        height_m = horizontal_distance / np.cos(latitude_rad) - prime_vertical_radius

        denominator = horizontal_distance * (
            1.0
            - WGS84_ECCENTRICITY_SQUARED
            * prime_vertical_radius
            / (prime_vertical_radius + height_m)
        )
        updated_latitude_rad = np.arctan2(z, denominator)

        maximum_change = np.max(np.abs(updated_latitude_rad - latitude_rad))
        latitude_rad = updated_latitude_rad

        if maximum_change <= tolerance_rad:
            break
    else:
        raise ValueError("ECEF to geodetic conversion did not converge")

    sin_latitude = np.sin(latitude_rad)
    prime_vertical_radius = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - WGS84_ECCENTRICITY_SQUARED * sin_latitude ** 2
    )
    height_m = horizontal_distance / np.cos(latitude_rad) - prime_vertical_radius

    return np.column_stack([
        np.degrees(latitude_rad),
        np.degrees(longitude_rad),
        height_m,
    ])


def mean_geodetic_position(geodetic_points):
    """Compute a mean geodetic location by averaging in ECEF coordinates."""
    ecef_points = geodetic_to_ecef(geodetic_points)
    mean_ecef = np.mean(ecef_points, axis=0, keepdims=True)
    return ecef_to_geodetic(mean_ecef)[0]


def ecef_to_enu(ecef_points, reference_geodetic):
    """Convert ECEF points to local ENU meters at one geodetic reference."""
    ecef_points = validate_point_array(ecef_points, 3, "ecef_points")
    reference_geodetic = np.asarray(reference_geodetic, dtype=float).reshape(-1)

    if reference_geodetic.shape != (3,):
        raise ValueError("reference_geodetic must contain latitude, longitude, height")

    reference_ecef = geodetic_to_ecef(reference_geodetic.reshape(1, 3))[0]

    latitude_rad = math.radians(reference_geodetic[0])
    longitude_rad = math.radians(reference_geodetic[1])

    sin_latitude = math.sin(latitude_rad)
    cos_latitude = math.cos(latitude_rad)
    sin_longitude = math.sin(longitude_rad)
    cos_longitude = math.cos(longitude_rad)

    rotation_enu_from_ecef = np.array(
        [
            [-sin_longitude, cos_longitude, 0.0],
            [
                -sin_latitude * cos_longitude,
                -sin_latitude * sin_longitude,
                cos_latitude,
            ],
            [
                cos_latitude * cos_longitude,
                cos_latitude * sin_longitude,
                sin_latitude,
            ],
        ],
        dtype=float,
    )

    ecef_difference = ecef_points - reference_ecef.reshape(1, 3)
    return (rotation_enu_from_ecef @ ecef_difference.T).T


def geodetic_to_enu(geodetic_points, reference_geodetic):
    ecef_points = geodetic_to_ecef(geodetic_points)
    return ecef_to_enu(ecef_points, reference_geodetic)


def validate_rotation_matrix(rotation_matrix, name="rotation_matrix"):
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)

    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape 3 by 3")

    if not np.all(np.isfinite(rotation_matrix)):
        raise ValueError(f"{name} contains NaN or infinite values")

    identity_error = np.linalg.norm(
        rotation_matrix.T @ rotation_matrix - np.eye(3)
    )
    determinant = np.linalg.det(rotation_matrix)

    if identity_error > 1e-6 or abs(determinant - 1.0) > 1e-6:
        raise ValueError(
            f"{name} is not a proper rotation matrix: "
            f"orthogonality error={identity_error:.3e}, det={determinant:.9f}"
        )

    return rotation_matrix


def make_transform(rotation_target_from_source, translation_target_from_source):
    rotation_target_from_source = validate_rotation_matrix(
        rotation_target_from_source,
        "rotation_target_from_source",
    )
    translation_target_from_source = np.asarray(
        translation_target_from_source,
        dtype=float,
    ).reshape(-1)

    if translation_target_from_source.shape != (3,):
        raise ValueError("translation_target_from_source must contain 3 values")

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_target_from_source
    transform[:3, 3] = translation_target_from_source
    return transform


def invert_transform(transform_target_from_source):
    transform_target_from_source = np.asarray(
        transform_target_from_source,
        dtype=float,
    )

    if transform_target_from_source.shape != (4, 4):
        raise ValueError("transform must have shape 4 by 4")

    rotation_target_from_source = validate_rotation_matrix(
        transform_target_from_source[:3, :3]
    )
    translation_target_from_source = transform_target_from_source[:3, 3]

    rotation_source_from_target = rotation_target_from_source.T
    translation_source_from_target = (
        -rotation_source_from_target @ translation_target_from_source
    )

    return make_transform(
        rotation_source_from_target,
        translation_source_from_target,
    )


def transform_points(transform_target_from_source, points_source):
    points_source = validate_point_array(points_source, 3, "points_source")
    transform_target_from_source = np.asarray(
        transform_target_from_source,
        dtype=float,
    )

    if transform_target_from_source.shape != (4, 4):
        raise ValueError("transform_target_from_source must have shape 4 by 4")

    homogeneous_points = np.column_stack(
        [points_source, np.ones(points_source.shape[0])]
    )
    transformed_homogeneous = (
        transform_target_from_source @ homogeneous_points.T
    ).T

    return transformed_homogeneous[:, :3]


def match_positions_by_timestamp(
    reference_timestamps,
    reference_positions,
    measured_timestamps,
    measured_positions,
    maximum_time_difference,
):
    reference_timestamps = np.asarray(reference_timestamps, dtype=float).reshape(-1)
    measured_timestamps = np.asarray(measured_timestamps, dtype=float).reshape(-1)
    reference_positions = validate_point_array(
        reference_positions, 3, "reference_positions"
    )
    measured_positions = validate_point_array(
        measured_positions, 3, "measured_positions"
    )

    measured_order = np.argsort(measured_timestamps)
    sorted_measured_timestamps = measured_timestamps[measured_order]

    reference_indices, measured_indices, time_differences, unmatched = [], [], [], []

    for reference_index, timestamp in enumerate(reference_timestamps):
        insertion_index = np.searchsorted(sorted_measured_timestamps, timestamp)
        candidate_sorted_indices = []

        if insertion_index > 0:
            candidate_sorted_indices.append(insertion_index - 1)
        if insertion_index < sorted_measured_timestamps.size:
            candidate_sorted_indices.append(insertion_index)

        best_sorted_index = candidate_sorted_indices[0]
        best_time_difference = abs(
            sorted_measured_timestamps[best_sorted_index] - timestamp
        )

        for candidate_sorted_index in candidate_sorted_indices[1:]:
            candidate_time_difference = abs(
                sorted_measured_timestamps[candidate_sorted_index] - timestamp
            )
            if candidate_time_difference < best_time_difference:
                best_sorted_index = candidate_sorted_index
                best_time_difference = candidate_time_difference

        if best_time_difference <= maximum_time_difference:
            reference_indices.append(reference_index)
            measured_indices.append(measured_order[best_sorted_index])
            time_differences.append(best_time_difference)
        else:
            unmatched.append(reference_index)

    return {
        "reference_indices": np.asarray(reference_indices, dtype=int),
        "measured_indices": np.asarray(measured_indices, dtype=int),
        "reference_positions": reference_positions[reference_indices],
        "measured_positions": measured_positions[measured_indices],
        "time_differences": np.asarray(time_differences, dtype=float),
        "unmatched_reference_indices": np.asarray(unmatched, dtype=int),
    }


def estimate_antenna_baseline(main_antenna_positions_gps, auxiliary_positions_gps):
    main_antenna_positions_gps = validate_point_array(
        main_antenna_positions_gps, 3, "main_antenna_positions_gps"
    )
    auxiliary_positions_gps = validate_point_array(
        auxiliary_positions_gps, 3, "auxiliary_positions_gps"
    )

    baseline_samples_gps = auxiliary_positions_gps - main_antenna_positions_gps
    mean_baseline_gps = np.mean(baseline_samples_gps, axis=0)

    if baseline_samples_gps.shape[0] > 1:
        baseline_standard_deviation = np.std(
            baseline_samples_gps, axis=0, ddof=1
        )
    else:
        baseline_standard_deviation = np.full(3, np.nan)

    return {
        "mean_baseline_gps": mean_baseline_gps,
        "baseline_samples_gps": baseline_samples_gps,
        "baseline_standard_deviation": baseline_standard_deviation,
    }


def gps_to_imu_transform_from_baseline(
    baseline_auxiliary_from_main_gps,
    rotation_imu_from_gps,
    imu_origin_to_auxiliary_antenna_imu,
):
    baseline_auxiliary_from_main_gps = np.asarray(
        baseline_auxiliary_from_main_gps, dtype=float
    ).reshape(-1)
    imu_origin_to_auxiliary_antenna_imu = np.asarray(
        imu_origin_to_auxiliary_antenna_imu, dtype=float
    ).reshape(-1)

    rotation_imu_from_gps = validate_rotation_matrix(
        rotation_imu_from_gps, "rotation_imu_from_gps"
    )
    rotation_gps_from_imu = rotation_imu_from_gps.T

    imu_origin_position_gps = (
        baseline_auxiliary_from_main_gps
        - rotation_gps_from_imu @ imu_origin_to_auxiliary_antenna_imu
    )
    translation_imu_from_gps = -rotation_imu_from_gps @ imu_origin_position_gps

    return make_transform(rotation_imu_from_gps, translation_imu_from_gps)


def moving_block_bootstrap_indices(sample_count, block_length, rng):
    indices = []
    maximum_start = sample_count - block_length

    while len(indices) < sample_count:
        block_start = int(rng.integers(0, maximum_start + 1))
        indices.extend(range(block_start, block_start + block_length))

    return np.asarray(indices[:sample_count], dtype=int)


def bootstrap_gps_to_imu_transforms(
    main_antenna_positions_gps,
    auxiliary_positions_gps,
    rotation_imu_from_gps,
    imu_origin_to_auxiliary_antenna_imu,
    bootstrap_iterations,
    block_length,
    rng,
):
    sample_count = main_antenna_positions_gps.shape[0]
    transforms_imu_from_gps = np.empty((bootstrap_iterations, 4, 4), dtype=float)
    baseline_samples = np.empty((bootstrap_iterations, 3), dtype=float)

    for bootstrap_index in range(bootstrap_iterations):
        sample_indices = moving_block_bootstrap_indices(
            sample_count, block_length, rng
        )
        sampled_baseline = (
            auxiliary_positions_gps[sample_indices]
            - main_antenna_positions_gps[sample_indices]
        )
        mean_baseline = np.mean(sampled_baseline, axis=0)
        baseline_samples[bootstrap_index] = mean_baseline

        transforms_imu_from_gps[bootstrap_index] = (
            gps_to_imu_transform_from_baseline(
                mean_baseline,
                rotation_imu_from_gps,
                imu_origin_to_auxiliary_antenna_imu,
            )
        )

    return {
        "transforms_imu_from_gps": transforms_imu_from_gps,
        "baseline_samples_gps": baseline_samples,
    }


def normalize_points_2d(image_points):
    image_points = validate_point_array(image_points, 2, "image_points")
    centroid = np.mean(image_points, axis=0)
    centered_points = image_points - centroid
    root_mean_square_distance = math.sqrt(
        np.mean(np.sum(centered_points ** 2, axis=1))
    )

    scale = math.sqrt(2.0) / root_mean_square_distance
    normalization = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    homogeneous_points = np.column_stack(
        [image_points, np.ones(image_points.shape[0])]
    )
    normalized_homogeneous = (normalization @ homogeneous_points.T).T
    return normalized_homogeneous[:, :2], normalization


def normalize_points_3d(world_points):
    world_points = validate_point_array(world_points, 3, "world_points")
    centroid = np.mean(world_points, axis=0)
    centered_points = world_points - centroid
    root_mean_square_distance = math.sqrt(
        np.mean(np.sum(centered_points ** 2, axis=1))
    )

    scale = math.sqrt(3.0) / root_mean_square_distance
    normalization = np.array(
        [
            [scale, 0.0, 0.0, -scale * centroid[0]],
            [0.0, scale, 0.0, -scale * centroid[1]],
            [0.0, 0.0, scale, -scale * centroid[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    homogeneous_points = np.column_stack(
        [world_points, np.ones(world_points.shape[0])]
    )
    normalized_homogeneous = (normalization @ homogeneous_points.T).T
    return normalized_homogeneous[:, :3], normalization


def undistort_image_points(
    image_points,
    camera_matrix,
    distortion_coefficients,
    iteration_count=10,
):
    """Remove Brown-Conrady distortion and return undistorted pixel points."""
    image_points = validate_point_array(image_points, 2, "image_points")
    camera_matrix = np.asarray(camera_matrix, dtype=float)
    distortion_coefficients = np.asarray(
        distortion_coefficients, dtype=float
    ).reshape(-1)

    if distortion_coefficients.size == 0:
        return image_points.copy()

    k1 = distortion_coefficients[0]
    k2 = distortion_coefficients[1] if distortion_coefficients.size >= 4 else 0.0
    p1 = distortion_coefficients[2] if distortion_coefficients.size >= 4 else 0.0
    p2 = distortion_coefficients[3] if distortion_coefficients.size >= 4 else 0.0
    k3 = distortion_coefficients[4] if distortion_coefficients.size == 5 else 0.0

    homogeneous_pixels = np.column_stack(
        [image_points, np.ones(image_points.shape[0])]
    )
    distorted_normalized_homogeneous = (
        np.linalg.inv(camera_matrix) @ homogeneous_pixels.T
    ).T
    distorted_normalized = (
        distorted_normalized_homogeneous[:, :2]
        / distorted_normalized_homogeneous[:, 2:3]
    )

    undistorted_normalized = distorted_normalized.copy()

    for _ in range(iteration_count):
        x = undistorted_normalized[:, 0]
        y = undistorted_normalized[:, 1]
        radius_squared = x ** 2 + y ** 2
        radial_scale = (
            1.0
            + k1 * radius_squared
            + k2 * radius_squared ** 2
            + k3 * radius_squared ** 3
        )

        tangential_x = 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x ** 2)
        tangential_y = p1 * (radius_squared + 2.0 * y ** 2) + 2.0 * p2 * x * y

        undistorted_normalized[:, 0] = (
            distorted_normalized[:, 0] - tangential_x
        ) / radial_scale
        undistorted_normalized[:, 1] = (
            distorted_normalized[:, 1] - tangential_y
        ) / radial_scale

    undistorted_homogeneous = np.column_stack(
        [undistorted_normalized, np.ones(undistorted_normalized.shape[0])]
    )
    undistorted_pixels_homogeneous = (camera_matrix @ undistorted_homogeneous.T).T

    return (
        undistorted_pixels_homogeneous[:, :2]
        / undistorted_pixels_homogeneous[:, 2:3]
    )


def estimate_projection_matrix_dlt(world_points_gps, image_points):
    """Estimate a 3 by 4 projective camera matrix with normalized DLT."""
    world_points_gps = validate_point_array(
        world_points_gps, 3, "world_points_gps", minimum_count=6
    )
    image_points = validate_point_array(
        image_points, 2, "image_points", minimum_count=6
    )

    normalized_world_points, world_normalization = normalize_points_3d(world_points_gps)
    normalized_image_points, image_normalization = normalize_points_2d(image_points)

    point_count = world_points_gps.shape[0]
    design_matrix = np.zeros((2 * point_count, 12), dtype=float)

    for point_index in range(point_count):
        x, y, z = normalized_world_points[point_index]
        u, v = normalized_image_points[point_index]
        homogeneous_world = np.array([x, y, z, 1.0], dtype=float)

        design_matrix[2 * point_index, 4:8] = -homogeneous_world
        design_matrix[2 * point_index, 8:12] = v * homogeneous_world
        design_matrix[2 * point_index + 1, 0:4] = homogeneous_world
        design_matrix[2 * point_index + 1, 8:12] = -u * homogeneous_world

    _, _, right_singular_vectors = np.linalg.svd(design_matrix, full_matrices=True)
    normalized_projection = right_singular_vectors[-1].reshape(3, 4)

    projection_matrix = (
        np.linalg.inv(image_normalization)
        @ normalized_projection
        @ world_normalization
    )

    projection_norm = np.linalg.norm(projection_matrix)
    return projection_matrix / projection_norm


def decompose_projection_matrix(projection_matrix, camera_matrix):
    """Recover H_camera_from_gps and the camera center using known intrinsics."""
    extrinsic_scaled = np.linalg.inv(camera_matrix) @ projection_matrix
    rotation_scaled = extrinsic_scaled[:, :3]

    if np.linalg.det(rotation_scaled) < 0.0:
        extrinsic_scaled = -extrinsic_scaled
        rotation_scaled = extrinsic_scaled[:, :3]

    _, singular_values, _ = np.linalg.svd(rotation_scaled)
    scale = np.mean(singular_values)

    rotation_approximate = rotation_scaled / scale
    left_singular_vectors, _, right_singular_vectors = np.linalg.svd(
        rotation_approximate
    )
    rotation_camera_from_gps = left_singular_vectors @ right_singular_vectors

    if np.linalg.det(rotation_camera_from_gps) < 0.0:
        left_singular_vectors[:, -1] = -left_singular_vectors[:, -1]
        rotation_camera_from_gps = left_singular_vectors @ right_singular_vectors

    translation_camera_from_gps = extrinsic_scaled[:, 3] / scale
    transform_camera_from_gps = make_transform(
        rotation_camera_from_gps, translation_camera_from_gps
    )
    camera_position_gps = -rotation_camera_from_gps.T @ translation_camera_from_gps

    return transform_camera_from_gps, camera_position_gps


def project_world_points(world_points_gps, transform_camera_from_gps, camera_matrix):
    world_points_gps = validate_point_array(
        world_points_gps, 3, "world_points_gps"
    )
    points_camera = transform_points(transform_camera_from_gps, world_points_gps)
    depth = points_camera[:, 2]

    image_homogeneous = (camera_matrix @ points_camera.T).T
    image_points = image_homogeneous[:, :2] / image_homogeneous[:, 2:3]
    return image_points, depth


def estimate_camera_pose_dlt(
    world_points_gps,
    image_points,
    camera_matrix,
    distortion_coefficients=None,
):
    if distortion_coefficients is None:
        distortion_coefficients = np.empty(0, dtype=float)

    undistorted_image_points = undistort_image_points(
        image_points, camera_matrix, distortion_coefficients
    )
    projection_matrix = estimate_projection_matrix_dlt(
        world_points_gps, undistorted_image_points
    )
    transform_camera_from_gps, camera_position_gps = decompose_projection_matrix(
        projection_matrix, camera_matrix
    )
    reprojected_points, depth = project_world_points(
        world_points_gps, transform_camera_from_gps, camera_matrix
    )
    reprojection_residuals = reprojected_points - undistorted_image_points
    reprojection_rmse_pixels = math.sqrt(
        np.mean(np.sum(reprojection_residuals ** 2, axis=1))
    )

    return {
        "projection_matrix": projection_matrix,
        "transform_camera_from_gps": transform_camera_from_gps,
        "camera_position_gps": camera_position_gps,
        "undistorted_image_points": undistorted_image_points,
        "reprojected_image_points": reprojected_points,
        "reprojection_residuals": reprojection_residuals,
        "reprojection_rmse_pixels": reprojection_rmse_pixels,
        "depth_camera": depth,
        "positive_depth_count": int(np.count_nonzero(depth > 0.0)),
    }


def bootstrap_camera_poses_dlt(
    world_points_gps,
    image_points,
    camera_matrix,
    distortion_coefficients,
    bootstrap_iterations,
    rng,
    bootstrap_indices=None,
):
    point_count = world_points_gps.shape[0]

    if bootstrap_indices is None:
        bootstrap_indices = rng.integers(
            0, point_count, size=(bootstrap_iterations, point_count)
        )

    transforms_camera_from_gps = np.full(
        (bootstrap_iterations, 4, 4), np.nan, dtype=float
    )
    camera_positions_gps = np.full(
        (bootstrap_iterations, 3), np.nan, dtype=float
    )
    reprojection_rmse_pixels = np.full(
        bootstrap_iterations, np.nan, dtype=float
    )
    failure_messages = [None] * bootstrap_iterations

    for bootstrap_index in range(bootstrap_iterations):
        sample_indices = bootstrap_indices[bootstrap_index]

        try:
            pose_result = estimate_camera_pose_dlt(
                world_points_gps[sample_indices],
                image_points[sample_indices],
                camera_matrix,
                distortion_coefficients,
            )
            transforms_camera_from_gps[bootstrap_index] = pose_result[
                "transform_camera_from_gps"
            ]
            camera_positions_gps[bootstrap_index] = pose_result[
                "camera_position_gps"
            ]

            full_reprojected_points, _ = project_world_points(
                world_points_gps,
                pose_result["transform_camera_from_gps"],
                camera_matrix,
            )
            full_undistorted_points = undistort_image_points(
                image_points, camera_matrix, distortion_coefficients
            )
            residuals = full_reprojected_points - full_undistorted_points
            reprojection_rmse_pixels[bootstrap_index] = math.sqrt(
                np.mean(np.sum(residuals ** 2, axis=1))
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            failure_messages[bootstrap_index] = str(error)

    valid_mask = np.all(
        np.isfinite(transforms_camera_from_gps.reshape(bootstrap_iterations, -1)),
        axis=1,
    )

    return {
        "transforms_camera_from_gps": transforms_camera_from_gps,
        "camera_positions_gps": camera_positions_gps,
        "reprojection_rmse_pixels": reprojection_rmse_pixels,
        "bootstrap_indices": bootstrap_indices,
        "valid_mask": valid_mask,
        "valid_count": int(np.count_nonzero(valid_mask)),
        "failure_count": int(np.count_nonzero(~valid_mask)),
        "failure_messages": failure_messages,
    }


def calculate_imu_to_camera_transform(
    transform_imu_from_gps,
    transform_camera_from_gps,
):
    """Compose H_camera_from_imu = H_camera_from_gps @ H_gps_from_imu."""
    transform_gps_from_imu = invert_transform(transform_imu_from_gps)
    return transform_camera_from_gps @ transform_gps_from_imu


def rotation_difference_degrees(rotation_a, rotation_b):
    rotation_a = validate_rotation_matrix(rotation_a, "rotation_a")
    rotation_b = validate_rotation_matrix(rotation_b, "rotation_b")
    relative_rotation = rotation_a @ rotation_b.T
    cosine_angle = (np.trace(relative_rotation) - 1.0) / 2.0
    cosine_angle = float(np.clip(cosine_angle, -1.0, 1.0))
    return math.degrees(math.acos(cosine_angle))


def percentile_interval(values, confidence_level):
    values = np.asarray(values, dtype=float)

    lower_percentile = 50.0 * (1.0 - confidence_level)
    upper_percentile = 100.0 - lower_percentile
    return np.percentile(
        values, [lower_percentile, 50.0, upper_percentile], axis=0
    )


def summarize_bootstrap_transforms(
    bootstrap_transforms,
    reference_transform,
    confidence_level,
):
    valid_mask = np.all(
        np.isfinite(
            bootstrap_transforms.reshape(bootstrap_transforms.shape[0], -1)
        ),
        axis=1,
    )
    valid_transforms = bootstrap_transforms[valid_mask]

    if valid_transforms.shape[0] == 0:
        raise ValueError("no valid bootstrap transforms are available")

    translations = valid_transforms[:, :3, 3]
    rotation_errors_degrees = np.empty(valid_transforms.shape[0], dtype=float)

    for sample_index, transform in enumerate(valid_transforms):
        rotation_errors_degrees[sample_index] = rotation_difference_degrees(
            transform[:3, :3], reference_transform[:3, :3]
        )

    return {
        "valid_count": int(valid_transforms.shape[0]),
        "failure_count": int(np.count_nonzero(~valid_mask)),
        "translation_interval": percentile_interval(
            translations, confidence_level
        ),
        "rotation_error_degrees_interval": percentile_interval(
            rotation_errors_degrees, confidence_level
        ),
    }


def find_csv_column(fieldnames, candidate_names, description, required=True):
    normalized_fields = {field.strip().lower(): field for field in fieldnames}

    for candidate in candidate_names:
        norm_candidate = candidate.strip().lower()
        if norm_candidate in normalized_fields:
            return normalized_fields[norm_candidate]

    if required:
        expected = ", ".join(candidate_names)
        raise ValueError(
            f"CSV is missing {description}. Accepted column names: {expected}"
        )

    return None

def parse_csv_float(row, column_name, row_number, input_path):
    text = row.get(column_name, "")
    try:
        if isinstance(text, str):
            text = text.replace(",", "").strip()
        value = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{input_path}, row {row_number}: {column_name} is not numeric"
        ) from error

    if not math.isfinite(value):
        raise ValueError(
            f"{input_path}, row {row_number}: {column_name} is not finite"
        )

    return value

def read_geodetic_csv(input_path):
    """Read GPS coordinates from CSV."""
    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"GPS input file does not exist: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")

        latitude_column = find_csv_column(
            reader.fieldnames,
            ["latitude_deg", "latitude", "lat", "gps_latitude_deg"],
            "latitude",
        )
        longitude_column = find_csv_column(
            reader.fieldnames,
            ["longitude_deg", "longitude", "lon", "gps_longitude_deg"],
            "longitude",
        )
        height_column = find_csv_column(
            reader.fieldnames,
            ["ellipsoid_height_m", "altitude_m", "height_m", "altitude", "height"],
            "ellipsoid height",
        )
        timestamp_column = find_csv_column(
            reader.fieldnames,
            ["timestamp_s", "timestamp", "time_s", "unix_time_s", "time"],
            "timestamp",
            required=False,
        )

        geodetic_rows, timestamps = [], []

        for row_number, row in enumerate(reader, start=2):
            lat = parse_csv_float(row, latitude_column, row_number, input_path)
            lon = parse_csv_float(row, longitude_column, row_number, input_path)
            alt = parse_csv_float(row, height_column, row_number, input_path)
            geodetic_rows.append([lat, lon, alt])

            if timestamp_column is not None:
                timestamps.append(
                    parse_csv_float(row, timestamp_column, row_number, input_path)
                )

    return {
        "geodetic": np.asarray(geodetic_rows, dtype=float),
        "timestamps": np.asarray(timestamps, dtype=float) if timestamp_column else None,
        "columns": {
            "latitude": latitude_column,
            "longitude": longitude_column,
            "height": height_column,
            "timestamp": timestamp_column,
        },
    }


def rq_decomposition_3x3(matrix):
    matrix = np.asarray(matrix, dtype=float)
    reversed_transpose = np.flipud(matrix).T
    orthogonal_reversed, upper_reversed = np.linalg.qr(reversed_transpose)
    upper_triangular = np.flipud(upper_reversed.T)
    upper_triangular = np.fliplr(upper_triangular)
    orthogonal = orthogonal_reversed.T
    orthogonal = np.flipud(orthogonal)
    return upper_triangular, orthogonal


def decompose_projection_matrix_unknown_intrinsics(
    projection_matrix,
    world_points_gps=None,
):
    """Decompose a DLT matrix into intrinsic K and extrinsic H_camera_from_gps."""
    projection_matrix = np.asarray(projection_matrix, dtype=float)
    camera_matrix, rotation_camera_from_gps = rq_decomposition_3x3(
        projection_matrix[:, :3]
    )

    diagonal_signs = np.sign(np.diag(camera_matrix))
    diagonal_signs[diagonal_signs == 0.0] = 1.0
    sign_matrix = np.diag(diagonal_signs)
    camera_matrix = camera_matrix @ sign_matrix
    rotation_camera_from_gps = sign_matrix @ rotation_camera_from_gps

    if np.linalg.det(rotation_camera_from_gps) < 0.0:
        camera_matrix = -camera_matrix
        rotation_camera_from_gps = -rotation_camera_from_gps

    intrinsic_scale = camera_matrix[2, 2]
    camera_matrix = camera_matrix / intrinsic_scale
    scaled_projection = projection_matrix / intrinsic_scale
    rotation_camera_from_gps = validate_rotation_matrix(
        rotation_camera_from_gps, "rotation_camera_from_gps"
    )
    translation_camera_from_gps = np.linalg.solve(
        camera_matrix, scaled_projection[:, 3]
    )
    transform_camera_from_gps = make_transform(
        rotation_camera_from_gps, translation_camera_from_gps
    )

    if world_points_gps is not None:
        world_points_gps = validate_point_array(
            world_points_gps, 3, "world_points_gps"
        )
        points_camera = transform_points(
            transform_camera_from_gps, world_points_gps
        )

        if np.count_nonzero(points_camera[:, 2] > 0.0) < (
            world_points_gps.shape[0] / 2.0
        ):
            flip_matrix = np.array([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0]
            ])
            rotation_camera_from_gps = flip_matrix @ rotation_camera_from_gps
            translation_camera_from_gps = flip_matrix @ translation_camera_from_gps
            transform_camera_from_gps[:3, :3] = rotation_camera_from_gps
            transform_camera_from_gps[:3, 3] = translation_camera_from_gps

    camera_position_gps = -rotation_camera_from_gps.T @ translation_camera_from_gps
    return camera_matrix, transform_camera_from_gps, camera_position_gps


def rotation_matrix_to_vector(rotation_matrix):
    rotation_matrix = validate_rotation_matrix(rotation_matrix)
    cosine_angle = (np.trace(rotation_matrix) - 1.0) / 2.0
    cosine_angle = float(np.clip(cosine_angle, -1.0, 1.0))
    angle = math.acos(cosine_angle)

    if angle < 1e-10:
        return np.array(
            [
                rotation_matrix[2, 1] - rotation_matrix[1, 2],
                rotation_matrix[0, 2] - rotation_matrix[2, 0],
                rotation_matrix[1, 0] - rotation_matrix[0, 1],
            ],
            dtype=float,
        ) / 2.0

    if abs(math.pi - angle) < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eig(rotation_matrix)
        axis_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, axis_index])
        axis = axis / np.linalg.norm(axis)
        return axis * angle

    axis = np.array(
        [
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1],
        ],
        dtype=float,
    )
    axis = axis / (2.0 * math.sin(angle))
    return axis * angle


def rotation_vector_to_matrix(rotation_vector):
    rotation_vector = np.asarray(rotation_vector, dtype=float).reshape(-1)
    angle = np.linalg.norm(rotation_vector)

    if angle < 1e-12:
        skew_matrix = np.array(
            [
                [0.0, -rotation_vector[2], rotation_vector[1]],
                [rotation_vector[2], 0.0, -rotation_vector[0]],
                [-rotation_vector[1], rotation_vector[0], 0.0],
            ],
            dtype=float,
        )
        return np.eye(3) + skew_matrix

    axis = rotation_vector / angle
    axis_x, axis_y, axis_z = axis
    skew_matrix = np.array(
        [
            [0.0, -axis_z, axis_y],
            [axis_z, 0.0, -axis_x],
            [-axis_y, axis_x, 0.0],
        ],
        dtype=float,
    )
    return (
        np.eye(3)
        + math.sin(angle) * skew_matrix
        + (1.0 - math.cos(angle)) * (skew_matrix @ skew_matrix)
    )


def calibration_parameters_from_matrices(
    camera_matrix,
    transform_camera_from_gps,
    distortion_coefficients=None,
):
    if distortion_coefficients is None:
        distortion_coefficients = np.zeros(5, dtype=float)

    rotation_vector = rotation_matrix_to_vector(transform_camera_from_gps[:3, :3])
    translation = transform_camera_from_gps[:3, 3]

    return np.concatenate(
        [
            np.array([
                camera_matrix[0, 0], camera_matrix[1, 1],
                camera_matrix[0, 1], camera_matrix[0, 2], camera_matrix[1, 2]
            ], dtype=float),
            rotation_vector,
            translation,
            distortion_coefficients,
        ]
    )


def calibration_matrices_from_parameters(parameters):
    focal_x, focal_y, skew, center_x, center_y = parameters[:5]

    camera_matrix = np.array(
        [
            [focal_x, skew, center_x],
            [0.0, focal_y, center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rotation_camera_from_gps = rotation_vector_to_matrix(parameters[5:8])
    transform_camera_from_gps = make_transform(
        rotation_camera_from_gps, parameters[8:11]
    )
    distortion_coefficients = parameters[11:16].copy()
    return camera_matrix, transform_camera_from_gps, distortion_coefficients


def project_points_with_distortion(
    world_points_gps,
    camera_matrix,
    transform_camera_from_gps,
    distortion_coefficients,
):
    points_camera = transform_points(transform_camera_from_gps, world_points_gps)
    depth = points_camera[:, 2]

    x = points_camera[:, 0] / depth
    y = points_camera[:, 1] / depth
    k1, k2, p1, p2, k3 = distortion_coefficients
    radius_squared = x ** 2 + y ** 2
    radial_scale = (
        1.0 + k1 * radius_squared + k2 * radius_squared ** 2 + k3 * radius_squared ** 3
    )
    x_distorted = (
        x * radial_scale + 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x ** 2)
    )
    y_distorted = (
        y * radial_scale + p1 * (radius_squared + 2.0 * y ** 2) + 2.0 * p2 * x * y
    )
    u = (
        camera_matrix[0, 0] * x_distorted
        + camera_matrix[0, 1] * y_distorted
        + camera_matrix[0, 2]
    )
    v = camera_matrix[1, 1] * y_distorted + camera_matrix[1, 2]
    return np.column_stack([u, v]), depth


def calibration_reprojection_residuals(
    parameters,
    world_points_gps,
    observed_image_points,
):
    camera_matrix, transform_camera_from_gps, distortion_coefficients = (
        calibration_matrices_from_parameters(parameters)
    )
    projected_image_points, depth = project_points_with_distortion(
        world_points_gps,
        camera_matrix,
        transform_camera_from_gps,
        distortion_coefficients,
    )

    if np.any(depth <= 0.0):
        depth = np.maximum(depth, 1e-3)

    return (projected_image_points - observed_image_points).reshape(-1)


def calculate_calibration_jacobian(
    parameters,
    residuals,
    world_points_gps,
    image_points,
    finite_difference_step,
):
    jacobian = np.empty((residuals.size, parameters.size), dtype=float)

    for parameter_index in range(parameters.size):
        parameter_scale = max(abs(parameters[parameter_index]), 1.0)
        step = finite_difference_step * parameter_scale
        positive_parameters = parameters.copy()
        negative_parameters = parameters.copy()
        positive_parameters[parameter_index] += step
        negative_parameters[parameter_index] -= step

        try:
            positive_residuals = calibration_reprojection_residuals(
                positive_parameters, world_points_gps, image_points
            )
        except ValueError:
            positive_residuals = None

        try:
            negative_residuals = calibration_reprojection_residuals(
                negative_parameters, world_points_gps, image_points
            )
        except ValueError:
            negative_residuals = None

        if positive_residuals is not None and negative_residuals is not None:
            jacobian[:, parameter_index] = (
                positive_residuals - negative_residuals
            ) / (2.0 * step)
        elif positive_residuals is not None:
            jacobian[:, parameter_index] = (positive_residuals - residuals) / step
        elif negative_residuals is not None:
            jacobian[:, parameter_index] = (residuals - negative_residuals) / step
        else:
            raise ValueError(
                f"Could not evaluate calibration residuals near parameter index {parameter_index}"
            )

    return jacobian


def refine_camera_calibration(
    world_points_gps,
    image_points,
    initial_camera_matrix,
    initial_transform_camera_from_gps,
    maximum_iterations=100,
    initial_damping=1e-3,
    finite_difference_step=1e-6,
    convergence_tolerance=1e-9,
):
    """Jointly refine intrinsics, pose, and five distortion coefficients."""
    world_points_gps = validate_point_array(
        world_points_gps, 3, "world_points_gps", minimum_count=8
    )
    image_points = validate_point_array(
        image_points, 2, "image_points", minimum_count=8
    )

    parameters = calibration_parameters_from_matrices(
        initial_camera_matrix,
        initial_transform_camera_from_gps,
        np.zeros(5, dtype=float),
    )
    damping = float(initial_damping)
    converged = False
    accepted_iterations = 0

    residuals = calibration_reprojection_residuals(
        parameters, world_points_gps, image_points
    )
    cost = 0.5 * float(residuals @ residuals)

    for iteration_index in range(maximum_iterations):
        jacobian = calculate_calibration_jacobian(
            parameters,
            residuals,
            world_points_gps,
            image_points,
            finite_difference_step,
        )

        approximate_hessian = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals
        hessian_diagonal = np.diag(approximate_hessian).copy()
        hessian_diagonal[hessian_diagonal <= np.finfo(float).eps] = 1.0
        damped_hessian = approximate_hessian + damping * np.diag(hessian_diagonal)

        try:
            parameter_update = np.linalg.solve(damped_hessian, -gradient)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue

        candidate_parameters = parameters + parameter_update

        try:
            candidate_residuals = calibration_reprojection_residuals(
                candidate_parameters, world_points_gps, image_points
            )
        except ValueError:
            damping *= 10.0
            continue

        candidate_cost = 0.5 * float(candidate_residuals @ candidate_residuals)

        if candidate_cost < cost:
            relative_update = np.linalg.norm(
                parameter_update / np.maximum(np.abs(parameters), 1.0)
            )
            relative_cost_change = abs(cost - candidate_cost) / max(cost, 1.0)
            parameters = candidate_parameters
            residuals = candidate_residuals
            cost = candidate_cost
            damping = max(damping / 3.0, 1e-15)
            accepted_iterations += 1

            if (
                relative_update <= convergence_tolerance
                or relative_cost_change <= convergence_tolerance
            ):
                converged = True
                break
        else:
            damping *= 10.0

    final_camera_matrix, final_transform_camera_from_gps, final_distortion = (
        calibration_matrices_from_parameters(parameters)
    )
    final_projected_points, final_depth = project_points_with_distortion(
        world_points_gps,
        final_camera_matrix,
        final_transform_camera_from_gps,
        final_distortion,
    )
    final_residuals_2d = final_projected_points - image_points
    final_rmse_pixels = math.sqrt(
        np.mean(np.sum(final_residuals_2d ** 2, axis=1))
    )

    final_jacobian = calculate_calibration_jacobian(
        parameters, residuals, world_points_gps, image_points, finite_difference_step
    )

    normal_matrix_condition = np.linalg.cond(final_jacobian.T @ final_jacobian)
    camera_position_gps = invert_transform(final_transform_camera_from_gps)[:3, 3]

    return {
        "camera_matrix": final_camera_matrix,
        "distortion_coefficients": final_distortion,
        "transform_camera_from_gps": final_transform_camera_from_gps,
        "camera_position_gps": camera_position_gps,
        "projected_image_points": final_projected_points,
        "reprojection_residuals": final_residuals_2d,
        "reprojection_rmse_pixels": final_rmse_pixels,
        "positive_depth_count": int(np.count_nonzero(final_depth > 0.0)),
        "converged": converged,
        "accepted_iterations": accepted_iterations,
        "attempted_iterations": iteration_index + 1,
        "normal_matrix_condition": normal_matrix_condition,
    }


def estimate_camera_intrinsics_distortion_and_pose(
    world_points_gps,
    image_points,
    maximum_iterations=100,
    initial_damping=1e-3,
    finite_difference_step=1e-6,
    convergence_tolerance=1e-9,
):
    """Use DLT for initialization, then refine K, distortion, R, and t."""
    world_points_gps = validate_point_array(
        world_points_gps, 3, "world_points_gps", minimum_count=8
    )
    image_points = validate_point_array(
        image_points, 2, "image_points", minimum_count=8
    )

    projection_matrix = estimate_projection_matrix_dlt(
        world_points_gps, image_points
    )
    initial_camera_matrix, initial_transform, initial_camera_position = (
        decompose_projection_matrix_unknown_intrinsics(
            projection_matrix, world_points_gps
        )
    )
    refined_result = refine_camera_calibration(
        world_points_gps,
        image_points,
        initial_camera_matrix,
        initial_transform,
        maximum_iterations=maximum_iterations,
        initial_damping=initial_damping,
        finite_difference_step=finite_difference_step,
        convergence_tolerance=convergence_tolerance,
    )
    refined_result["projection_matrix_dlt"] = projection_matrix
    refined_result["initial_camera_matrix_dlt"] = initial_camera_matrix
    refined_result["initial_camera_position_gps_dlt"] = initial_camera_position
    return refined_result


def estimate_static_antenna_baseline(
    main_antenna_positions_gps, auxiliary_antenna_positions_gps
):
    mean_main = np.mean(main_antenna_positions_gps, axis=0)
    mean_aux = np.mean(auxiliary_antenna_positions_gps, axis=0)
    mean_baseline = mean_aux - mean_main

    return {
        "mean_main_position_gps": mean_main,
        "mean_auxiliary_position_gps": mean_aux,
        "mean_baseline_gps": mean_baseline,
        "main_standard_deviation": np.std(main_antenna_positions_gps, axis=0, ddof=1)
        if main_antenna_positions_gps.shape[0] > 1 else np.full(3, np.nan),
        "auxiliary_standard_deviation": np.std(auxiliary_antenna_positions_gps, axis=0, ddof=1)
        if auxiliary_antenna_positions_gps.shape[0] > 1 else np.full(3, np.nan),
    }


def bootstrap_static_gps_to_imu_transforms(
    main_antenna_positions_gps,
    auxiliary_antenna_positions_gps,
    rotation_imu_from_gps_value,
    imu_origin_to_auxiliary_antenna_imu_value,
    iteration_count,
    block_length,
    rng,
):
    baseline_samples_gps = np.empty((iteration_count, 3), dtype=float)
    transforms_imu_from_gps = np.empty((iteration_count, 4, 4), dtype=float)

    for bootstrap_index in range(iteration_count):
        main_indices = moving_block_bootstrap_indices(
            main_antenna_positions_gps.shape[0], block_length, rng
        )
        auxiliary_indices = moving_block_bootstrap_indices(
            auxiliary_antenna_positions_gps.shape[0], block_length, rng
        )
        mean_main = np.mean(main_antenna_positions_gps[main_indices], axis=0)
        mean_auxiliary = np.mean(
            auxiliary_antenna_positions_gps[auxiliary_indices], axis=0
        )
        baseline = mean_auxiliary - mean_main
        baseline_samples_gps[bootstrap_index] = baseline
        transforms_imu_from_gps[bootstrap_index] = (
            gps_to_imu_transform_from_baseline(
                baseline,
                rotation_imu_from_gps_value,
                imu_origin_to_auxiliary_antenna_imu_value,
            )
        )

    return {
        "baseline_samples_gps": baseline_samples_gps,
        "transforms_imu_from_gps": transforms_imu_from_gps,
    }


def calculate_from_input_files():
    """Run the file-based three-camera calibration workflow with manual 2D points."""
    configured_rotation_imu_from_gps = validate_rotation_matrix(
        rotation_imu_from_gps, "rotation_imu_from_gps"
    )
    configured_imu_lever_arm = np.asarray(
        imu_origin_to_auxiliary_antenna_imu, dtype=float
    ).reshape(-1)

    main_gps_data = read_geodetic_csv(gps_points)
    imu_antenna_data = read_geodetic_csv(imu_points)
    reference_geodetic = mean_geodetic_position(main_gps_data["geodetic"])

    main_positions_gps = geodetic_to_enu(
        main_gps_data["geodetic"], reference_geodetic
    )
    auxiliary_positions_gps = geodetic_to_enu(
        imu_antenna_data["geodetic"], reference_geodetic
    )
    rng = np.random.default_rng(random_seed)

    if gps_maximum_time_difference_seconds is None:
        baseline_method = "independent static means"
        baseline_result = estimate_static_antenna_baseline(
            main_positions_gps, auxiliary_positions_gps
        )
        mean_baseline_gps = baseline_result["mean_baseline_gps"]
        gps_bootstrap_result = bootstrap_static_gps_to_imu_transforms(
            main_positions_gps,
            auxiliary_positions_gps,
            configured_rotation_imu_from_gps,
            configured_imu_lever_arm,
            bootstrap_iterations,
            gps_bootstrap_block_length,
            rng,
        )
        gps_matching_result = None
    else:
        baseline_method = "nearest timestamp paired samples"
        gps_matching_result = match_positions_by_timestamp(
            main_gps_data["timestamps"],
            main_positions_gps,
            imu_antenna_data["timestamps"],
            auxiliary_positions_gps,
            gps_maximum_time_difference_seconds,
        )
        baseline_result = estimate_antenna_baseline(
            gps_matching_result["reference_positions"],
            gps_matching_result["measured_positions"],
        )
        mean_baseline_gps = baseline_result["mean_baseline_gps"]
        gps_bootstrap_result = bootstrap_gps_to_imu_transforms(
            gps_matching_result["reference_positions"],
            gps_matching_result["measured_positions"],
            configured_rotation_imu_from_gps,
            configured_imu_lever_arm,
            bootstrap_iterations,
            gps_bootstrap_block_length,
            rng,
        )

    transform_imu_from_gps_value = gps_to_imu_transform_from_baseline(
        mean_baseline_gps,
        configured_rotation_imu_from_gps,
        configured_imu_lever_arm,
    )

    camera_names = ["camera_1", "camera_2", "camera_3"]
    
    # Read 3D geodetic target points from CSV
    target_geodetic_data = read_geodetic_csv(camera_antenna_point)
    target_points_gps = geodetic_to_enu(
        target_geodetic_data["geodetic"], reference_geodetic
    )

    # Read 2D image coordinates from JSON
    camera_image_points_dict = read_manual_camera_target_json(
        manual_json_path, camera_names
    )

    if target_points_gps.shape[0] < 8:
        raise ValueError(
            "At least 8 camera target observations are required to estimate "
            "intrinsics, pose, and distortion"
        )

    shared_camera_bootstrap_indices = rng.integers(
        0,
        target_points_gps.shape[0],
        size=(bootstrap_iterations, target_points_gps.shape[0]),
    )

    results = {
        "coordinate_convention": (
            "H_target_from_source maps homogeneous column coordinates "
            "from source to target"
        ),
        "enu_axis_order": ["east_m", "north_m", "up_m"],
        "input_files": {
            "gps_points": str(Path(gps_points).resolve()),
            "imu_points": str(Path(imu_points).resolve()),
            "camera_antenna_point": str(Path(camera_antenna_point).resolve()),
            "manual_json_path": str(Path(manual_json_path).resolve()),
        },
        "settings": {
            "bootstrap_iterations": bootstrap_iterations,
            "gps_bootstrap_block_length": gps_bootstrap_block_length,
            "random_seed": random_seed,
            "confidence_level": confidence_level,
            "calibration_maximum_iterations": calibration_maximum_iterations,
            "calibration_initial_damping": calibration_initial_damping,
            "calibration_finite_difference_step": calibration_finite_difference_step,
            "calibration_convergence_tolerance": calibration_convergence_tolerance,
        },
        "gps_imu": {
            "baseline_method": baseline_method,
            "reference_geodetic": reference_geodetic,
            "mean_baseline_auxiliary_from_main_gps": mean_baseline_gps,
            "rotation_imu_from_gps": configured_rotation_imu_from_gps,
            "imu_origin_to_auxiliary_antenna_imu": configured_imu_lever_arm,
            "transform_imu_from_gps": transform_imu_from_gps_value,
            "baseline_interval_gps": percentile_interval(
                gps_bootstrap_result["baseline_samples_gps"], confidence_level
            ),
        },
        "cameras": {},
    }

    for camera_name in camera_names:
        print(f"Processing manual points for {camera_name}")
        image_points = camera_image_points_dict[camera_name]

        calibration_result = estimate_camera_intrinsics_distortion_and_pose(
            target_points_gps,
            image_points,
            maximum_iterations=calibration_maximum_iterations,
            initial_damping=calibration_initial_damping,
            finite_difference_step=calibration_finite_difference_step,
            convergence_tolerance=calibration_convergence_tolerance,
        )

        transform_camera_from_gps_value = calibration_result[
            "transform_camera_from_gps"
        ]
        transform_camera_from_imu_value = calculate_imu_to_camera_transform(
            transform_imu_from_gps_value, transform_camera_from_gps_value
        )
        camera_position_imu_value = invert_transform(
            transform_camera_from_imu_value
        )[:3, 3]

        camera_bootstrap_result = bootstrap_camera_poses_dlt(
            target_points_gps,
            image_points,
            calibration_result["camera_matrix"],
            calibration_result["distortion_coefficients"],
            bootstrap_iterations,
            rng,
            bootstrap_indices=shared_camera_bootstrap_indices,
        )
        transforms_camera_from_imu = np.full(
            (bootstrap_iterations, 4, 4), np.nan, dtype=float
        )
        camera_positions_imu = np.full(
            (bootstrap_iterations, 3), np.nan, dtype=float
        )

        for bootstrap_index in range(bootstrap_iterations):
            if not camera_bootstrap_result["valid_mask"][bootstrap_index]:
                continue

            transform_camera_from_imu_sample = calculate_imu_to_camera_transform(
                gps_bootstrap_result["transforms_imu_from_gps"][bootstrap_index],
                camera_bootstrap_result["transforms_camera_from_gps"][bootstrap_index],
            )
            transforms_camera_from_imu[bootstrap_index] = transform_camera_from_imu_sample
            camera_positions_imu[bootstrap_index] = invert_transform(
                transform_camera_from_imu_sample
            )[:3, 3]

        transform_summary = summarize_bootstrap_transforms(
            transforms_camera_from_imu,
            transform_camera_from_imu_value,
            confidence_level,
        )
        valid_mask = camera_bootstrap_result["valid_mask"]

        results["cameras"][camera_name] = {
            "observation_count": len(image_points),
            "manual_image_points": image_points,
            "target_points_gps": target_points_gps,
            "projection_matrix_dlt": calibration_result["projection_matrix_dlt"],
            "intrinsic_matrix": calibration_result["camera_matrix"],
            "distortion_model": "Brown-Conrady [k1, k2, p1, p2, k3]",
            "distortion_coefficients": calibration_result["distortion_coefficients"],
            "transform_camera_from_gps": transform_camera_from_gps_value,
            "camera_position_gps": calibration_result["camera_position_gps"],
            "transform_camera_from_imu": transform_camera_from_imu_value,
            "camera_position_imu": camera_position_imu_value,
            "reprojection_rmse_pixels": calibration_result["reprojection_rmse_pixels"],
            "positive_depth_count": calibration_result["positive_depth_count"],
            "nonlinear_refinement_converged": calibration_result["converged"],
            "bootstrap": {
                "valid_count": camera_bootstrap_result["valid_count"],
                "failure_count": camera_bootstrap_result["failure_count"],
                "camera_positions_gps": camera_bootstrap_result["camera_positions_gps"],
                "camera_positions_imu": camera_positions_imu,
                "transforms_camera_from_gps": camera_bootstrap_result["transforms_camera_from_gps"],
                "transforms_camera_from_imu": transforms_camera_from_imu,
                "camera_position_gps_interval": percentile_interval(
                    camera_bootstrap_result["camera_positions_gps"][valid_mask],
                    confidence_level,
                ),
                "camera_position_imu_interval": percentile_interval(
                    camera_positions_imu[valid_mask], confidence_level
                ),
                "transform_summary": transform_summary,
            },
        }

        print(
            f"{camera_name}: reprojection RMSE = "
            f"{calibration_result['reprojection_rmse_pixels']:.6f} pixels"
        )

    save_results_json(results, output_json)
    print(f"Saved coordinate transformations to {Path(output_json).resolve()}")
    
    # Generate visualization plots
    plot_reprojection_residuals(results, save_path=script_directory / "reprojection_residuals.png")
    plot_3d_calibration_scene(results, save_path=script_directory / "calibration_3d_scene.png")
    
    return results


def convert_to_json_compatible(value):
    if isinstance(value, np.ndarray):
        return convert_to_json_compatible(value.tolist())
    if isinstance(value, (float, np.floating)):
        float_value = float(value)
        if math.isnan(float_value):
            return "NaN"
        if math.isinf(float_value):
            return "Infinity" if float_value > 0.0 else "-Infinity"
        return float_value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {key: convert_to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [convert_to_json_compatible(item) for item in value]
    return value


def save_results_json(results, output_path):
    output_path = str(output_path)
    serializable_results = convert_to_json_compatible(results)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(serializable_results, output_file, indent=2, allow_nan=False)


def run_synthetic_self_test():
    """Run deterministic synthetic test with manual points."""
    reference_geodetic = np.array([40.0, -80.0, 300.0], dtype=float)

    yaw_rad = math.radians(12.0)
    pitch_rad = math.radians(-4.0)
    rotation_z = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
        [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotation_y = np.array([
        [math.cos(pitch_rad), 0.0, math.sin(pitch_rad)],
        [0.0, 1.0, 0.0],
        [-math.sin(pitch_rad), 0.0, math.cos(pitch_rad)],
    ])
    rotation_camera_from_gps = rotation_z @ rotation_y
    camera_position_gps = np.array([1.2, -0.4, -3.5], dtype=float)
    translation_camera_from_gps = -rotation_camera_from_gps @ camera_position_gps
    known_transform_camera_from_gps = make_transform(
        rotation_camera_from_gps, translation_camera_from_gps
    )

    camera_matrix = np.array([
        [900.0, 0.0, 360.0],
        [0.0, 910.0, 270.0],
        [0.0, 0.0, 1.0],
    ])
    world_points_gps = np.array([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.2], [-1.0, 1.0, 0.5], [1.0, 1.0, 0.9],
        [0.0, -0.5, 1.4], [0.5, 0.3, 1.8], [-0.4, 0.6, 2.2], [1.3, 0.1, 2.6],
        [-0.8, -0.2, 3.0], [0.2, 1.1, 3.4],
    ])
    image_points, depth = project_world_points(
        world_points_gps, known_transform_camera_from_gps, camera_matrix
    )

    estimated_pose = estimate_camera_pose_dlt(
        world_points_gps, image_points, camera_matrix, np.empty(0)
    )
    camera_position_error = np.linalg.norm(
        estimated_pose["camera_position_gps"] - camera_position_gps
    )
    assert camera_position_error < 1e-4, f"Synthetic test failed: position error {camera_position_error}"
    print("Synthetic self-test passed successfully!")
    

def plot_reprojection_residuals(results, save_path=None):
    """Plot 2D observed vs reprojected pixel coordinates and error vectors."""
    cameras = results["cameras"]
    num_cameras = len(cameras)
    
    fig, axes = plt.subplots(1, num_cameras, figsize=(6 * num_cameras, 5))
    if num_cameras == 1:
        axes = [axes]

    for ax, (cam_name, cam_data) in zip(axes, cameras.items()):
        obs = np.asarray(cam_data["manual_image_points"])
        
        # Calculate reprojected points using calibrated parameters
        K = np.asarray(cam_data["intrinsic_matrix"])
        H = np.asarray(cam_data["transform_camera_from_gps"])
        dist = np.asarray(cam_data["distortion_coefficients"])
        targets_3d = np.asarray(cam_data["target_points_gps"])
        
        reproj, _ = project_points_with_distortion(targets_3d, K, H, dist)
        
        # 2D Scatter plot
        ax.scatter(obs[:, 0], obs[:, 1], c='red', marker='o', label='Observed (Manual)', alpha=0.8)
        ax.scatter(reproj[:, 0], reproj[:, 1], c='blue', marker='x', label='Reprojected', alpha=0.8)
        
        # Draw residual vectors
        for i in range(len(obs)):
            ax.plot([obs[i, 0], reproj[i, 0]], [obs[i, 1], reproj[i, 1]], 'k--', alpha=0.5)

        rmse = cam_data["reprojection_rmse_pixels"]
        ax.set_title(f"{cam_name}\nRMSE: {rmse:.3f} px")
        ax.set_xlabel("u (pixels)")
        ax.set_ylabel("v (pixels)")
        ax.invert_yaxis()  # Image pixel coordinate system convention
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_3d_calibration_scene(results, save_path=None):
    """Plot 3D spatial geometry in ENU frame: target points, camera poses, and uncertainty clouds."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    cameras = results["cameras"]

    # 1. Plot 3D Target Points
    first_cam = next(iter(cameras.values()))
    target_pts = np.asarray(first_cam["target_points_gps"])
    ax.scatter(
        target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
        c='green', marker='^', s=50, label='Target GPS Points (ENU)'
    )

    colors = ['tab:blue', 'tab:orange', 'tab:purple']

    # 2. Plot Camera Poses & Bootstrap Uncertainty
    for idx, (cam_name, cam_data) in enumerate(cameras.items()):
        cam_color = colors[idx % len(colors)]
        cam_pos = np.asarray(cam_data["camera_position_gps"])
        
        # Plot estimated camera position
        ax.scatter(
            cam_pos[0], cam_pos[1], cam_pos[2],
            color=cam_color, marker='o', s=100, label=f"{cam_name} Center"
        )
        
        # Plot viewing direction (optical axis +Z in camera frame mapped to ENU)
        H_cam_from_gps = np.asarray(cam_data["transform_camera_from_gps"])
        R_cam_from_gps = H_cam_from_gps[:3, :3]
        optical_axis_enu = R_cam_from_gps.T @ np.array([0, 0, 1])  # Inverse rotation
        
        ax.quiver(
            cam_pos[0], cam_pos[1], cam_pos[2],
            optical_axis_enu[0], optical_axis_enu[1], optical_axis_enu[2],
            length=0.5, color=cam_color, normalize=True, alpha=0.8
        )

        # Plot bootstrap position cloud if available
        if "bootstrap" in cam_data and "camera_positions_gps" in cam_data["bootstrap"]:
            boot_positions = np.asarray(cam_data["bootstrap"]["camera_positions_gps"])
            valid_mask = np.all(np.isfinite(boot_positions), axis=1)
            ax.scatter(
                boot_positions[valid_mask, 0],
                boot_positions[valid_mask, 1],
                boot_positions[valid_mask, 2],
                color=cam_color, s=5, alpha=0.2, label=f"{cam_name} Bootstrap"
            )

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title("3D Camera-GPS Calibration Scene Geometry (ENU Frame)")
    ax.legend(loc='best')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    run_synthetic_self_test()
    # Uncomment to execute on local input files:
    calculate_from_input_files()