"""Calculate GPS, IMU, and camera coordinate transformations.

Transform names follow ``H_target_from_source``. Homogeneous column points are
transformed as

    point_target = H_target_from_source @ point_source

GPS coordinates are converted to a local East-North-Up (ENU) frame before any
camera calibration. A complete GPS-to-IMU transform requires the IMU attitude
and the physical lever arm from the IMU origin to the auxiliary GPS antenna.
Two stationary GPS antennas alone determine only a baseline, not a full 3-D
rotation.

The high-level ``calculate_all_sensor_transforms`` function expects camera data
in this form::

    camera_data = {
        "camera_1": {
            "target_geodetic": target_lla,       # N by 3: degrees, degrees, m
            "image_points": image_points,        # N by 2: u, v pixels
            "camera_matrix": K,                  # 3 by 3
            "distortion_coefficients": distortion,
        }
    }

For direct file execution, edit the input paths near the top of this file. GPS
CSV files require latitude, longitude, and ellipsoid-height columns. The camera
target CSV uses the same coordinate columns and may contain either a shared
``image_name`` column or ``camera_1_image``, ``camera_2_image``, and
``camera_3_image`` columns. When image columns are absent, the numerically
sorted image count in every camera directory must equal the CSV row count.

One camera target row supplies one 3-D GPS point. The corresponding image
supplies one detected 2-D black-dot center. Cameras must remain fixed during
the complete calibration sequence, and the collected target points must not
all be coplanar. Distortion uses the Brown-Conrady order [k1, k2, p1, p2, k3].
"""

import csv
import json
import math
from pathlib import Path
import re

import numpy as np


# -----------------------------------------------------------------------------
# Input and output paths
# -----------------------------------------------------------------------------

script_directory = Path(__file__).resolve().parent

gps_points = script_directory / "gps_main.csv"
imu_points = script_directory / "gps_imu_antenna.csv"
camera_antenna_point = script_directory / "camera_antenna_point.csv"

camera_1 = script_directory / "camera_1"
camera_2 = script_directory / "camera_2"
camera_3 = script_directory / "camera_3"

output_json = script_directory / "coordinate_transforms.json"


# A complete GPS-to-IMU transform cannot be recovered from two stationary GPS
# positions alone. Replace these two values using the measured installation.
# rotation_imu_from_gps maps ENU vectors into the IMU coordinate axes.
rotation_imu_from_gps = None
imu_origin_to_auxiliary_antenna_imu = None


# Calibration and bootstrap settings. Keep these values visible so that every
# result can be reproduced from the output JSON.
bootstrap_iterations = 200
gps_bootstrap_block_length = 1
random_seed = 20260827
confidence_level = 0.95

# Set a value only when both GPS files contain timestamps and the platform was
# moving. For the requested static one-hour calibration, None independently
# averages the two recordings and does not require row-by-row synchronization.
gps_maximum_time_difference_seconds = None


# Black-dot detection settings. Set a per-camera ROI to (x_min, y_min,
# x_max, y_max) if other dark circular objects occur in the images.
black_dot_threshold = None
black_dot_minimum_area_pixels = 20
black_dot_maximum_area_fraction = 0.10
black_dot_minimum_aspect_ratio = 0.65
black_dot_minimum_fill_ratio = 0.45

camera_rois = {
    "camera_1": None,
    "camera_2": None,
    "camera_3": None,
}


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

    x = (
        prime_vertical_radius + height_m
    ) * cos_latitude * cos_longitude

    y = (
        prime_vertical_radius + height_m
    ) * cos_latitude * sin_longitude

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

        height_m = horizontal_distance / np.cos(latitude_rad)
        height_m = height_m - prime_vertical_radius

        denominator = horizontal_distance * (
            1.0
            - WGS84_ECCENTRICITY_SQUARED
            * prime_vertical_radius
            / (prime_vertical_radius + height_m)
        )
        updated_latitude_rad = np.arctan2(z, denominator)

        maximum_change = np.max(
            np.abs(updated_latitude_rad - latitude_rad)
        )
        latitude_rad = updated_latitude_rad

        if maximum_change <= tolerance_rad:
            break
    else:
        raise ValueError("ECEF to geodetic conversion did not converge")

    sin_latitude = np.sin(latitude_rad)
    prime_vertical_radius = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - WGS84_ECCENTRICITY_SQUARED * sin_latitude ** 2
    )
    height_m = horizontal_distance / np.cos(latitude_rad)
    height_m = height_m - prime_vertical_radius

    return np.column_stack(
        [
            np.degrees(latitude_rad),
            np.degrees(longitude_rad),
            height_m,
        ]
    )


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

    if not np.all(np.isfinite(reference_geodetic)):
        raise ValueError("reference_geodetic contains NaN or infinite values")

    reference_ecef = geodetic_to_ecef(
        reference_geodetic.reshape(1, 3)
    )[0]

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

    if not np.all(np.isfinite(translation_target_from_source)):
        raise ValueError("translation_target_from_source contains invalid values")

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
    """Nearest-time matching without interpolating or deleting source data."""
    reference_timestamps = np.asarray(reference_timestamps, dtype=float).reshape(-1)
    measured_timestamps = np.asarray(measured_timestamps, dtype=float).reshape(-1)
    reference_positions = validate_point_array(
        reference_positions,
        3,
        "reference_positions",
    )
    measured_positions = validate_point_array(
        measured_positions,
        3,
        "measured_positions",
    )

    if reference_timestamps.size != reference_positions.shape[0]:
        raise ValueError("reference timestamps and positions have different lengths")

    if measured_timestamps.size != measured_positions.shape[0]:
        raise ValueError("measured timestamps and positions have different lengths")

    if maximum_time_difference < 0.0:
        raise ValueError("maximum_time_difference must be non-negative")

    if not np.all(np.isfinite(reference_timestamps)):
        raise ValueError("reference_timestamps contains invalid values")

    if not np.all(np.isfinite(measured_timestamps)):
        raise ValueError("measured_timestamps contains invalid values")

    measured_order = np.argsort(measured_timestamps)
    sorted_measured_timestamps = measured_timestamps[measured_order]

    reference_indices = []
    measured_indices = []
    time_differences = []
    unmatched_reference_indices = []

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
            unmatched_reference_indices.append(reference_index)

    reference_indices = np.asarray(reference_indices, dtype=int)
    measured_indices = np.asarray(measured_indices, dtype=int)

    return {
        "reference_indices": reference_indices,
        "measured_indices": measured_indices,
        "reference_positions": reference_positions[reference_indices],
        "measured_positions": measured_positions[measured_indices],
        "time_differences": np.asarray(time_differences, dtype=float),
        "unmatched_reference_indices": np.asarray(
            unmatched_reference_indices,
            dtype=int,
        ),
    }


def estimate_antenna_baseline(main_antenna_positions_gps, auxiliary_positions_gps):
    """Estimate the main-to-auxiliary antenna baseline in the GPS frame."""
    main_antenna_positions_gps = validate_point_array(
        main_antenna_positions_gps,
        3,
        "main_antenna_positions_gps",
    )
    auxiliary_positions_gps = validate_point_array(
        auxiliary_positions_gps,
        3,
        "auxiliary_positions_gps",
    )

    if main_antenna_positions_gps.shape[0] != auxiliary_positions_gps.shape[0]:
        raise ValueError("main and auxiliary antenna arrays must be synchronized")

    baseline_samples_gps = (
        auxiliary_positions_gps - main_antenna_positions_gps
    )
    mean_baseline_gps = np.mean(baseline_samples_gps, axis=0)

    if baseline_samples_gps.shape[0] > 1:
        baseline_standard_deviation = np.std(
            baseline_samples_gps,
            axis=0,
            ddof=1,
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
    """Construct H_imu_from_gps using attitude and the measured lever arm."""
    baseline_auxiliary_from_main_gps = np.asarray(
        baseline_auxiliary_from_main_gps,
        dtype=float,
    ).reshape(-1)
    imu_origin_to_auxiliary_antenna_imu = np.asarray(
        imu_origin_to_auxiliary_antenna_imu,
        dtype=float,
    ).reshape(-1)

    if baseline_auxiliary_from_main_gps.shape != (3,):
        raise ValueError("baseline_auxiliary_from_main_gps must contain 3 values")

    if imu_origin_to_auxiliary_antenna_imu.shape != (3,):
        raise ValueError(
            "imu_origin_to_auxiliary_antenna_imu must contain 3 values"
        )

    rotation_imu_from_gps = validate_rotation_matrix(
        rotation_imu_from_gps,
        "rotation_imu_from_gps",
    )
    rotation_gps_from_imu = rotation_imu_from_gps.T

    # The auxiliary antenna position is the IMU origin plus its lever arm.
    imu_origin_position_gps = (
        baseline_auxiliary_from_main_gps
        - rotation_gps_from_imu @ imu_origin_to_auxiliary_antenna_imu
    )

    translation_imu_from_gps = (
        -rotation_imu_from_gps @ imu_origin_position_gps
    )

    return make_transform(
        rotation_imu_from_gps,
        translation_imu_from_gps,
    )


def moving_block_bootstrap_indices(sample_count, block_length, rng):
    if sample_count < 1:
        raise ValueError("sample_count must be positive")

    if block_length < 1 or block_length > sample_count:
        raise ValueError("block_length must be between 1 and sample_count")

    indices = []
    maximum_start = sample_count - block_length

    while len(indices) < sample_count:
        block_start = int(rng.integers(0, maximum_start + 1))
        block_stop = block_start + block_length
        indices.extend(range(block_start, block_stop))

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
    main_antenna_positions_gps = validate_point_array(
        main_antenna_positions_gps,
        3,
        "main_antenna_positions_gps",
    )
    auxiliary_positions_gps = validate_point_array(
        auxiliary_positions_gps,
        3,
        "auxiliary_positions_gps",
    )

    if main_antenna_positions_gps.shape != auxiliary_positions_gps.shape:
        raise ValueError("main and auxiliary antenna arrays must have equal shapes")

    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")

    sample_count = main_antenna_positions_gps.shape[0]
    transforms_imu_from_gps = np.empty(
        (bootstrap_iterations, 4, 4),
        dtype=float,
    )
    baseline_samples = np.empty((bootstrap_iterations, 3), dtype=float)

    for bootstrap_index in range(bootstrap_iterations):
        sample_indices = moving_block_bootstrap_indices(
            sample_count,
            block_length,
            rng,
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

    if root_mean_square_distance <= np.finfo(float).eps:
        raise ValueError("2-D points do not have enough spatial variation")

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

    if root_mean_square_distance <= np.finfo(float).eps:
        raise ValueError("3-D points do not have enough spatial variation")

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
        distortion_coefficients,
        dtype=float,
    ).reshape(-1)

    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape 3 by 3")

    if not np.all(np.isfinite(camera_matrix)):
        raise ValueError("camera_matrix contains invalid values")

    if distortion_coefficients.size not in (0, 1, 4, 5):
        raise ValueError(
            "distortion_coefficients must contain 0, 1, 4, or 5 values"
        )

    if not np.all(np.isfinite(distortion_coefficients)):
        raise ValueError("distortion_coefficients contains invalid values")

    if distortion_coefficients.size == 0:
        return image_points.copy()

    if iteration_count < 1:
        raise ValueError("iteration_count must be positive")

    k1 = distortion_coefficients[0]
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0
    k3 = 0.0

    if distortion_coefficients.size >= 4:
        k2 = distortion_coefficients[1]
        p1 = distortion_coefficients[2]
        p2 = distortion_coefficients[3]

    if distortion_coefficients.size == 5:
        k3 = distortion_coefficients[4]

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

        if np.any(np.abs(radial_scale) <= np.finfo(float).eps):
            raise ValueError("radial distortion scale reached zero")

        tangential_x = (
            2.0 * p1 * x * y
            + p2 * (radius_squared + 2.0 * x ** 2)
        )
        tangential_y = (
            p1 * (radius_squared + 2.0 * y ** 2)
            + 2.0 * p2 * x * y
        )

        undistorted_normalized[:, 0] = (
            distorted_normalized[:, 0] - tangential_x
        ) / radial_scale
        undistorted_normalized[:, 1] = (
            distorted_normalized[:, 1] - tangential_y
        ) / radial_scale

    undistorted_homogeneous = np.column_stack(
        [
            undistorted_normalized,
            np.ones(undistorted_normalized.shape[0]),
        ]
    )
    undistorted_pixels_homogeneous = (
        camera_matrix @ undistorted_homogeneous.T
    ).T

    return (
        undistorted_pixels_homogeneous[:, :2]
        / undistorted_pixels_homogeneous[:, 2:3]
    )


def estimate_projection_matrix_dlt(world_points_gps, image_points):
    """Estimate a 3 by 4 projective camera matrix with normalized DLT."""
    world_points_gps = validate_point_array(
        world_points_gps,
        3,
        "world_points_gps",
        minimum_count=6,
    )
    image_points = validate_point_array(
        image_points,
        2,
        "image_points",
        minimum_count=6,
    )

    if world_points_gps.shape[0] != image_points.shape[0]:
        raise ValueError("world_points_gps and image_points must have equal lengths")

    centered_world_points = world_points_gps - np.mean(world_points_gps, axis=0)
    centered_image_points = image_points - np.mean(image_points, axis=0)

    if np.linalg.matrix_rank(centered_world_points) < 3:
        raise ValueError(
            "DLT world points are coplanar or otherwise rank deficient; "
            "collect non-coplanar target positions or use a planar pose method"
        )

    if np.linalg.matrix_rank(centered_image_points) < 2:
        raise ValueError("DLT image points are collinear or rank deficient")

    normalized_world_points, world_normalization = normalize_points_3d(
        world_points_gps
    )
    normalized_image_points, image_normalization = normalize_points_2d(
        image_points
    )

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

    if np.linalg.matrix_rank(design_matrix) < 11:
        raise ValueError("DLT design matrix is rank deficient")

    _, _, right_singular_vectors = np.linalg.svd(
        design_matrix,
        full_matrices=True,
    )
    normalized_projection = right_singular_vectors[-1].reshape(3, 4)

    projection_matrix = (
        np.linalg.inv(image_normalization)
        @ normalized_projection
        @ world_normalization
    )

    projection_norm = np.linalg.norm(projection_matrix)

    if projection_norm <= np.finfo(float).eps:
        raise ValueError("DLT returned a zero projection matrix")

    return projection_matrix / projection_norm


def decompose_projection_matrix(projection_matrix, camera_matrix):
    """Recover H_camera_from_gps and the camera center using known intrinsics."""
    projection_matrix = np.asarray(projection_matrix, dtype=float)
    camera_matrix = np.asarray(camera_matrix, dtype=float)

    if projection_matrix.shape != (3, 4):
        raise ValueError("projection_matrix must have shape 3 by 4")

    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape 3 by 3")

    if abs(np.linalg.det(camera_matrix)) <= np.finfo(float).eps:
        raise ValueError("camera_matrix is singular")

    extrinsic_scaled = np.linalg.inv(camera_matrix) @ projection_matrix
    rotation_scaled = extrinsic_scaled[:, :3]

    if np.linalg.det(rotation_scaled) < 0.0:
        extrinsic_scaled = -extrinsic_scaled
        rotation_scaled = extrinsic_scaled[:, :3]

    _, singular_values, _ = np.linalg.svd(rotation_scaled)
    scale = np.mean(singular_values)

    if scale <= np.finfo(float).eps:
        raise ValueError("projection matrix has an invalid extrinsic scale")

    rotation_approximate = rotation_scaled / scale
    left_singular_vectors, _, right_singular_vectors = np.linalg.svd(
        rotation_approximate
    )
    rotation_camera_from_gps = (
        left_singular_vectors @ right_singular_vectors
    )

    if np.linalg.det(rotation_camera_from_gps) < 0.0:
        left_singular_vectors[:, -1] = -left_singular_vectors[:, -1]
        rotation_camera_from_gps = (
            left_singular_vectors @ right_singular_vectors
        )

    translation_camera_from_gps = extrinsic_scaled[:, 3] / scale
    transform_camera_from_gps = make_transform(
        rotation_camera_from_gps,
        translation_camera_from_gps,
    )
    camera_position_gps = (
        -rotation_camera_from_gps.T @ translation_camera_from_gps
    )

    return transform_camera_from_gps, camera_position_gps


def project_world_points(world_points_gps, transform_camera_from_gps, camera_matrix):
    world_points_gps = validate_point_array(
        world_points_gps,
        3,
        "world_points_gps",
    )
    camera_matrix = np.asarray(camera_matrix, dtype=float)

    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape 3 by 3")

    points_camera = transform_points(
        transform_camera_from_gps,
        world_points_gps,
    )
    depth = points_camera[:, 2]

    if np.any(np.abs(depth) <= np.finfo(float).eps):
        raise ValueError("one or more points project at zero camera depth")

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
        image_points,
        camera_matrix,
        distortion_coefficients,
    )
    projection_matrix = estimate_projection_matrix_dlt(
        world_points_gps,
        undistorted_image_points,
    )
    transform_camera_from_gps, camera_position_gps = (
        decompose_projection_matrix(
            projection_matrix,
            camera_matrix,
        )
    )
    reprojected_points, depth = project_world_points(
        world_points_gps,
        transform_camera_from_gps,
        camera_matrix,
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
    world_points_gps = validate_point_array(
        world_points_gps,
        3,
        "world_points_gps",
        minimum_count=6,
    )
    image_points = validate_point_array(
        image_points,
        2,
        "image_points",
        minimum_count=6,
    )

    if world_points_gps.shape[0] != image_points.shape[0]:
        raise ValueError("world_points_gps and image_points must have equal lengths")

    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")

    point_count = world_points_gps.shape[0]

    if bootstrap_indices is None:
        bootstrap_indices = rng.integers(
            0,
            point_count,
            size=(bootstrap_iterations, point_count),
        )
    else:
        bootstrap_indices = np.asarray(bootstrap_indices, dtype=int)

        if bootstrap_indices.shape != (bootstrap_iterations, point_count):
            raise ValueError(
                "bootstrap_indices must have shape "
                f"({bootstrap_iterations}, {point_count})"
            )

        if np.any(bootstrap_indices < 0) or np.any(bootstrap_indices >= point_count):
            raise ValueError("bootstrap_indices contains an out-of-range index")

    transforms_camera_from_gps = np.full(
        (bootstrap_iterations, 4, 4),
        np.nan,
        dtype=float,
    )
    camera_positions_gps = np.full(
        (bootstrap_iterations, 3),
        np.nan,
        dtype=float,
    )
    reprojection_rmse_pixels = np.full(
        bootstrap_iterations,
        np.nan,
        dtype=float,
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
            transforms_camera_from_gps[bootstrap_index] = (
                pose_result["transform_camera_from_gps"]
            )
            camera_positions_gps[bootstrap_index] = (
                pose_result["camera_position_gps"]
            )

            full_reprojected_points, _ = project_world_points(
                world_points_gps,
                pose_result["transform_camera_from_gps"],
                camera_matrix,
            )
            full_undistorted_points = undistort_image_points(
                image_points,
                camera_matrix,
                distortion_coefficients,
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

    if values.shape[0] < 1:
        raise ValueError("values must contain at least one sample")

    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    lower_percentile = 50.0 * (1.0 - confidence_level)
    upper_percentile = 100.0 - lower_percentile
    return np.percentile(
        values,
        [lower_percentile, 50.0, upper_percentile],
        axis=0,
    )


def summarize_bootstrap_transforms(
    bootstrap_transforms,
    reference_transform,
    confidence_level,
):
    bootstrap_transforms = np.asarray(bootstrap_transforms, dtype=float)
    reference_transform = np.asarray(reference_transform, dtype=float)

    if bootstrap_transforms.ndim != 3 or bootstrap_transforms.shape[1:] != (4, 4):
        raise ValueError("bootstrap_transforms must have shape B by 4 by 4")

    if reference_transform.shape != (4, 4):
        raise ValueError("reference_transform must have shape 4 by 4")

    valid_mask = np.all(
        np.isfinite(bootstrap_transforms.reshape(bootstrap_transforms.shape[0], -1)),
        axis=1,
    )
    valid_transforms = bootstrap_transforms[valid_mask]

    if valid_transforms.shape[0] == 0:
        raise ValueError("no valid bootstrap transforms are available")

    translations = valid_transforms[:, :3, 3]
    rotation_errors_degrees = np.empty(valid_transforms.shape[0], dtype=float)

    for sample_index, transform in enumerate(valid_transforms):
        rotation_errors_degrees[sample_index] = rotation_difference_degrees(
            transform[:3, :3],
            reference_transform[:3, :3],
        )

    return {
        "valid_count": int(valid_transforms.shape[0]),
        "failure_count": int(np.count_nonzero(~valid_mask)),
        "translation_interval": percentile_interval(
            translations,
            confidence_level,
        ),
        "rotation_error_degrees_interval": percentile_interval(
            rotation_errors_degrees,
            confidence_level,
        ),
    }


def calculate_all_sensor_transforms(
    main_antenna_geodetic,
    auxiliary_antenna_geodetic,
    rotation_imu_from_gps,
    imu_origin_to_auxiliary_antenna_imu,
    camera_data,
    bootstrap_iterations,
    gps_block_length,
    random_seed,
    reference_geodetic=None,
    confidence_level=0.95,
    shared_camera_events=False,
):
    """Run the full GPS, IMU, and multi-camera transformation calculation."""
    main_antenna_geodetic = validate_point_array(
        main_antenna_geodetic,
        3,
        "main_antenna_geodetic",
    )
    auxiliary_antenna_geodetic = validate_point_array(
        auxiliary_antenna_geodetic,
        3,
        "auxiliary_antenna_geodetic",
    )

    if main_antenna_geodetic.shape != auxiliary_antenna_geodetic.shape:
        raise ValueError(
            "main and auxiliary geodetic arrays must be synchronized and equal"
        )

    if not isinstance(camera_data, dict) or len(camera_data) == 0:
        raise ValueError("camera_data must be a non-empty dictionary")

    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")

    if reference_geodetic is None:
        reference_geodetic = mean_geodetic_position(main_antenna_geodetic)
    else:
        reference_geodetic = np.asarray(reference_geodetic, dtype=float).reshape(-1)

        if reference_geodetic.shape != (3,):
            raise ValueError("reference_geodetic must contain 3 values")

    main_antenna_gps = geodetic_to_enu(
        main_antenna_geodetic,
        reference_geodetic,
    )
    auxiliary_antenna_gps = geodetic_to_enu(
        auxiliary_antenna_geodetic,
        reference_geodetic,
    )

    baseline_result = estimate_antenna_baseline(
        main_antenna_gps,
        auxiliary_antenna_gps,
    )
    transform_imu_from_gps = gps_to_imu_transform_from_baseline(
        baseline_result["mean_baseline_gps"],
        rotation_imu_from_gps,
        imu_origin_to_auxiliary_antenna_imu,
    )

    rng = np.random.default_rng(random_seed)
    gps_bootstrap = bootstrap_gps_to_imu_transforms(
        main_antenna_gps,
        auxiliary_antenna_gps,
        rotation_imu_from_gps,
        imu_origin_to_auxiliary_antenna_imu,
        bootstrap_iterations,
        gps_block_length,
        rng,
    )

    shared_indices = None

    if shared_camera_events:
        camera_point_counts = []

        for camera_name, data in camera_data.items():
            target_geodetic = validate_point_array(
                data["target_geodetic"],
                3,
                f"{camera_name} target_geodetic",
                minimum_count=6,
            )
            camera_point_counts.append(target_geodetic.shape[0])

        if len(set(camera_point_counts)) != 1:
            raise ValueError(
                "shared_camera_events requires the same observation count "
                "and row ordering for every camera"
            )

        shared_point_count = camera_point_counts[0]
        shared_indices = rng.integers(
            0,
            shared_point_count,
            size=(bootstrap_iterations, shared_point_count),
        )

    results = {
        "coordinate_convention": (
            "H_target_from_source maps homogeneous column coordinates "
            "from source to target"
        ),
        "enu_axis_order": ["east_m", "north_m", "up_m"],
        "reference_geodetic": reference_geodetic,
        "mean_baseline_auxiliary_from_main_gps": (
            baseline_result["mean_baseline_gps"]
        ),
        "baseline_standard_deviation": (
            baseline_result["baseline_standard_deviation"]
        ),
        "transform_imu_from_gps": transform_imu_from_gps,
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "gps_block_length": gps_block_length,
            "random_seed": random_seed,
            "confidence_level": confidence_level,
            "baseline_samples_gps": gps_bootstrap["baseline_samples_gps"],
            "transforms_imu_from_gps": gps_bootstrap[
                "transforms_imu_from_gps"
            ],
            "baseline_interval_gps": percentile_interval(
                gps_bootstrap["baseline_samples_gps"],
                confidence_level,
            ),
        },
        "cameras": {},
    }

    for camera_name, data in camera_data.items():
        target_geodetic = validate_point_array(
            data["target_geodetic"],
            3,
            f"{camera_name} target_geodetic",
            minimum_count=6,
        )
        image_points = validate_point_array(
            data["image_points"],
            2,
            f"{camera_name} image_points",
            minimum_count=6,
        )

        if target_geodetic.shape[0] != image_points.shape[0]:
            raise ValueError(
                f"{camera_name} target and image arrays have different lengths"
            )

        camera_matrix = np.asarray(data["camera_matrix"], dtype=float)
        distortion_coefficients = np.asarray(
            data.get("distortion_coefficients", np.empty(0)),
            dtype=float,
        )
        target_points_gps = geodetic_to_enu(
            target_geodetic,
            reference_geodetic,
        )

        pose_result = estimate_camera_pose_dlt(
            target_points_gps,
            image_points,
            camera_matrix,
            distortion_coefficients,
        )
        transform_camera_from_gps = pose_result["transform_camera_from_gps"]
        transform_camera_from_imu = calculate_imu_to_camera_transform(
            transform_imu_from_gps,
            transform_camera_from_gps,
        )
        transform_imu_from_camera = invert_transform(
            transform_camera_from_imu
        )

        camera_bootstrap = bootstrap_camera_poses_dlt(
            target_points_gps,
            image_points,
            camera_matrix,
            distortion_coefficients,
            bootstrap_iterations,
            rng,
            bootstrap_indices=shared_indices,
        )
        transforms_camera_from_imu = np.full(
            (bootstrap_iterations, 4, 4),
            np.nan,
            dtype=float,
        )
        camera_positions_imu = np.full(
            (bootstrap_iterations, 3),
            np.nan,
            dtype=float,
        )

        for bootstrap_index in range(bootstrap_iterations):
            if not camera_bootstrap["valid_mask"][bootstrap_index]:
                continue

            transforms_camera_from_imu[bootstrap_index] = (
                calculate_imu_to_camera_transform(
                    gps_bootstrap["transforms_imu_from_gps"][bootstrap_index],
                    camera_bootstrap["transforms_camera_from_gps"][bootstrap_index],
                )
            )
            bootstrap_transform_imu_from_camera = invert_transform(
                transforms_camera_from_imu[bootstrap_index]
            )
            camera_positions_imu[bootstrap_index] = (
                bootstrap_transform_imu_from_camera[:3, 3]
            )

        bootstrap_summary = summarize_bootstrap_transforms(
            transforms_camera_from_imu,
            transform_camera_from_imu,
            confidence_level,
        )

        results["cameras"][camera_name] = {
            "projection_matrix": pose_result["projection_matrix"],
            "camera_position_gps": pose_result["camera_position_gps"],
            "camera_position_imu": transform_imu_from_camera[:3, 3],
            "transform_camera_from_gps": transform_camera_from_gps,
            "transform_camera_from_imu": transform_camera_from_imu,
            "reprojection_rmse_pixels": pose_result[
                "reprojection_rmse_pixels"
            ],
            "positive_depth_count": pose_result["positive_depth_count"],
            "observation_count": target_points_gps.shape[0],
            "bootstrap": {
                "valid_count": camera_bootstrap["valid_count"],
                "failure_count": camera_bootstrap["failure_count"],
                "failure_messages": camera_bootstrap["failure_messages"],
                "valid_mask": camera_bootstrap["valid_mask"],
                "sample_indices": camera_bootstrap["bootstrap_indices"],
                "camera_positions_gps": camera_bootstrap[
                    "camera_positions_gps"
                ],
                "camera_positions_imu": camera_positions_imu,
                "transforms_camera_from_gps": camera_bootstrap[
                    "transforms_camera_from_gps"
                ],
                "transforms_camera_from_imu": transforms_camera_from_imu,
                "reprojection_rmse_pixels": camera_bootstrap[
                    "reprojection_rmse_pixels"
                ],
                "camera_position_gps_interval": percentile_interval(
                    camera_bootstrap["camera_positions_gps"][
                        camera_bootstrap["valid_mask"]
                    ],
                    confidence_level,
                ),
                "camera_position_imu_interval": percentile_interval(
                    camera_positions_imu[
                        camera_bootstrap["valid_mask"]
                    ],
                    confidence_level,
                ),
                "transform_summary": bootstrap_summary,
            },
        }

    return results


def find_csv_column(fieldnames, candidate_names, description, required=True):
    normalized_fields = {}

    for fieldname in fieldnames:
        normalized_fields[fieldname.strip().lower()] = fieldname

    for candidate_name in candidate_names:
        normalized_candidate = candidate_name.strip().lower()

        if normalized_candidate in normalized_fields:
            return normalized_fields[normalized_candidate]

    if required:
        expected = ", ".join(candidate_names)
        raise ValueError(
            f"CSV is missing {description}. Accepted column names: {expected}"
        )

    return None


def parse_csv_float(row, column_name, row_number, input_path):
    text = row.get(column_name, "")

    try:
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
    """Read GPS coordinates without silently removing invalid rows."""
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
            [
                "ellipsoid_height_m",
                "altitude_m",
                "height_m",
                "altitude",
                "height",
            ],
            "ellipsoid height",
        )
        timestamp_column = find_csv_column(
            reader.fieldnames,
            ["timestamp_s", "timestamp", "time_s", "unix_time_s", "time"],
            "timestamp",
            required=False,
        )

        geodetic_rows = []
        timestamps = []

        for row_number, row in enumerate(reader, start=2):
            latitude = parse_csv_float(
                row,
                latitude_column,
                row_number,
                input_path,
            )
            longitude = parse_csv_float(
                row,
                longitude_column,
                row_number,
                input_path,
            )
            height = parse_csv_float(
                row,
                height_column,
                row_number,
                input_path,
            )
            geodetic_rows.append([latitude, longitude, height])

            if timestamp_column is not None:
                timestamp = parse_csv_float(
                    row,
                    timestamp_column,
                    row_number,
                    input_path,
                )
                timestamps.append(timestamp)

    if len(geodetic_rows) == 0:
        raise ValueError(f"CSV contains no data rows: {input_path}")

    result = {
        "geodetic": np.asarray(geodetic_rows, dtype=float),
        "columns": {
            "latitude": latitude_column,
            "longitude": longitude_column,
            "height": height_column,
            "timestamp": timestamp_column,
        },
    }

    if timestamp_column is None:
        result["timestamps"] = None
    else:
        result["timestamps"] = np.asarray(timestamps, dtype=float)

    return result


def read_camera_target_csv(input_path, camera_names):
    """Read target GPS coordinates and optional per-camera image filenames."""
    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Camera target input file does not exist: {input_path}"
        )

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")

        latitude_column = find_csv_column(
            reader.fieldnames,
            ["latitude_deg", "latitude", "lat", "gps_latitude_deg"],
            "target latitude",
        )
        longitude_column = find_csv_column(
            reader.fieldnames,
            ["longitude_deg", "longitude", "lon", "gps_longitude_deg"],
            "target longitude",
        )
        height_column = find_csv_column(
            reader.fieldnames,
            [
                "ellipsoid_height_m",
                "altitude_m",
                "height_m",
                "altitude",
                "height",
            ],
            "target ellipsoid height",
        )
        shared_image_column = find_csv_column(
            reader.fieldnames,
            ["image_name", "filename", "image", "file_name"],
            "shared image filename",
            required=False,
        )

        camera_image_columns = {}

        for camera_name in camera_names:
            camera_number = camera_name.split("_")[-1]
            camera_image_columns[camera_name] = find_csv_column(
                reader.fieldnames,
                [
                    f"{camera_name}_image",
                    f"camera{camera_number}_image",
                    f"image_{camera_name}",
                    f"{camera_name}_filename",
                ],
                f"{camera_name} image filename",
                required=False,
            )

        geodetic_rows = []
        image_names = {}

        for camera_name in camera_names:
            image_names[camera_name] = []

        for row_number, row in enumerate(reader, start=2):
            latitude = parse_csv_float(
                row,
                latitude_column,
                row_number,
                input_path,
            )
            longitude = parse_csv_float(
                row,
                longitude_column,
                row_number,
                input_path,
            )
            height = parse_csv_float(
                row,
                height_column,
                row_number,
                input_path,
            )
            geodetic_rows.append([latitude, longitude, height])

            for camera_name in camera_names:
                image_column = camera_image_columns[camera_name]

                if image_column is None:
                    image_column = shared_image_column

                if image_column is None:
                    image_names[camera_name].append(None)
                    continue

                image_name = str(row.get(image_column, "")).strip()

                if image_name == "":
                    raise ValueError(
                        f"{input_path}, row {row_number}: "
                        f"{image_column} is empty"
                    )

                image_names[camera_name].append(image_name)

    if len(geodetic_rows) == 0:
        raise ValueError(f"CSV contains no target rows: {input_path}")

    return {
        "geodetic": np.asarray(geodetic_rows, dtype=float),
        "image_names": image_names,
        "columns": {
            "latitude": latitude_column,
            "longitude": longitude_column,
            "height": height_column,
            "shared_image": shared_image_column,
            "camera_images": camera_image_columns,
        },
    }


def numerical_filename_key(path):
    parts = re.split(r"(\d+)", Path(path).name.lower())
    key = []

    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))

    return key


def list_camera_images(camera_directory):
    camera_directory = Path(camera_directory)

    if not camera_directory.is_dir():
        raise FileNotFoundError(
            f"Camera image directory does not exist: {camera_directory}"
        )

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }
    image_paths = []

    for path in camera_directory.iterdir():
        if path.is_file() and path.suffix.lower() in supported_extensions:
            image_paths.append(path)

    image_paths.sort(key=numerical_filename_key)

    if len(image_paths) == 0:
        raise ValueError(f"No supported images found in {camera_directory}")

    return image_paths


def match_target_rows_to_images(camera_directory, requested_image_names):
    image_paths = list_camera_images(camera_directory)

    if all(image_name is None for image_name in requested_image_names):
        if len(image_paths) != len(requested_image_names):
            raise ValueError(
                f"{camera_directory} contains {len(image_paths)} images but "
                f"camera_antenna_point.csv contains {len(requested_image_names)} "
                "rows. Add image_name or camera-specific image columns."
            )

        return image_paths

    paths_by_name = {}
    paths_by_stem = {}

    for image_path in image_paths:
        paths_by_name[image_path.name.lower()] = image_path
        normalized_stem = image_path.stem.lower()

        if normalized_stem in paths_by_stem:
            paths_by_stem[normalized_stem] = None
        else:
            paths_by_stem[normalized_stem] = image_path

    matched_paths = []

    for row_index, requested_name in enumerate(requested_image_names):
        if requested_name is None:
            raise ValueError(
                "Image names must be provided for all rows or for no rows"
            )

        normalized_name = Path(requested_name).name.lower()
        matched_path = paths_by_name.get(normalized_name)

        if matched_path is None:
            normalized_stem = Path(requested_name).stem.lower()
            matched_path = paths_by_stem.get(normalized_stem)

        if matched_path is None:
            raise ValueError(
                f"No unique image matching '{requested_name}' for target row "
                f"{row_index + 2} in {camera_directory}"
            )

        matched_paths.append(matched_path)

    return matched_paths


def otsu_grayscale_threshold(gray_image):
    gray_image = np.asarray(gray_image, dtype=np.uint8)
    histogram = np.bincount(gray_image.reshape(-1), minlength=256).astype(float)
    pixel_count = gray_image.size
    total_intensity = np.dot(np.arange(256, dtype=float), histogram)
    background_weight = 0.0
    background_intensity = 0.0
    maximum_variance = -1.0
    selected_threshold = 0

    for threshold in range(256):
        background_weight += histogram[threshold]

        if background_weight == 0.0:
            continue

        foreground_weight = pixel_count - background_weight

        if foreground_weight == 0.0:
            break

        background_intensity += threshold * histogram[threshold]
        background_mean = background_intensity / background_weight
        foreground_mean = (
            total_intensity - background_intensity
        ) / foreground_weight
        between_class_variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )

        if between_class_variance > maximum_variance:
            maximum_variance = between_class_variance
            selected_threshold = threshold

    return selected_threshold


def detect_black_circular_dot(
    image_path,
    threshold=None,
    minimum_area_pixels=20,
    maximum_area_fraction=0.10,
    minimum_aspect_ratio=0.65,
    minimum_fill_ratio=0.45,
    roi=None,
):
    """Detect the most circle-like dark connected component in one image."""
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            "Pillow is required to read calibration images"
        ) from error

    image_path = Path(image_path)

    with Image.open(image_path) as image:
        gray_image = np.asarray(image.convert("L"), dtype=np.uint8)

    image_height, image_width = gray_image.shape

    if roi is None:
        x_min = 0
        y_min = 0
        x_max = image_width
        y_max = image_height
    else:
        if len(roi) != 4:
            raise ValueError("ROI must contain x_min, y_min, x_max, y_max")

        x_min, y_min, x_max, y_max = [int(value) for value in roi]

        if (
            x_min < 0
            or y_min < 0
            or x_max > image_width
            or y_max > image_height
            or x_min >= x_max
            or y_min >= y_max
        ):
            raise ValueError(f"Invalid ROI {roi} for image {image_path}")

    cropped_gray = gray_image[y_min:y_max, x_min:x_max]

    if threshold is None:
        used_threshold = otsu_grayscale_threshold(cropped_gray)
    else:
        used_threshold = int(threshold)

        if used_threshold < 0 or used_threshold > 255:
            raise ValueError("black dot threshold must be between 0 and 255")

    dark_mask = cropped_gray <= used_threshold
    maximum_area_pixels = int(
        maximum_area_fraction * cropped_gray.shape[0] * cropped_gray.shape[1]
    )

    if minimum_area_pixels < 1:
        raise ValueError("minimum_area_pixels must be positive")

    if maximum_area_fraction <= 0.0 or maximum_area_fraction > 1.0:
        raise ValueError("maximum_area_fraction must be between 0 and 1")

    if minimum_aspect_ratio <= 0.0 or minimum_aspect_ratio > 1.0:
        raise ValueError("minimum_aspect_ratio must be between 0 and 1")

    if minimum_fill_ratio <= 0.0 or minimum_fill_ratio > 1.0:
        raise ValueError("minimum_fill_ratio must be between 0 and 1")

    if maximum_area_pixels < minimum_area_pixels:
        raise ValueError("maximum dark component area is below the minimum area")

    visited = np.zeros(dark_mask.shape, dtype=bool)
    dark_rows, dark_columns = np.nonzero(dark_mask)
    candidates = []

    for start_row, start_column in zip(dark_rows, dark_columns):
        if visited[start_row, start_column]:
            continue

        stack = [(int(start_row), int(start_column))]
        visited[start_row, start_column] = True
        area = 0
        sum_x = 0.0
        sum_y = 0.0
        minimum_x = int(start_column)
        maximum_x = int(start_column)
        minimum_y = int(start_row)
        maximum_y = int(start_row)
        intensity_sum = 0.0

        while stack:
            row, column = stack.pop()
            area += 1
            sum_x += column
            sum_y += row
            intensity_sum += cropped_gray[row, column]
            minimum_x = min(minimum_x, column)
            maximum_x = max(maximum_x, column)
            minimum_y = min(minimum_y, row)
            maximum_y = max(maximum_y, row)

            for row_offset in (-1, 0, 1):
                neighbor_row = row + row_offset

                if neighbor_row < 0 or neighbor_row >= dark_mask.shape[0]:
                    continue

                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue

                    neighbor_column = column + column_offset

                    if (
                        neighbor_column < 0
                        or neighbor_column >= dark_mask.shape[1]
                    ):
                        continue

                    if (
                        dark_mask[neighbor_row, neighbor_column]
                        and not visited[neighbor_row, neighbor_column]
                    ):
                        visited[neighbor_row, neighbor_column] = True
                        stack.append((neighbor_row, neighbor_column))

        if area < minimum_area_pixels or area > maximum_area_pixels:
            continue

        bounding_width = maximum_x - minimum_x + 1
        bounding_height = maximum_y - minimum_y + 1
        aspect_ratio = min(bounding_width, bounding_height) / max(
            bounding_width,
            bounding_height,
        )
        fill_ratio = area / (bounding_width * bounding_height)

        if aspect_ratio < minimum_aspect_ratio:
            continue

        if fill_ratio < minimum_fill_ratio:
            continue

        shape_error = abs(1.0 - aspect_ratio)
        shape_error += abs(fill_ratio - math.pi / 4.0)
        candidates.append(
            {
                "center_u": sum_x / area + x_min,
                "center_v": sum_y / area + y_min,
                "area_pixels": area,
                "bounding_box": [
                    minimum_x + x_min,
                    minimum_y + y_min,
                    maximum_x + x_min,
                    maximum_y + y_min,
                ],
                "aspect_ratio": aspect_ratio,
                "fill_ratio": fill_ratio,
                "mean_grayscale": intensity_sum / area,
                "shape_error": shape_error,
            }
        )

    if len(candidates) == 0:
        raise ValueError(
            f"No black circular component passed the configured checks: {image_path}"
        )

    selected = candidates[0]

    for candidate in candidates[1:]:
        selected_score = (
            selected["shape_error"],
            -selected["area_pixels"],
        )
        candidate_score = (
            candidate["shape_error"],
            -candidate["area_pixels"],
        )

        if candidate_score < selected_score:
            selected = candidate

    selected["image_path"] = str(image_path)
    selected["threshold"] = used_threshold
    selected["candidate_count"] = len(candidates)
    return selected


def detect_camera_image_points(
    image_paths,
    threshold,
    minimum_area_pixels,
    maximum_area_fraction,
    minimum_aspect_ratio,
    minimum_fill_ratio,
    roi,
):
    image_points = np.empty((len(image_paths), 2), dtype=float)
    detection_records = []

    for image_index, image_path in enumerate(image_paths):
        detection = detect_black_circular_dot(
            image_path,
            threshold=threshold,
            minimum_area_pixels=minimum_area_pixels,
            maximum_area_fraction=maximum_area_fraction,
            minimum_aspect_ratio=minimum_aspect_ratio,
            minimum_fill_ratio=minimum_fill_ratio,
            roi=roi,
        )
        image_points[image_index] = [
            detection["center_u"],
            detection["center_v"],
        ]
        detection_records.append(detection)
        print(
            f"Detected black dot {image_index + 1}/{len(image_paths)}: "
            f"{Path(image_path).name} -> "
            f"({detection['center_u']:.3f}, {detection['center_v']:.3f})"
        )

    return image_points, detection_records


def rq_decomposition_3x3(matrix):
    matrix = np.asarray(matrix, dtype=float)

    if matrix.shape != (3, 3):
        raise ValueError("RQ decomposition input must have shape 3 by 3")

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
    """Decompose a DLT matrix into K and H_camera_from_gps."""
    projection_matrix = np.asarray(projection_matrix, dtype=float)

    if projection_matrix.shape != (3, 4):
        raise ValueError("projection_matrix must have shape 3 by 4")

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

    if abs(intrinsic_scale) <= np.finfo(float).eps:
        raise ValueError("DLT projection matrix has an invalid intrinsic scale")

    camera_matrix = camera_matrix / intrinsic_scale
    scaled_projection = projection_matrix / intrinsic_scale
    rotation_camera_from_gps = validate_rotation_matrix(
        rotation_camera_from_gps,
        "rotation_camera_from_gps",
    )
    translation_camera_from_gps = np.linalg.solve(
        camera_matrix,
        scaled_projection[:, 3],
    )
    transform_camera_from_gps = make_transform(
        rotation_camera_from_gps,
        translation_camera_from_gps,
    )

    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise ValueError("DLT decomposition returned non-positive focal length")

    if world_points_gps is not None:
        world_points_gps = validate_point_array(
            world_points_gps,
            3,
            "world_points_gps",
        )
        points_camera = transform_points(
            transform_camera_from_gps,
            world_points_gps,
        )

        if np.count_nonzero(points_camera[:, 2] > 0.0) < (
            world_points_gps.shape[0] / 2.0
        ):
            raise ValueError(
                "DLT decomposition places most target points behind the camera"
            )

    camera_position_gps = (
        -rotation_camera_from_gps.T @ translation_camera_from_gps
    )
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
        axis_norm = np.linalg.norm(axis)

        if axis_norm <= np.finfo(float).eps:
            raise ValueError("Could not recover rotation axis near 180 degrees")

        axis = axis / axis_norm
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

    if rotation_vector.shape != (3,):
        raise ValueError("rotation_vector must contain 3 values")

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
    camera_matrix = np.asarray(camera_matrix, dtype=float)
    transform_camera_from_gps = np.asarray(
        transform_camera_from_gps,
        dtype=float,
    )

    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape 3 by 3")

    if transform_camera_from_gps.shape != (4, 4):
        raise ValueError("transform_camera_from_gps must have shape 4 by 4")

    if distortion_coefficients is None:
        distortion_coefficients = np.zeros(5, dtype=float)
    else:
        distortion_coefficients = np.asarray(
            distortion_coefficients,
            dtype=float,
        ).reshape(-1)

        if distortion_coefficients.shape != (5,):
            raise ValueError("distortion_coefficients must contain 5 values")

    rotation_vector = rotation_matrix_to_vector(
        transform_camera_from_gps[:3, :3]
    )
    translation = transform_camera_from_gps[:3, 3]

    return np.concatenate(
        [
            np.array(
                [
                    camera_matrix[0, 0],
                    camera_matrix[1, 1],
                    camera_matrix[0, 1],
                    camera_matrix[0, 2],
                    camera_matrix[1, 2],
                ],
                dtype=float,
            ),
            rotation_vector,
            translation,
            distortion_coefficients,
        ]
    )


def calibration_matrices_from_parameters(parameters):
    parameters = np.asarray(parameters, dtype=float).reshape(-1)

    if parameters.shape != (16,):
        raise ValueError("calibration parameter vector must contain 16 values")

    focal_x, focal_y, skew, center_x, center_y = parameters[:5]

    if focal_x <= 0.0 or focal_y <= 0.0:
        raise ValueError("focal lengths must be positive")

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
        rotation_camera_from_gps,
        parameters[8:11],
    )
    distortion_coefficients = parameters[11:16].copy()
    return camera_matrix, transform_camera_from_gps, distortion_coefficients


def project_points_with_distortion(
    world_points_gps,
    camera_matrix,
    transform_camera_from_gps,
    distortion_coefficients,
):
    world_points_gps = validate_point_array(
        world_points_gps,
        3,
        "world_points_gps",
    )
    camera_matrix = np.asarray(camera_matrix, dtype=float)
    distortion_coefficients = np.asarray(
        distortion_coefficients,
        dtype=float,
    ).reshape(-1)

    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape 3 by 3")

    if distortion_coefficients.shape != (5,):
        raise ValueError("distortion_coefficients must contain 5 values")

    points_camera = transform_points(
        transform_camera_from_gps,
        world_points_gps,
    )
    depth = points_camera[:, 2]

    if np.any(np.abs(depth) <= np.finfo(float).eps):
        raise ValueError("one or more points have zero camera depth")

    x = points_camera[:, 0] / depth
    y = points_camera[:, 1] / depth
    k1, k2, p1, p2, k3 = distortion_coefficients
    radius_squared = x ** 2 + y ** 2
    radial_scale = (
        1.0
        + k1 * radius_squared
        + k2 * radius_squared ** 2
        + k3 * radius_squared ** 3
    )
    x_distorted = (
        x * radial_scale
        + 2.0 * p1 * x * y
        + p2 * (radius_squared + 2.0 * x ** 2)
    )
    y_distorted = (
        y * radial_scale
        + p1 * (radius_squared + 2.0 * y ** 2)
        + 2.0 * p2 * x * y
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
        raise ValueError("calibration candidate places a target behind the camera")

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
        positive_residuals = None
        negative_residuals = None

        try:
            positive_residuals = calibration_reprojection_residuals(
                positive_parameters,
                world_points_gps,
                image_points,
            )
        except ValueError:
            positive_residuals = None

        try:
            negative_residuals = calibration_reprojection_residuals(
                negative_parameters,
                world_points_gps,
                image_points,
            )
        except ValueError:
            negative_residuals = None

        if positive_residuals is not None and negative_residuals is not None:
            jacobian[:, parameter_index] = (
                positive_residuals - negative_residuals
            ) / (2.0 * step)
        elif positive_residuals is not None:
            jacobian[:, parameter_index] = (
                positive_residuals - residuals
            ) / step
        elif negative_residuals is not None:
            jacobian[:, parameter_index] = (
                residuals - negative_residuals
            ) / step
        else:
            raise ValueError(
                "Could not evaluate calibration residuals around parameter "
                f"index {parameter_index}"
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
        world_points_gps,
        3,
        "world_points_gps",
        minimum_count=8,
    )
    image_points = validate_point_array(
        image_points,
        2,
        "image_points",
        minimum_count=8,
    )

    if world_points_gps.shape[0] != image_points.shape[0]:
        raise ValueError("world points and image points must have equal lengths")

    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")

    if initial_damping <= 0.0:
        raise ValueError("initial_damping must be positive")

    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive")

    if convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be positive")

    parameters = calibration_parameters_from_matrices(
        initial_camera_matrix,
        initial_transform_camera_from_gps,
        np.zeros(5, dtype=float),
    )
    damping = float(initial_damping)
    converged = False
    accepted_iterations = 0

    residuals = calibration_reprojection_residuals(
        parameters,
        world_points_gps,
        image_points,
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
        damped_hessian = approximate_hessian + damping * np.diag(
            hessian_diagonal
        )

        try:
            parameter_update = np.linalg.solve(damped_hessian, -gradient)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue

        candidate_parameters = parameters + parameter_update

        try:
            candidate_residuals = calibration_reprojection_residuals(
                candidate_parameters,
                world_points_gps,
                image_points,
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
        parameters,
        residuals,
        world_points_gps,
        image_points,
        finite_difference_step,
    )

    normal_matrix_condition = np.linalg.cond(final_jacobian.T @ final_jacobian)
    camera_position_gps = invert_transform(
        final_transform_camera_from_gps
    )[:3, 3]

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
        world_points_gps,
        3,
        "world_points_gps",
        minimum_count=8,
    )
    image_points = validate_point_array(
        image_points,
        2,
        "image_points",
        minimum_count=8,
    )

    projection_matrix = estimate_projection_matrix_dlt(
        world_points_gps,
        image_points,
    )
    initial_camera_matrix, initial_transform, initial_camera_position = (
        decompose_projection_matrix_unknown_intrinsics(
            projection_matrix,
            world_points_gps,
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
    main_antenna_positions_gps,
    auxiliary_antenna_positions_gps,
):
    """Estimate a static baseline when the two logs are not time paired."""
    main_antenna_positions_gps = validate_point_array(
        main_antenna_positions_gps,
        3,
        "main_antenna_positions_gps",
    )
    auxiliary_antenna_positions_gps = validate_point_array(
        auxiliary_antenna_positions_gps,
        3,
        "auxiliary_antenna_positions_gps",
    )
    mean_main_position_gps = np.mean(main_antenna_positions_gps, axis=0)
    mean_auxiliary_position_gps = np.mean(
        auxiliary_antenna_positions_gps,
        axis=0,
    )
    mean_baseline_gps = mean_auxiliary_position_gps - mean_main_position_gps

    if main_antenna_positions_gps.shape[0] > 1:
        main_standard_deviation = np.std(
            main_antenna_positions_gps,
            axis=0,
            ddof=1,
        )
    else:
        main_standard_deviation = np.full(3, np.nan)

    if auxiliary_antenna_positions_gps.shape[0] > 1:
        auxiliary_standard_deviation = np.std(
            auxiliary_antenna_positions_gps,
            axis=0,
            ddof=1,
        )
    else:
        auxiliary_standard_deviation = np.full(3, np.nan)

    return {
        "mean_main_position_gps": mean_main_position_gps,
        "mean_auxiliary_position_gps": mean_auxiliary_position_gps,
        "mean_baseline_gps": mean_baseline_gps,
        "main_standard_deviation": main_standard_deviation,
        "auxiliary_standard_deviation": auxiliary_standard_deviation,
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
    """Independently resample two static GPS logs and estimate the baseline."""
    main_antenna_positions_gps = validate_point_array(
        main_antenna_positions_gps,
        3,
        "main_antenna_positions_gps",
    )
    auxiliary_antenna_positions_gps = validate_point_array(
        auxiliary_antenna_positions_gps,
        3,
        "auxiliary_antenna_positions_gps",
    )

    if iteration_count < 1:
        raise ValueError("iteration_count must be positive")

    if block_length > main_antenna_positions_gps.shape[0]:
        raise ValueError("GPS block length exceeds the main GPS sample count")

    if block_length > auxiliary_antenna_positions_gps.shape[0]:
        raise ValueError("GPS block length exceeds the IMU antenna sample count")

    baseline_samples_gps = np.empty((iteration_count, 3), dtype=float)
    transforms_imu_from_gps = np.empty((iteration_count, 4, 4), dtype=float)

    for bootstrap_index in range(iteration_count):
        main_indices = moving_block_bootstrap_indices(
            main_antenna_positions_gps.shape[0],
            block_length,
            rng,
        )
        auxiliary_indices = moving_block_bootstrap_indices(
            auxiliary_antenna_positions_gps.shape[0],
            block_length,
            rng,
        )
        mean_main = np.mean(main_antenna_positions_gps[main_indices], axis=0)
        mean_auxiliary = np.mean(
            auxiliary_antenna_positions_gps[auxiliary_indices],
            axis=0,
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
    """Run the configured file-based three-camera calibration workflow."""
    if rotation_imu_from_gps is None:
        raise ValueError(
            "Set rotation_imu_from_gps near the top of this file. Two static "
            "GPS antennas do not determine the IMU coordinate-axis rotation."
        )

    if imu_origin_to_auxiliary_antenna_imu is None:
        raise ValueError(
            "Set imu_origin_to_auxiliary_antenna_imu near the top of this file "
            "using the measured lever arm in meters."
        )

    configured_rotation_imu_from_gps = validate_rotation_matrix(
        rotation_imu_from_gps,
        "rotation_imu_from_gps",
    )
    configured_imu_lever_arm = np.asarray(
        imu_origin_to_auxiliary_antenna_imu,
        dtype=float,
    ).reshape(-1)

    if configured_imu_lever_arm.shape != (3,):
        raise ValueError(
            "imu_origin_to_auxiliary_antenna_imu must contain 3 values in meters"
        )

    main_gps_data = read_geodetic_csv(gps_points)
    imu_antenna_data = read_geodetic_csv(imu_points)
    reference_geodetic = mean_geodetic_position(main_gps_data["geodetic"])
    main_positions_gps = geodetic_to_enu(
        main_gps_data["geodetic"],
        reference_geodetic,
    )
    auxiliary_positions_gps = geodetic_to_enu(
        imu_antenna_data["geodetic"],
        reference_geodetic,
    )
    rng = np.random.default_rng(random_seed)

    if gps_maximum_time_difference_seconds is None:
        baseline_method = "independent static means"
        baseline_result = estimate_static_antenna_baseline(
            main_positions_gps,
            auxiliary_positions_gps,
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
        if main_gps_data["timestamps"] is None:
            raise ValueError(
                "gps_main.csv needs a timestamp column when timestamp matching "
                "is enabled"
            )

        if imu_antenna_data["timestamps"] is None:
            raise ValueError(
                "gps_imu_antenna.csv needs a timestamp column when timestamp "
                "matching is enabled"
            )

        baseline_method = "nearest timestamp paired samples"
        gps_matching_result = match_positions_by_timestamp(
            main_gps_data["timestamps"],
            main_positions_gps,
            imu_antenna_data["timestamps"],
            auxiliary_positions_gps,
            gps_maximum_time_difference_seconds,
        )

        if gps_matching_result["reference_positions"].shape[0] == 0:
            raise ValueError("GPS timestamp matching produced no valid pairs")

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

    camera_directories = {
        "camera_1": Path(camera_1),
        "camera_2": Path(camera_2),
        "camera_3": Path(camera_3),
    }
    camera_names = list(camera_directories.keys())
    target_data = read_camera_target_csv(
        camera_antenna_point,
        camera_names,
    )
    target_points_gps = geodetic_to_enu(
        target_data["geodetic"],
        reference_geodetic,
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
            "camera_antenna_point": str(
                Path(camera_antenna_point).resolve()
            ),
            "camera_directories": {
                name: str(path.resolve())
                for name, path in camera_directories.items()
            },
        },
        "csv_columns": {
            "gps_main": main_gps_data["columns"],
            "gps_imu_antenna": imu_antenna_data["columns"],
            "camera_antenna_point": target_data["columns"],
        },
        "settings": {
            "bootstrap_iterations": bootstrap_iterations,
            "gps_bootstrap_block_length": gps_bootstrap_block_length,
            "random_seed": random_seed,
            "confidence_level": confidence_level,
            "gps_maximum_time_difference_seconds": (
                gps_maximum_time_difference_seconds
            ),
            "black_dot_threshold": black_dot_threshold,
            "black_dot_minimum_area_pixels": (
                black_dot_minimum_area_pixels
            ),
            "black_dot_maximum_area_fraction": (
                black_dot_maximum_area_fraction
            ),
            "black_dot_minimum_aspect_ratio": (
                black_dot_minimum_aspect_ratio
            ),
            "black_dot_minimum_fill_ratio": black_dot_minimum_fill_ratio,
            "camera_rois": camera_rois,
            "calibration_maximum_iterations": (
                calibration_maximum_iterations
            ),
            "calibration_initial_damping": calibration_initial_damping,
            "calibration_finite_difference_step": (
                calibration_finite_difference_step
            ),
            "calibration_convergence_tolerance": (
                calibration_convergence_tolerance
            ),
        },
        "gps_imu": {
            "baseline_method": baseline_method,
            "reference_geodetic": reference_geodetic,
            "mean_baseline_auxiliary_from_main_gps": mean_baseline_gps,
            "rotation_imu_from_gps": configured_rotation_imu_from_gps,
            "imu_origin_to_auxiliary_antenna_imu": configured_imu_lever_arm,
            "transform_imu_from_gps": transform_imu_from_gps_value,
            "baseline_interval_gps": percentile_interval(
                gps_bootstrap_result["baseline_samples_gps"],
                confidence_level,
            ),
            "baseline_samples_gps": gps_bootstrap_result[
                "baseline_samples_gps"
            ],
            "transforms_imu_from_gps": gps_bootstrap_result[
                "transforms_imu_from_gps"
            ],
        },
        "cameras": {},
    }

    if gps_matching_result is not None:
        results["gps_imu"]["matched_sample_count"] = int(
            gps_matching_result["reference_indices"].size
        )
        results["gps_imu"]["unmatched_main_indices"] = (
            gps_matching_result["unmatched_reference_indices"]
        )
        results["gps_imu"]["time_differences_seconds"] = (
            gps_matching_result["time_differences"]
        )

    for camera_name in camera_names:
        print(f"Processing {camera_name}")
        image_paths = match_target_rows_to_images(
            camera_directories[camera_name],
            target_data["image_names"][camera_name],
        )
        image_points, detection_records = detect_camera_image_points(
            image_paths,
            black_dot_threshold,
            black_dot_minimum_area_pixels,
            black_dot_maximum_area_fraction,
            black_dot_minimum_aspect_ratio,
            black_dot_minimum_fill_ratio,
            camera_rois[camera_name],
        )
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
            transform_imu_from_gps_value,
            transform_camera_from_gps_value,
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
            (bootstrap_iterations, 4, 4),
            np.nan,
            dtype=float,
        )
        camera_positions_imu = np.full(
            (bootstrap_iterations, 3),
            np.nan,
            dtype=float,
        )

        for bootstrap_index in range(bootstrap_iterations):
            if not camera_bootstrap_result["valid_mask"][bootstrap_index]:
                continue

            transform_camera_from_imu_sample = (
                calculate_imu_to_camera_transform(
                    gps_bootstrap_result["transforms_imu_from_gps"][
                        bootstrap_index
                    ],
                    camera_bootstrap_result["transforms_camera_from_gps"][
                        bootstrap_index
                    ],
                )
            )
            transforms_camera_from_imu[bootstrap_index] = (
                transform_camera_from_imu_sample
            )
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
            "image_count": len(image_paths),
            "image_paths": [str(path.resolve()) for path in image_paths],
            "black_dot_detections": detection_records,
            "image_points": image_points,
            "target_points_gps": target_points_gps,
            "projection_matrix_dlt": calibration_result[
                "projection_matrix_dlt"
            ],
            "intrinsic_matrix": calibration_result["camera_matrix"],
            "distortion_model": "Brown-Conrady [k1, k2, p1, p2, k3]",
            "distortion_coefficients": calibration_result[
                "distortion_coefficients"
            ],
            "transform_camera_from_gps": transform_camera_from_gps_value,
            "camera_position_gps": calibration_result[
                "camera_position_gps"
            ],
            "transform_camera_from_imu": transform_camera_from_imu_value,
            "camera_position_imu": camera_position_imu_value,
            "reprojection_rmse_pixels": calibration_result[
                "reprojection_rmse_pixels"
            ],
            "positive_depth_count": calibration_result[
                "positive_depth_count"
            ],
            "nonlinear_refinement_converged": calibration_result[
                "converged"
            ],
            "nonlinear_refinement_accepted_iterations": calibration_result[
                "accepted_iterations"
            ],
            "normal_matrix_condition": calibration_result[
                "normal_matrix_condition"
            ],
            "bootstrap": {
                "valid_count": camera_bootstrap_result["valid_count"],
                "failure_count": camera_bootstrap_result["failure_count"],
                "failure_messages": camera_bootstrap_result[
                    "failure_messages"
                ],
                "sample_indices": camera_bootstrap_result[
                    "bootstrap_indices"
                ],
                "camera_positions_gps": camera_bootstrap_result[
                    "camera_positions_gps"
                ],
                "camera_positions_imu": camera_positions_imu,
                "transforms_camera_from_gps": camera_bootstrap_result[
                    "transforms_camera_from_gps"
                ],
                "transforms_camera_from_imu": transforms_camera_from_imu,
                "camera_position_gps_interval": percentile_interval(
                    camera_bootstrap_result["camera_positions_gps"][
                        valid_mask
                    ],
                    confidence_level,
                ),
                "camera_position_imu_interval": percentile_interval(
                    camera_positions_imu[valid_mask],
                    confidence_level,
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
    return results


def convert_to_json_compatible(value):
    if isinstance(value, np.ndarray):
        return convert_to_json_compatible(value.tolist())

    if isinstance(value, (float, np.floating)):
        float_value = float(value)

        if math.isnan(float_value):
            return "NaN"

        if math.isinf(float_value):
            if float_value > 0.0:
                return "Infinity"

            return "-Infinity"

        return float_value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        converted = {}

        for key, item in value.items():
            converted[key] = convert_to_json_compatible(item)

        return converted

    if isinstance(value, list):
        return [convert_to_json_compatible(item) for item in value]

    return value


def save_results_json(results, output_path):
    output_path = str(output_path)
    serializable_results = convert_to_json_compatible(results)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(
            serializable_results,
            output_file,
            indent=2,
            allow_nan=False,
        )


def run_synthetic_self_test():
    """Run deterministic checks without requiring experimental input files."""
    reference_geodetic = np.array([40.0, -80.0, 300.0], dtype=float)
    reference_enu = geodetic_to_enu(
        reference_geodetic.reshape(1, 3),
        reference_geodetic,
    )
    assert np.linalg.norm(reference_enu) < 1e-8

    yaw_rad = math.radians(12.0)
    pitch_rad = math.radians(-4.0)
    rotation_z = np.array(
        [
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rotation_y = np.array(
        [
            [math.cos(pitch_rad), 0.0, math.sin(pitch_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch_rad), 0.0, math.cos(pitch_rad)],
        ],
        dtype=float,
    )
    rotation_camera_from_gps = rotation_z @ rotation_y
    camera_position_gps = np.array([1.2, -0.4, -3.5], dtype=float)
    translation_camera_from_gps = (
        -rotation_camera_from_gps @ camera_position_gps
    )
    known_transform_camera_from_gps = make_transform(
        rotation_camera_from_gps,
        translation_camera_from_gps,
    )

    camera_matrix = np.array(
        [
            [900.0, 0.0, 360.0],
            [0.0, 910.0, 270.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    world_points_gps = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.2],
            [-1.0, 1.0, 0.5],
            [1.0, 1.0, 0.9],
            [0.0, -0.5, 1.4],
            [0.5, 0.3, 1.8],
            [-0.4, 0.6, 2.2],
            [1.3, 0.1, 2.6],
            [-0.8, -0.2, 3.0],
            [0.2, 1.1, 3.4],
        ],
        dtype=float,
    )
    image_points, depth = project_world_points(
        world_points_gps,
        known_transform_camera_from_gps,
        camera_matrix,
    )
    assert np.all(depth > 0.0)

    estimated_pose = estimate_camera_pose_dlt(
        world_points_gps,
        image_points,
        camera_matrix,
        np.empty(0),
    )
    camera_position_error_m = np.linalg.norm(
        estimated_pose["camera_position_gps"] - camera_position_gps
    )
    rotation_error_degrees = rotation_difference_degrees(
        estimated_pose["transform_camera_from_gps"][:3, :3],
        rotation_camera_from_gps,
    )

    assert camera_position_error_m < 1e-8
    assert rotation_error_degrees < 1e-6
    assert estimated_pose["reprojection_rmse_pixels"] < 1e-8

    rotation_imu_from_gps = np.eye(3, dtype=float)
    auxiliary_baseline_gps = np.array([0.25, -0.1, 0.8], dtype=float)
    imu_to_auxiliary_imu = np.array([0.0, 0.0, 0.2], dtype=float)
    transform_imu_from_gps = gps_to_imu_transform_from_baseline(
        auxiliary_baseline_gps,
        rotation_imu_from_gps,
        imu_to_auxiliary_imu,
    )
    transform_camera_from_imu = calculate_imu_to_camera_transform(
        transform_imu_from_gps,
        known_transform_camera_from_gps,
    )
    recomposed_transform = (
        transform_camera_from_imu @ transform_imu_from_gps
    )

    assert np.allclose(
        recomposed_transform,
        known_transform_camera_from_gps,
        atol=1e-10,
    )

    print("Synthetic coordinate calculation self-test passed")
    print(f"Camera position error: {camera_position_error_m:.3e} m")
    print(f"Camera rotation error: {rotation_error_degrees:.3e} degrees")
    print(
        "Reprojection RMSE: "
        f"{estimated_pose['reprojection_rmse_pixels']:.3e} pixels"
    )


if __name__ == "__main__":
    calculate_from_input_files()
