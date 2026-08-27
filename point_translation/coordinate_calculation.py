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

Distortion coefficients use the order [k1, k2, p1, p2, k3]. A single value is
also accepted for a simple radial k1 model. Pass an empty array when image
points are already undistorted.
"""

import json
import math

import numpy as np


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


def convert_to_json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

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
        json.dump(serializable_results, output_file, indent=2)


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
    run_synthetic_self_test()
