# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:01:42 2026

@author: Desktop
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# Folder and output settings
# ============================================================

input_folder = Path(r"D:\GPS_Data")

output_file = input_folder / "GPS_KMeans_XYZ.txt"

# False means only read CSV files directly inside the folder
# True means also read CSV files inside subfolders
search_subfolders = False


# ============================================================
# KMeans settings
# ============================================================

# Enter a fixed number such as 10
# Enter None to automatically select the cluster count
number_of_clusters = 10

minimum_clusters = 2
maximum_clusters = 15


# ============================================================
# CSV column names
# ============================================================

latitude_column = "latitude"
longitude_column = "longitude"
altitude_column = "altitude_m"

required_columns = [
    latitude_column,
    longitude_column,
    altitude_column
]


# ============================================================
# Find all CSV files
# ============================================================

if search_subfolders:

    csv_files = list(
        input_folder.rglob("*.csv")
    )

else:

    csv_files = list(
        input_folder.glob("*.csv")
    )


# Sort filenames naturally
# Example order:
# GPS_1.csv
# GPS_2.csv
# GPS_10.csv

csv_files = sorted(
    csv_files,
    key=lambda path: [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]
)


if len(csv_files) == 0:
    raise FileNotFoundError(
        "No CSV files were found in: "
        + str(input_folder)
    )


print()
print("CSV files found:", len(csv_files))

for csv_file in csv_files:
    print(csv_file.name)


# ============================================================
# Read and combine all valid CSV files
# ============================================================

gps_dataframes = []

used_files = []
skipped_files = []

for csv_file in csv_files:

    print()
    print("Reading:", csv_file.name)

    try:

        current_df = pd.read_csv(
            csv_file
        )

    except Exception as error:

        print("Unable to read:", csv_file.name)
        print(error)

        skipped_files.append(
            csv_file.name
        )

        continue


    missing_columns = [
        column
        for column in required_columns
        if column not in current_df.columns
    ]


    if len(missing_columns) > 0:

        print(
            "Skipped because columns are missing:",
            missing_columns
        )

        skipped_files.append(
            csv_file.name
        )

        continue


    current_df = current_df[
        required_columns
    ].copy()


    current_df[latitude_column] = pd.to_numeric(
        current_df[latitude_column],
        errors="coerce"
    )

    current_df[longitude_column] = pd.to_numeric(
        current_df[longitude_column],
        errors="coerce"
    )

    current_df[altitude_column] = pd.to_numeric(
        current_df[altitude_column],
        errors="coerce"
    )


    current_df = current_df.dropna(
        subset=required_columns
    )


    finite_mask = np.isfinite(
        current_df[required_columns].to_numpy(
            dtype=float
        )
    ).all(axis=1)


    current_df = current_df.loc[
        finite_mask
    ].copy()


    if len(current_df) == 0:

        print(
            "Skipped because no valid GPS rows were found"
        )

        skipped_files.append(
            csv_file.name
        )

        continue


    gps_dataframes.append(
        current_df
    )

    used_files.append(
        csv_file.name
    )

    print(
        "Valid rows:",
        len(current_df)
    )


if len(gps_dataframes) == 0:

    raise ValueError(
        "No valid GPS data were found"
    )


gps_df = pd.concat(
    gps_dataframes,
    ignore_index=True
)


latitude = gps_df[
    latitude_column
].to_numpy(dtype=float)

longitude = gps_df[
    longitude_column
].to_numpy(dtype=float)

altitude = gps_df[
    altitude_column
].to_numpy(dtype=float)


if len(gps_df) < 2:

    raise ValueError(
        "Not enough valid GPS points"
    )


# ============================================================
# Convert latitude, longitude, altitude to ECEF
# ============================================================

semi_major_axis = 6378137.0

flattening = (
    1.0
    / 298.257223563
)

eccentricity_squared = (
    flattening
    * np.subtract(
        2.0,
        flattening
    )
)


latitude_rad = np.radians(
    latitude
)

longitude_rad = np.radians(
    longitude
)


sin_latitude = np.sin(
    latitude_rad
)

cos_latitude = np.cos(
    latitude_rad
)

sin_longitude = np.sin(
    longitude_rad
)

cos_longitude = np.cos(
    longitude_rad
)


prime_vertical_radius = (
    semi_major_axis
    / np.sqrt(
        np.subtract(
            1.0,
            eccentricity_squared
            * sin_latitude ** 2
        )
    )
)


ecef_x = (
    prime_vertical_radius
    + altitude
) * cos_latitude * cos_longitude


ecef_y = (
    prime_vertical_radius
    + altitude
) * cos_latitude * sin_longitude


ecef_z = (
    prime_vertical_radius
    * np.subtract(
        1.0,
        eccentricity_squared
    )
    + altitude
) * sin_latitude


ecef_points = np.column_stack(
    [
        ecef_x,
        ecef_y,
        ecef_z
    ]
)


# ============================================================
# Use the first valid GPS sample as the ENU reference
# ============================================================

reference_latitude = latitude[0]
reference_longitude = longitude[0]
reference_altitude = altitude[0]


reference_latitude_rad = math.radians(
    reference_latitude
)

reference_longitude_rad = math.radians(
    reference_longitude
)


reference_sin_latitude = math.sin(
    reference_latitude_rad
)

reference_cos_latitude = math.cos(
    reference_latitude_rad
)

reference_sin_longitude = math.sin(
    reference_longitude_rad
)

reference_cos_longitude = math.cos(
    reference_longitude_rad
)


reference_prime_vertical_radius = (
    semi_major_axis
    / math.sqrt(
        np.subtract(
            1.0,
            eccentricity_squared
            * reference_sin_latitude ** 2
        )
    )
)


reference_ecef_x = (
    reference_prime_vertical_radius
    + reference_altitude
) * reference_cos_latitude * reference_cos_longitude


reference_ecef_y = (
    reference_prime_vertical_radius
    + reference_altitude
) * reference_cos_latitude * reference_sin_longitude


reference_ecef_z = (
    reference_prime_vertical_radius
    * np.subtract(
        1.0,
        eccentricity_squared
    )
    + reference_altitude
) * reference_sin_latitude


reference_ecef = np.array(
    [
        reference_ecef_x,
        reference_ecef_y,
        reference_ecef_z
    ],
    dtype=float
)


rotation_matrix = np.array(
    [
        [
            np.negative(
                reference_sin_longitude
            ),

            reference_cos_longitude,

            0.0
        ],

        [
            np.negative(
                reference_sin_latitude
                * reference_cos_longitude
            ),

            np.negative(
                reference_sin_latitude
                * reference_sin_longitude
            ),

            reference_cos_latitude
        ],

        [
            reference_cos_latitude
            * reference_cos_longitude,

            reference_cos_latitude
            * reference_sin_longitude,

            reference_sin_latitude
        ]
    ],
    dtype=float
)


ecef_difference = np.subtract(
    ecef_points,
    reference_ecef
)


enu_points = (
    ecef_difference
    @ rotation_matrix.T
)


# ============================================================
# Automatically select the cluster count
# ============================================================

if number_of_clusters is None:

    maximum_test_clusters = int(
        min(
            maximum_clusters,
            np.subtract(
                len(enu_points),
                1
            )
        )
    )


    best_cluster_count = None
    best_silhouette_score = None


    for cluster_count in range(
        minimum_clusters,
        maximum_test_clusters + 1
    ):

        test_model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=20,
            max_iter=500
        )


        test_labels = test_model.fit_predict(
            enu_points
        )


        if len(np.unique(test_labels)) < 2:
            continue


        current_score = silhouette_score(
            enu_points,
            test_labels
        )


        print(
            "Cluster count:",
            cluster_count,
            "Silhouette score:",
            current_score
        )


        if best_silhouette_score is None:

            best_silhouette_score = current_score
            best_cluster_count = cluster_count

        elif current_score > best_silhouette_score:

            best_silhouette_score = current_score
            best_cluster_count = cluster_count


    if best_cluster_count is None:

        raise RuntimeError(
            "Unable to determine the cluster count"
        )


    number_of_clusters = best_cluster_count


# ============================================================
# Check the cluster count
# ============================================================

if number_of_clusters < 1:

    raise ValueError(
        "The cluster count must be greater than zero"
    )


if number_of_clusters > len(enu_points):

    raise ValueError(
        "The cluster count is larger than the number of GPS samples"
    )


# ============================================================
# Final KMeans clustering
# ============================================================

kmeans_model = KMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    n_init=20,
    max_iter=500
)


cluster_labels = kmeans_model.fit_predict(
    enu_points
)


cluster_centers = (
    kmeans_model.cluster_centers_
)


# ============================================================
# Sort cluster centers by acquisition order
# ============================================================

cluster_order = []


for cluster_id in range(
    number_of_clusters
):

    cluster_indices = np.where(
        cluster_labels == cluster_id
    )[0]


    first_index = int(
        cluster_indices[0]
    )


    cluster_order.append(
        [
            first_index,
            cluster_id
        ]
    )


cluster_order = sorted(
    cluster_order,
    key=lambda value: value[0]
)


ordered_cluster_ids = [
    value[1]
    for value in cluster_order
]


ordered_cluster_centers = cluster_centers[
    ordered_cluster_ids
]


# ============================================================
# Set the first cluster center as X 0, Y 0, Z 0
# ============================================================

ordered_cluster_centers = np.subtract(
    ordered_cluster_centers,
    ordered_cluster_centers[0]
)


# ============================================================
# Export X, Y, Z only
# ============================================================

np.savetxt(
    output_file,
    ordered_cluster_centers,
    fmt="%.4f",
    delimiter="\t"
)


# ============================================================
# Print results
# ============================================================

print()
print("Processing completed")
print("CSV files used:", len(used_files))
print("CSV files skipped:", len(skipped_files))
print("Total valid GPS samples:", len(gps_df))
print("Number of clusters:", number_of_clusters)
print("Number of exported XYZ points:", len(ordered_cluster_centers))
print("Output file:", output_file)

print()
print("Cluster centers")
print(ordered_cluster_centers)


if len(skipped_files) > 0:

    print()
    print("Skipped CSV files")

    for skipped_file in skipped_files:
        print(skipped_file)