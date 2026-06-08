# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:31:27 2026

@author: Desktop
"""

import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull


def read_ply_points(ply_path):
    """
    Read xyz and optional rgb from a PLY file using plyfile.
    """
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]

    x = np.asarray(vertex["x"], dtype=np.float64)
    y = np.asarray(vertex["y"], dtype=np.float64)
    z = np.asarray(vertex["z"], dtype=np.float64)

    points = np.column_stack((x, y, z))

    has_rgb = all(name in vertex.data.dtype.names for name in ["red", "green", "blue"])

    if has_rgb:
        rgb = np.column_stack((
            np.asarray(vertex["red"], dtype=np.uint8),
            np.asarray(vertex["green"], dtype=np.uint8),
            np.asarray(vertex["blue"], dtype=np.uint8)
        ))
    else:
        rgb = None

    return points, rgb


def fit_plane_pca(points):
    """
    Fit a plane to 3D points using PCA.

    Plane form:
        n_x x + n_y y + n_z z + d = 0

    Returns:
        centroid, normal, d
    """
    centroid = np.mean(points, axis=0)

    centered = points - centroid

    covariance = centered.T @ centered / points.shape[0]

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    normal = eigenvectors[:, np.argmin(eigenvalues)]
    normal = normal / np.linalg.norm(normal)

    d = -np.dot(normal, centroid)

    return centroid, normal, d


def create_plane_basis(normal):
    """
    Create two orthonormal basis vectors lying on the plane.
    """
    normal = normal / np.linalg.norm(normal)

    temp = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(temp, normal)) > 0.9:
        temp = np.array([0.0, 1.0, 0.0])

    u = temp - np.dot(temp, normal) * normal
    u = u / np.linalg.norm(u)

    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    return u, v


def project_points_to_plane(points, centroid, normal):
    """
    Project 3D points onto fitted plane.
    """
    signed_distances = (points - centroid) @ normal
    projected_points = points - signed_distances[:, None] * normal

    return projected_points, signed_distances


def make_2d_polygon_facet(projected_points, centroid, u, v):
    """
    Convert projected 3D points to 2D plane coordinates and compute a convex polygon.
    This approximates the 2D polygon facet boundary.
    """
    local = projected_points - centroid

    plane_x = local @ u
    plane_y = local @ v

    points_2d = np.column_stack((plane_x, plane_y))

    hull = ConvexHull(points_2d)

    polygon_2d = points_2d[hull.vertices]

    polygon_3d = (
        centroid
        + polygon_2d[:, 0:1] * u
        + polygon_2d[:, 1:2] * v
    )

    return points_2d, polygon_2d, polygon_3d


def write_ply_with_distances(output_path, points, rgb, signed_distances):
    """
    Save point cloud with distance scalar fields.
    """
    abs_distances = np.abs(signed_distances)

    if rgb is not None:
        vertex_data = np.empty(
            points.shape[0],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("signed_distance", "f4"),
                ("abs_distance", "f4")
            ]
        )

        vertex_data["x"] = points[:, 0]
        vertex_data["y"] = points[:, 1]
        vertex_data["z"] = points[:, 2]
        vertex_data["red"] = rgb[:, 0]
        vertex_data["green"] = rgb[:, 1]
        vertex_data["blue"] = rgb[:, 2]
        vertex_data["signed_distance"] = signed_distances
        vertex_data["abs_distance"] = abs_distances

    else:
        vertex_data = np.empty(
            points.shape[0],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("signed_distance", "f4"),
                ("abs_distance", "f4")
            ]
        )

        vertex_data["x"] = points[:, 0]
        vertex_data["y"] = points[:, 1]
        vertex_data["z"] = points[:, 2]
        vertex_data["signed_distance"] = signed_distances
        vertex_data["abs_distance"] = abs_distances

    element = PlyElement.describe(vertex_data, "vertex")
    PlyData([element], text=True).write(output_path)


def plot_distance_map(points, signed_distances, title="Point to fitted facet distance"):
    """
    Plot top view colored by signed distance.
    """
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=signed_distances,
        s=1
    )
    plt.colorbar(sc, label="Signed distance")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.axis("equal")
    plt.show()


def plot_polygon_facet(points_2d, polygon_2d):
    """
    Plot the 2D polygon boundary of the fitted facet.
    """
    closed_polygon = np.vstack([polygon_2d, polygon_2d[0]])

    plt.figure(figsize=(7, 7))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1)
    plt.plot(closed_polygon[:, 0], closed_polygon[:, 1], linewidth=2)
    plt.xlabel("Facet local X")
    plt.ylabel("Facet local Y")
    plt.title("2D polygon facet boundary")
    plt.axis("equal")
    plt.show()


def main():
    input_ply = r"D:\Ryan\GitHub\paper\road_defect_detection\images\reconstruction_global\scene_dense.ply"
    output_ply = r"D:\Ryan\GitHub\paper\road_defect_detection\images\reconstruction_global\point_cloud_with_facet_distances.ply"

    points, rgb = read_ply_points(input_ply)

    print("Loaded points:", points.shape[0])

    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    if rgb is not None:
        rgb = rgb[valid]

    centroid, normal, d = fit_plane_pca(points)

    print("Fitted plane:")
    print("normal =", normal)
    print("d =", d)
    print("plane equation:")
    print(f"{normal[0]:.6f} x + {normal[1]:.6f} y + {normal[2]:.6f} z + {d:.6f} = 0")

    projected_points, signed_distances = project_points_to_plane(points, centroid, normal)

    u, v = create_plane_basis(normal)

    points_2d, polygon_2d, polygon_3d = make_2d_polygon_facet(
        projected_points,
        centroid,
        u,
        v
    )

    abs_distances = np.abs(signed_distances)

    print("\nDistance statistics:")
    print("Mean signed distance:", np.mean(signed_distances))
    print("Mean absolute distance:", np.mean(abs_distances))
    print("Max absolute distance:", np.max(abs_distances))
    print("Min signed distance:", np.min(signed_distances))
    print("Max signed distance:", np.max(signed_distances))
    print("Standard deviation:", np.std(signed_distances))

    write_ply_with_distances(output_ply, points, rgb, signed_distances)

    print("\nSaved output:")
    print(output_ply)

    plot_distance_map(points, signed_distances)
    plot_polygon_facet(points_2d, polygon_2d)


if __name__ == "__main__":
    main()