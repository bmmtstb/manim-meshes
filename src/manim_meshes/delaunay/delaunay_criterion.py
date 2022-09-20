"""
functions to check delaunay criterion
"""
# python imports

# third-party imports
import numpy as np
# local imports
from manim_meshes.models.data_models.mesh import Mesh


def get_triangle_circum_circle_params(
        pt1: np.ndarray,
        pt2: np.ndarray,
        pt3: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Given the three corner-points of a triangle, calculate the parameters of the circum-circle
    :param: three corner-points of one triangle
    :returns: center point, radius
    """
    div = 2 * np.linalg.norm(np.cross(pt1 - pt2, pt2 - pt3)) ** 2
    alpha = np.linalg.norm(pt2 - pt3) ** 2 * (pt1 - pt2).dot(pt1 - pt3) / div
    beta = np.linalg.norm(pt1 - pt3) ** 2 * (pt2 - pt1).dot(pt2 - pt3) / div
    gamma = np.linalg.norm(pt1 - pt2) ** 2 * (pt3 - pt1).dot(pt3 - pt2) / div
    center = alpha * pt1 + beta * pt2 + gamma * pt3
    div = 2 * np.linalg.norm(np.cross(pt1 - pt2, pt2 - pt3))
    radius = np.linalg.norm(pt1 - pt2) * np.linalg.norm(pt2 - pt3) * np.linalg.norm(pt3 - pt1) / div
    return center, radius


def get_point_indices_violating_delaunay(mesh: Mesh, face_id: int):
    """given a triangle by id, get all indices of points violating delaunay criterion"""
    indices = []
    face = mesh.faces[face_id]
    center, radius = get_triangle_circum_circle_params(*[mesh.get_3d_vertices()[i] for i in face])

    # TODO: [improve to be faster] don't loop all vertices, only loop ones that are "close"
    for idx, point in enumerate(mesh.get_3d_vertices()):
        if idx not in face:
            distance = np.linalg.norm(center - point)
            if distance < radius:  # inside circle
                indices.append(idx)
    return indices


def is_point_violating_delaunay(mesh: Mesh, vertex_idx: int, face_idx):
    """
        returns True if the vertex with index vertex_idx is violating the delaunay criterion
        w.r.t. the provided face (face_idx).
    """
    point = mesh.get_3d_vertices()[vertex_idx]
    face = mesh.faces[face_idx]
    center, radius = get_triangle_circum_circle_params(*[mesh.get_3d_vertices()[i] for i in face])
    distance = np.linalg.norm(center - point)
    return distance < radius
