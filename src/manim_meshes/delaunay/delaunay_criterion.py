"""
functions to check delaunay criterion
"""
# python imports
from typing import List
import numpy as np
# third-party imports
import manim as m
# local imports

from manim_meshes.models.manim_models.triangle_mesh import TriangleManim2DMesh


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


def get_circum_circle(triangle_mesh: TriangleManim2DMesh, face_idx: int, **kwargs) -> m.Circle:
    """ create a circum-circle around face with given idx, returns manim Circle object

        :param triangle_mesh: the triangle 2D mesh
        :param face_idx: index of face to create circle around
        :param kwargs: key arguments passed to Circle, e.g. stroke_width
    """
    face = triangle_mesh.mesh.faces[face_idx]
    vertices = [triangle_mesh.mesh.get_3d_vertices()[i] for i in face]
    center, radius = get_triangle_circum_circle_params(*vertices)
    if 'stroke_width' not in kwargs:
        kwargs['stroke_width'] = 2
    circ = m.Circle(radius, **kwargs)
    circ.shift(center)
    return circ


def get_point_indices_violating_delaunay(triangle_mesh: TriangleManim2DMesh, face_id: int) -> List[int]:
    """given a triangle by id, get all indices of points violating delaunay criterion"""
    indices: List[int] = []
    face = triangle_mesh.mesh.faces[face_id]
    center, radius = get_triangle_circum_circle_params(*[triangle_mesh.mesh.get_3d_vertices()[i] for i in face])

    # TODO: [improve to be faster] don't loop all vertices, only loop ones that are "close", how?
    #  should be possible to do using numpy functions, should be faster and more readable
    for idx, point in enumerate(triangle_mesh.mesh.get_3d_vertices()):
        if idx not in face:
            distance = np.linalg.norm(center - point)
            if distance < radius:  # inside circle
                indices.append(idx)
    return indices


def is_point_violating_delaunay(triangle_mesh: TriangleManim2DMesh, vertex_idx: int, face_idx) -> bool:
    """returns whether the vertex with index vertex_idx is violating the delaunay criterion w.r.t. the provided face"""
    point = triangle_mesh.mesh.get_3d_vertices()[vertex_idx]
    face = triangle_mesh.mesh.faces[face_idx]
    center, radius = get_triangle_circum_circle_params(*[triangle_mesh.mesh.get_3d_vertices()[i] for i in face])
    distance = np.linalg.norm(center - point)
    return distance < radius
