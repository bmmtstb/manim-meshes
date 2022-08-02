"""
define a few basic mesh-structures to be used as examples or test
"""
# python imports
import pathlib
from typing import List, Tuple
# third-party imports
import numpy as np
import trimesh
# local imports


from manim_meshes.models.mesh import Mesh


def create_triangle() -> Mesh:
    """
    create most basic triangle mesh
    """
    vertices = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    faces = [
        [0, 1, 2],
    ]
    return Mesh(
        verts=vertices,
        faces=faces,
    )


def create_pyramid(triangles_only: bool = True) -> Mesh:
    """
    create a basic 3D pyramid
    """
    vertices = [
        [1, 1, 0],  # 0
        [1, -1, 0],  # 1
        [-1, -1, 0],  # 2
        [-1, 1, 0],  # 3
        [0, 0, 2],  # 4 - tip
    ]
    if triangles_only:
        faces = [
            [0, 4, 1],
            [1, 4, 2],
            [2, 4, 3],
            [3, 4, 0],
            [0, 1, 2],
            [2, 3, 0],
        ]
        parts = [
            [0, 1, 2, 3, 4, 5],
        ]
    else:
        faces = [
            [0, 4, 1],
            [1, 4, 2],
            [2, 4, 3],
            [3, 4, 0],
            [0, 1, 2, 3],
        ]
        parts = [
            [0, 1, 2, 3, 4],
        ]
    return Mesh(
        verts=vertices,
        faces=faces,
        parts=parts,
    )


def create_cube(triangles_only: bool = True) -> Mesh:
    """
    create a equal sided cube
    """
    vertices = [
        [0, 0, 0],  # V0
        [1, 0, 0],  # V1
        [0, 1, 0],  # V2
        [0, 0, 1],  # V3
        [0, 1, 1],  # V4
        [1, 0, 1],  # V5
        [1, 1, 0],  # V6
        [1, 1, 1],  # V7
    ]
    if triangles_only:
        faces = [
            [0, 1, 2],  # F0
            [1, 6, 2],  # F1
            [0, 2, 3],  # F2
            [2, 4, 3],  # F3
            [0, 3, 1],  # F4
            [1, 3, 5],  # F5
            [1, 5, 6],  # F6
            [5, 7, 6],  # F7
            [2, 6, 4],  # F8
            [4, 6, 7],  # F9
            [3, 4, 5],  # F10
            [4, 7, 5],  # F11
        ]
        parts = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ]
    else:
        faces = [
            [0, 1, 6, 2],  # F0
            [0, 2, 4, 3],  # F1
            [0, 3, 5, 1],  # F2
            [1, 5, 7, 6],  # F3
            [2, 6, 7, 4],  # F4
            [3, 4, 7, 5],  # F5
        ]
        parts = [
            [0, 1, 2, 3, 4, 5],
        ]

    return Mesh(
        verts=vertices,
        faces=faces,
        parts=parts,
    )


def create_coplanar_triangles() -> Mesh:
    """
    create a basic 2D mesh
    """
    vertices = [
        [-1, 1],
        [1, 2],
        [1, 0],
        [0, 0],
        [-1, -1],
        [1, -1],
        [-1, 2],
    ]
    faces = [
        [0, 1, 2],
        [2, 3, 0],
        [3, 5, 4],
        [3, 2, 5],
        [4, 0, 3],
        [6, 1, 0]
    ]
    return Mesh(
        verts=vertices,
        faces=faces,
    )


def create_grid(areas: List[Tuple[float, float, int]]) -> Mesh:
    """
    given min, max and mount per direction create a mesh as a grid
    :param areas: list with one tuple per dimension, with every tuple consisting of
                 min, max and point-amount in this direction
    """
    dim = len(areas)
    vertices = np.array(
        np.meshgrid(*[np.linspace(area[0], area[1], int(area[2])) for area in areas], indexing='ij')
    ).T.reshape((-1, dim))

    if dim == 1:
        faces = None
        parts = None
    elif dim == 2:
        u, v = areas[0][2], areas[1][2]
        faces = [np.array([
                    i + j * u,  # bottom left
                    i + j * u + 1,  # bottom right
                    i + (j + 1) * u + 1,  # top right
                    i + (j + 1) * u  # top left
                ]) for j in range(v - 1) for i in range(u - 1)]
        parts = None
    elif dim == 3:
        # FIXME implement 3D
        # u, v, w = areas[0][2], areas[1][2], areas[2][2]
        # faces = None
        # parts = None
        raise NotImplementedError("3D not yet implemented")
    else:
        raise ValueError("Only 1D, 2D and 3D meshes implemented.")
    return Mesh(
        verts=vertices,
        faces=faces,
        parts=parts,
    )


def create_model(filepath: str = "", name: str = "") -> Mesh:
    """
    load a model from file in /data/models/
    """
    if len(name) == 0 and len(filepath) == 0:
        raise FileNotFoundError("Either provide a name or a filepath.")
    path_to_models = "data/models/"
    if name in ["armadillo", "suzanne"]:
        filepath = pathlib.Path(__file__).parent.parent.parent.joinpath(
            path_to_models, name + ".ply")
    elif name in ["Handle", "Land", "Octocat-v1", "squirrel", "tail_topper"]:
        filepath = pathlib.Path(__file__).parent.parent.parent.joinpath(
            path_to_models, name + ".stl")
    elif name != "":
        raise FileNotFoundError(f'{name} is not a valid object name.')
    tmesh = trimesh.load(filepath, force="mesh")
    return Mesh(tmesh.vertices, tmesh.faces)
