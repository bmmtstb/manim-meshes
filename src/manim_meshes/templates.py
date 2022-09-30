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
from manim_meshes.models.data_models.mesh import Mesh


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
        vertices=vertices,
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
        vertices=vertices,
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
        vertices=vertices,
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
        vertices=vertices,
        faces=faces,
    )


def create_coplanar_points() -> Mesh:
    """
    create a basic 2D mesh without faces ~> exemplary 2D point cloud / set
    """
    vertices = np.array([[+1.77291731, -2.42097974],
                         [+1.72006063,  2.77386102],
                         [-2.47248328, -1.53451374],
                         [+0.32320519,  2.34811395],
                         [-0.83498357,  0.34056556],
                         [+0.43649618,  0.79539595],
                         [-0.24413860, -2.04571182],
                         [+2.83770675,  2.33446802],
                         [+1.27516666, -0.08275843],
                         [+1.94458474, -2.79598506]]
                        )
    return Mesh(
        vertices=vertices,
        faces=[]
    )


def create_grid(areas: List[Tuple[float, float, int]]) -> Mesh:
    """
    given min, max and amount per direction create a mesh as a grid
    currently works for 1D and 2D
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
    # elif dim == 3:
    #     # u, v, w = areas[0][2], areas[1][2], areas[2][2]
    #     # faces = [... for k in range(w - 1) for j in range(v - 1) for i in range(u - 1)]
    #     # parts = None
    #     # TODO fully implement 3D grid
    #     raise NotImplementedError("3D grid generation is not yet implemented")
    else:
        raise NotImplementedError("Only 1D, 2D meshes implemented.")
    return Mesh(
        vertices=vertices,
        faces=faces,
        parts=parts,
    )


def create_model(filepath: str = "", name: str = "") -> Mesh:
    """
    load a model from file in /data/models/
    be careful, the larger meshes can not be displayed with the BasicMesh class (>32GB RAM, >30 min)
    """
    if len(name) == 0 and len(filepath) == 0:
        raise FileNotFoundError("Either provide a name or a filepath.")
    path_to_models = "data/models/"
    # load the ply files
    if name in ["armadillo", "suzanne"]:
        filepath = pathlib.Path(__file__).parent.joinpath(
            path_to_models, name + ".ply")
    # load the stl files
    elif name in ["tail_topper"]:
        filepath = pathlib.Path(__file__).parent.joinpath(
            path_to_models, name + ".stl")
    elif name != "":
        raise FileNotFoundError(f'{name} is not a valid object name.')
    # use trimesh to load the files, no parts?
    tmesh = trimesh.load(filepath, force="mesh")
    return Mesh(vertices=tmesh.vertices, faces=tmesh.faces)
