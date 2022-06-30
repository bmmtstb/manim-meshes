"""
define a few basic mesh-structures to be used as examples or test
"""
# python imports
import pathlib
# third-party imports
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


def create_pyramid() -> Mesh:
    """
    create a basic 3D pyramid
    """
    vertices = [
        [1, 1, 0],  # 0
        [1, -1, 0],   # 1
        [-1, -1, 0],  # 2
        [-1, 1, 0],  # 3
        [0, 0, 2],  # 4 - tip
    ]
    faces = [
        [0, 4, 1],
        [1, 4, 2],
        [2, 4, 3],
        [3, 4, 0],
        [0, 1, 2],
        [2, 3, 0],
    ]
    return Mesh(
        verts=vertices,
        faces=faces,
    )


def create_coplanar_triangles() -> Mesh:
    """
    create a basic 2D mesh
    """
    vertices = [
        [-1, 1, 0],
        [1, 2, 0],
        [1, 0, 0],
        [0, 0, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 2, 0]
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
