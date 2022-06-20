"""
define a few basic mesh-structures to be used as examples or test
"""
# python imports
import pathlib
# third-party imports
import trimesh


def create_triangle() -> trimesh.Trimesh:
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
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
    )


def create_pyramid() -> trimesh.Trimesh:
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
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
    )


def create_coplanar_triangles() -> trimesh.Trimesh:
    """
    create a basic 3D pyramid
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
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
    )


def create_model(filepath: str = "", name: str = "") -> trimesh.Trimesh:
    """
    load a model from file in /data/models/
    """
    if len(name) == 0 and len(filepath) == 0:
        raise FileNotFoundError("Either provide a name or a filepath.")
    if name in ["armadillo", "suzanne"]:
        path_to_models = "data/models/"
        filepath = pathlib.Path(__file__).parent.parent.parent.joinpath(
            path_to_models, name + ".ply")
    elif name in ["Handle", "Land", "Octocat-v1", "squirrel", "tail_topper"]:
        path_to_models = "data/models/"
        filepath = pathlib.Path(__file__).parent.parent.parent.joinpath(
            path_to_models, name + ".stl")
    elif name != "":
        raise FileNotFoundError(f'{name} is not a valid object name.')
    return trimesh.load(filepath, force="mesh")
