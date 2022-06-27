"""
tests for manim-meshes plugin
"""
# third-party imports
import numpy as np
# local imports
from manim_meshes.templates import create_triangle, create_model, create_pyramid


def test_basic_triangle():
    """test creation of one basic triangle mesh"""
    tri_mesh = create_triangle()
    assert len(tri_mesh.vertices) == 3
    assert np.asarray(tri_mesh.triangles).shape[0] == 1
    assert not tri_mesh.is_volume
    assert not tri_mesh.is_watertight


def test_basic_pyramid():
    """test creation of one basic triangle mesh"""
    mesh = create_pyramid()
    assert len(mesh.vertices) == 5
    assert np.asarray(mesh.triangles).shape[0] == 6
    assert mesh.is_volume
    assert mesh.is_watertight


def test_armadillo():
    """test loading of predefined meshes"""
    mesh = create_model(name="armadillo")
    assert not mesh.is_empty
    assert mesh.is_volume
    assert mesh.body_count == 1
    assert mesh.is_watertight
