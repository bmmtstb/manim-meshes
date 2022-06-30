"""
tests for manim-meshes plugin
"""
# third-party imports
import numpy as np
# local imports
from manim_meshes.templates import create_triangle, create_model, create_pyramid


def test_basic_triangle():
    """test creation of one basic triangle mesh"""
    mesh = create_triangle()
    assert len(mesh.get_vertices()) == 3
    assert np.asarray(mesh.get_faces()).shape[0] == 1


def test_basic_pyramid():
    """test creation of one basic triangle mesh"""
    mesh = create_pyramid()
    assert len(mesh.get_vertices()) == 5
    assert np.asarray(mesh.get_faces()).shape[0] == 6


def test_armadillo():
    """test loading of predefined meshes"""
    mesh = create_model(name="armadillo")
    assert len(mesh.get_vertices()) != 0
