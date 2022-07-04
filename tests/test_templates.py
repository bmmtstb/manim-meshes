"""
tests for manim-meshes plugin
"""
# local imports
from manim_meshes.templates import create_triangle, create_model, create_pyramid


def test_basic_triangle():
    """test creation of one basic triangle mesh"""
    mesh = create_triangle()
    assert len(mesh.get_vertices()) == 3  # there are 3 verts in total
    assert len(mesh.get_faces()) == 1  # there is one face
    assert mesh.get_faces()[0].shape[0] == 3  # face has 3 verts


def test_basic_pyramid():
    """test creation of one basic triangle mesh"""
    mesh = create_pyramid(triangles_only=True)
    assert len(mesh.get_vertices()) == 5
    assert len(mesh.get_faces()) == 6
    assert all(len(f) == 3 for f in mesh.get_faces())
    # mixes meshes
    mesh = create_pyramid(triangles_only=False)
    assert len(mesh.get_vertices()) == 5
    assert len(mesh.get_faces()) == 5
    assert any(len(f) == 4 for f in mesh.get_faces())


def test_armadillo():
    """test loading of predefined meshes"""
    mesh = create_model(name="tail_topper")
    assert len(mesh.get_vertices()) != 0
