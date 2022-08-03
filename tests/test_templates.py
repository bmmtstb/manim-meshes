"""
tests for manim-meshes plugin
"""
# local imports
import numpy as np

from manim_meshes.templates import create_grid, create_triangle, create_model, create_pyramid


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


def test_create_grid():
    # 2D
    m_2d = create_grid([(-3, 3, 7), (-1, 1, 3)])
    assert len(m_2d.find_vertex(np.array([-3, -1]))) == 1
    assert len(m_2d.find_vertex(np.array([-3, 0]))) == 1
    assert len(m_2d.find_vertex(np.array([3, 1]))) == 1
    assert len(m_2d.get_vertices()) == 7 * 3
    assert any(np.array_equal(np.array([0, 1, 8, 7]), face) for face in m_2d.get_faces())
    assert any(np.array_equal(np.array([8, 9, 16, 15]), face) for face in m_2d.get_faces())
    assert len(m_2d.get_faces()) == (7 - 1) * (3 - 1)
    # 1D
    m_1d = create_grid([(0, 10, 11)])
    assert len(m_1d.find_vertex(np.array([1]))) == 1
    assert len(m_1d.find_vertex(np.array([7]))) == 1
    assert m_1d.get_faces() == []
    assert m_1d.get_parts() == []
    # 3D
    # m_3d = create_grid([(-1, 1, 3), (-2, 2, 5), (-1, 1, 3)])
    # assert m_3d.has_vertex(np.array([0, 0, 0]))
    # assert len(m_3d.get_vertices()) == 3 * 5 * 3
    # assert any(np.array_equal(np.array([0, 1, 4, 3]), face) for face in m_3d.get_faces())
    # assert any(np.array_equal(np.array([0, 1, 16, 15]), face) for face in m_3d.get_faces())
    # assert any(np.array_equal(np.array([0, 3, 18, 15]), face) for face in m_3d.get_faces())
