"""
test mesh functionalities
"""
# python imports
from copy import deepcopy
# third-party imports
import numpy as np
import pytest
# local imports
from manim_meshes.helpers import are_edges_equal
from manim_meshes.models.mesh import Mesh
from manim_meshes.templates import create_pyramid
from manim_meshes.exceptions import InvalidMeshException


def test_mesh_equality_plus_mesh_creation():
    """sanity check for equality plus some sanity checks for creating meshes"""
    # __eq__ and __ne__ (==, !=)
    tm1 = create_pyramid(triangles_only=True)
    tm2 = create_pyramid(triangles_only=True)
    tm3 = create_pyramid(triangles_only=False)
    assert tm1 == tm2
    assert tm1 != tm3
    assert tm2 != tm3

    tm4 = Mesh(verts=np.arange(21).reshape((7, 3)), faces=None, parts=None)
    tm5 = Mesh(verts=np.arange(21).reshape((7, 3)), faces=[], parts=[])
    tm6 = Mesh(verts=np.arange(21).reshape((7, 3)), faces=[], parts=None)
    tm7 = Mesh(verts=np.arange(21).reshape((7, 3)), faces=[[0, 1, 2], [1, 2, 3], [3, 4, 5]], parts=None)
    # parts reference the same vertices, even if faces and or parts are shuffled
    tm8 = Mesh(verts=np.arange(21).reshape((7, 3)), faces=[[0, 1, 2], [1, 2, 3], [3, 4, 5]], parts=[[0, 1, 2]])
    tm9 = Mesh(verts=np.arange(21).reshape((7, 3)), faces=[[0, 1, 2], [1, 2, 3], [3, 4, 5]], parts=[[1, 2, 0]])
    tm10 = Mesh(verts=np.arange(21).reshape((7, 3)), faces=[[1, 2, 3], [3, 4, 5], [0, 1, 2]], parts=[[1, 2, 0]])
    tm11 = Mesh(verts=np.arange(21).reshape((7, 3)), faces=[[1, 2, 3], [3, 4, 5], [0, 1, 2]], parts=[[0, 1, 2]])
    tm12 = Mesh(verts=np.arange(21).reshape((7, 3)), faces=[[1, 2, 3], [0, 1, 2], [3, 4, 5]], parts=[[0, 1, 2]])
    # face reference the same vertices even if they are mixed up
    tm13 = Mesh(
        verts=np.array([[6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20], [0, 1, 2], [3, 4, 5]]),
        faces=[[5, 6, 0], [6, 0, 1], [1, 2, 3]],
        parts=[[0, 1, 2]])
    tm14 = Mesh(
        verts=np.array([[9, 10, 11], [3, 4, 5], [12, 13, 14], [15, 16, 17], [6, 7, 8], [18, 19, 20], [0, 1, 2]]),
        faces=[[6, 1, 4], [1, 4, 0], [0, 2, 3]],
        parts=[[2, 0, 1]])
    tm15 = Mesh(
        verts=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20]]),
        faces=[[0, 1, 3], [1, 2, 3], [3, 4, 5]],  # 0,1,3 instead of 0,1,2
        parts=[[0, 1, 2]])
    # "not in" is the same as "!=" for all three
    assert tm4 == tm5 == tm6
    assert tm7 not in (tm4, tm5, tm6)
    assert tm8 not in (tm4, tm5, tm6)
    assert tm7 != tm8
    assert tm8 == tm9 and tm8 == tm10 and tm8 == tm11 and tm9 == tm10 and tm9 == tm11 and tm10 == tm11
    assert tm12 not in (tm8, tm9, tm10, tm11)
    assert tm8 == tm13 and tm8 == tm14 and tm13 == tm14
    assert tm15 not in (tm8, tm13, tm14)


def test_find_vertex():
    m = Mesh(verts=np.array([[1, 2, 3], [1, 2, 3], [1, 0, 1], [1, 2, 3]]), faces=None)
    assert m.find_vertex(np.array([1, 0, 1])) == [2]
    assert m.find_vertex(np.array([1, 2, 3])) == [0, 1, 3]
    assert m.find_vertex(np.array([1, 2, 3]), start=1) == [1, 3]
    assert m.find_vertex(np.array([0, 0, 0])) == []
    assert m.find_vertex(np.array([1, 0])) == []
    assert m.find_vertex(np.array([1, 0, 1, 0])) == []
    assert m.find_vertex(np.array([1, 0, 1.000001])) == []


def test_find_face():
    m = Mesh(
        verts=np.array([[1, 2, 3], [1, 2, 3], [1, 0, 1], [1, 2, 3]]),
        faces=np.array([[1, 2, 3], [2, 3, 1], [1, 3, 2], [1, 2, 0], [3, 1, 2], [3, 1, 2, 3]])
    )
    assert m.find_face(np.array([1, 2, 3])) == [0, 1, 4]
    assert m.find_face(np.array([3, 1, 2]), start=2) == [4]
    assert m.find_face(np.array([3, 1, 2, 3])) == [5]
    assert m.find_face(np.array([1, 2])) == []


def test_dangling_vertices():
    # node 4 is not used -> has dangling vertices
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=[[0, 1, 2]]
    )
    assert m.dangling_vert_check()

    # with second triangle - all nodes are used
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=[[0, 1, 2], [2, 3, 1]]
    )
    assert not m.dangling_vert_check()

    # test with None
    m = Mesh(
        verts=np.array([[0, 0, 0]]),
        faces=None,
        parts=None
    )
    assert m.dangling_vert_check()

    # test with []
    m = Mesh(
        verts=np.array([[0, 0, 0]]),
        faces=[],
        parts=None
    )
    assert m.dangling_vert_check()
    m = Mesh(
        verts=np.array([[0, 0, 0]]),
        faces=[],
        parts=[]
    )
    assert m.dangling_vert_check()


def test_dangling_faces():
    m = create_pyramid()
    assert not m.dangling_face_check()

    m.add_faces([np.array([1, 2, 3])])
    assert m.dangling_face_check()

    # test with None
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=[[0, 1, 2], [2, 3, 1], [1, 2, 3]],
        parts=None
    )
    assert m.dangling_face_check()
    # test with empty values []
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=[[0, 1, 2], [2, 3, 1]],
        parts=[]
    )
    assert m.dangling_face_check()


def test_different_face_sizes():
    """triangles and squares as faces!"""
    m = create_pyramid(triangles_only=False)
    assert len(m.get_faces()) == 5
    assert not m.dangling_vert_check()
    assert not m.dangling_face_check()


def test_faulty_dims_on_vertices():
    """2D and 3D vertices should not work"""
    with pytest.raises(InvalidMeshException) as _:
        Mesh(
            verts=[[1, 0], [0, 1],
                   [0, 0, 1],  # third value has wrong dimension
                   ],
            faces=[[0, 1, 2]],
        )
    with pytest.raises(InvalidMeshException) as _:
        Mesh(
            verts=[[1, 0], [0, 1],
                   [0, ],  # missing second value
                   ],
            faces=[[0, 1, 2]],
        )
    with pytest.raises(InvalidMeshException) as _:
        Mesh(
            verts=[[1, 0], [0, 1], [1, 1],
                   [0],  # missing second value and vertex unused
                   ],
            faces=[[0, 1, 2]],
        )


def test_faulty_verts_type_vertices():
    """2D and 3D vertices should not work"""
    # set in second layer
    with pytest.raises(InvalidMeshException) as _:
        Mesh(
            verts=[[1, 0], {0, 1}, [0, 0]],
            faces=[[0, 1, 2]],
        )
    # dict
    with pytest.raises(InvalidMeshException) as _:
        Mesh(
            verts=[[1, 0], {0: 1}, [0, 0]],
            faces=[[0, 1, 2]],
        )


def test_different_dims_on_faces():
    """2D and 3D vertices should not work"""
    try:
        Mesh(
            verts=[[1, 0], [0, 1], [0, 0], [1, 1]],
            faces=[[0, 1, 2], [0, 1, 2, 3]],
        )
    except InvalidMeshException as e:
        pytest.fail("Different sized faces should work." + str(e))


def test_remove_parts():
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 0]],
        parts=[[0, 1, 2]]
    )
    # mesh stays the same except for parts
    m1 = deepcopy(m)
    m1.remove_parts([0])
    assert Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 0]],
        parts=None
    ) == m1
    # removing non existent part raises an exception
    with pytest.raises(IndexError) as _:
        m2 = deepcopy(m)
        m2.remove_parts([1])
    with pytest.raises(IndexError) as _:
        m2_1 = deepcopy(m)
        m2_1.remove_parts([-1])
    # order matters and no multi indices
    m3 = deepcopy(m)
    m4 = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 0]],
        parts=[[0, 1, 2], [1, 1, 1], [2, 2, 2], [1, 2, 1], [2, 1, 1]]
    )
    m4.remove_parts([1, 3, 2, 1, 4])
    assert m3 == m4
    assert are_edges_equal(m3.get_edges(), m4.get_edges())


def test_remove_faces():
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 4], [1, 3, 4]],
        parts=[[0, 1, 2], [1, 1, 1]]
    )
    # removing invalid index results in error
    with pytest.raises(IndexError) as _:
        m1 = deepcopy(m)
        m1.remove_faces([4])
    with pytest.raises(IndexError) as _:
        m1 = deepcopy(m)
        m1.remove_faces([-1])
    # removing one dangling face does not change parts but changes edges
    m2 = deepcopy(m)
    m2.remove_faces([3])
    assert len(m2.get_vertices()) == 4
    assert len(m2.get_faces()) == 3
    assert len(m2.get_parts()) == 2
    assert not m2.dangling_face_check()
    assert are_edges_equal(m2.get_edges(), [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (3, 4), (2, 4)])
    # removing one non-dangling face removes the part that uses it and the respective edges
    m3 = deepcopy(m)
    m3.remove_faces([2])
    assert len(m3.get_vertices()) == 4
    assert len(m3.get_faces()) == 3
    assert len(m3.get_parts()) == 1
    assert m3.dangling_face_check()
    assert are_edges_equal(m3.get_edges(), [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (3, 4), (1, 4)])
    # removing all faces works and order does not matter
    m4 = deepcopy(m)
    m4.remove_faces([0, 3, 1, 2])
    assert len(m4.get_vertices()) == 4
    assert len(m4.get_faces()) == 0
    assert len(m4.get_parts()) == 0
    assert are_edges_equal(m4.get_edges(), [])
    # removing one face may result in multiple part deletions
    m5 = deepcopy(m)
    m5.remove_faces([1])
    assert len(m5.get_vertices()) == 4
    assert len(m5.get_faces()) == 3
    assert len(m5.get_parts()) == 0
    assert are_edges_equal(m5.get_edges(), [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (3, 4), (2, 4), (1, 4)])


def test_remove_vertices():
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 2, 2], [3, 3, 3]]),
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 1], [1, 3, 4]],
        parts=[[1, 1, 1], [0, 1, 2]]
    )
    # throws exception if index out of range
    with pytest.raises(IndexError) as _:
        m1 = deepcopy(m)
        m1.remove_vertices([6])
    with pytest.raises(IndexError) as _:
        m1 = deepcopy(m)
        m1.remove_vertices([-1])
    # removing unused vertex does not change mesh but changes edges
    m2 = deepcopy(m)
    m2.remove_vertices([5])
    assert len(m2.get_vertices()) == 5
    assert len(m2.get_faces()) == 4
    assert len(m2.get_parts()) == 2
    assert are_edges_equal(m2.get_edges(), [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (3, 4), (1, 4)])
    # removing faces but not independent parts
    m3 = deepcopy(m)
    m3.remove_vertices([4])
    assert len(m3.get_vertices()) == 5
    assert len(m3.get_faces()) == 3
    assert len(m3.get_parts()) == 2
    assert not m3.dangling_face_check()
    assert are_edges_equal(m3.get_edges(), [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)])
    # removing multiple vertices works and order does not matter
    m4 = deepcopy(m)
    m4.remove_vertices([0, 3, 1, 2, 5])  # not 4
    assert len(m4.get_vertices()) == 1
    assert len(m4.get_faces()) == 0
    assert len(m4.get_parts()) == 0
    assert are_edges_equal(m4.get_edges(), [])
    # removing vertex results in removing faces and parts only if necessary
    # index shift for edges!
    m5 = deepcopy(m)
    m5.remove_vertices([0])
    assert len(m5.get_vertices()) == 5
    assert len(m5.get_faces()) == 3
    assert len(m5.get_parts()) == 1
    assert are_edges_equal(m5.get_edges(), [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)])


def test_add_parts():
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 4], [1, 3, 4]],
        parts=None
    )
    # invalid object raises an exception
    with pytest.raises(InvalidMeshException) as _:
        m1 = deepcopy(m)
        # noinspection PyTypeChecker
        m1.add_parts([1, 2, 3])
    # index out of range raises an IndexError
    with pytest.raises(IndexError) as _:
        m2 = deepcopy(m)
        m2.add_parts([np.array([2, 3, 4])])
    with pytest.raises(IndexError) as _:
        m2 = deepcopy(m)
        m2.add_parts([np.array([0, 1, -1])])
    # add single to non-existing
    m3 = deepcopy(m)
    m3.add_parts([np.array([0, 1, 2])])
    assert len(m3.get_parts()) == 1
    # add single to existing - and size can be different
    m3.add_parts([np.array([1, 2, 3, 3])])
    assert len(m3.get_parts()) == 2
    # add multiple to non-existing
    m4 = deepcopy(m)
    m4.add_parts([np.array([0, 1, 2]), np.array([1, 2, 3])])
    assert len(m4.get_parts()) == 2
    # add multiple to existing
    m4.add_parts([np.array([1, 1, 1]), np.array([2, 2, 2, 2])])
    assert len(m4.get_parts()) == 4


def test_add_faces():
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=None,
        parts=None
    )
    # invalid object raises an exception
    with pytest.raises(InvalidMeshException) as _:
        m1 = deepcopy(m)
        # noinspection PyTypeChecker
        m1.add_faces([1, 2, 3])
    # index out of range raises an IndexError
    with pytest.raises(IndexError) as _:
        m2 = deepcopy(m)
        m2.add_faces([np.array([2, 3, 4])])
    with pytest.raises(IndexError) as _:
        m2 = deepcopy(m)
        m2.add_faces([np.array([0, 1, -1])])
    # add single to non-existing
    m3 = deepcopy(m)
    m3.add_faces([np.array([0, 1, 2])])
    assert len(m3.get_faces()) == 1
    assert are_edges_equal(m3.get_edges(), [(0, 1), (1, 2), (0, 2)])
    # add single to existing - and size can be different
    m3.add_faces([np.array([1, 2, 3, 3])])
    assert len(m3.get_faces()) == 2
    assert are_edges_equal(m3.get_edges(), [(0, 1), (1, 2), (0, 2), (2, 3), (3, 3), (1, 3)])
    # add multiple to non-existing
    m4 = deepcopy(m)
    m4.add_faces([np.array([0, 1, 2]), np.array([1, 2, 3])])
    assert len(m4.get_faces()) == 2
    assert are_edges_equal(m4.get_edges(), [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3)])
    # add multiple to existing
    m4.add_faces([np.array([1, 1, 1]), np.array([2, 2, 2, 2])])
    assert len(m4.get_faces()) == 4
    assert are_edges_equal(m4.get_edges(), [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (2, 2), (1, 1)])


def test_add_vertices():
    m = Mesh(
        verts=np.array([[0, 0, 0]]),
        faces=None,
        parts=None
    )
    # invalid object raises an exception
    with pytest.raises(InvalidMeshException) as _:
        m1_wrong_nested = deepcopy(m)
        # noinspection PyTypeChecker
        m1_wrong_nested.add_vertices(np.array([1, 2, 3]))
    with pytest.raises(InvalidMeshException) as _:
        m1_wrong_type = deepcopy(m)
        # noinspection PyTypeChecker
        m1_wrong_type.add_vertices([[1, 2, 3]])
    # add with wrong dimensions raises an exception
    with pytest.raises(InvalidMeshException) as _:
        m2 = deepcopy(m)
        m2.add_vertices(np.array([[2, 3, 4, 5]]))
    # add single - add same point
    m3 = deepcopy(m)
    m3.add_vertices(np.array([[0, 0, 0]]))
    assert len(m3.get_vertices()) == 2
    assert are_edges_equal(m3.get_edges(), m.get_edges())
    # add multiple
    m4 = deepcopy(m)
    m4.add_vertices(np.array([[0, 0, 0], [1, 1, 1], [1, 2, 3]]))
    assert len(m4.get_vertices()) == 4
    assert are_edges_equal(m4.get_edges(), m.get_edges())


def test_update_part():
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 2, 2], [3, 3, 3]]),
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 1], [1, 3, 4]],
        parts=[[1, 1, 1], [0, 1, 2]]
    )
    m_wo = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 2, 2], [3, 3, 3]]),
        faces=None,
        parts=None
    )
    # invalid part object raises an exception
    with pytest.raises(TypeError) as _:
        m1_wrong_nested = deepcopy(m)
        # noinspection PyTypeChecker
        m1_wrong_nested.update_part(0, np.array([[1, 2, 3]]))
    # invalid part index raises exception
    with pytest.raises(IndexError) as _:
        m2_w = deepcopy(m)
        m2_w.update_part(2, [1, 2, 3])
    with pytest.raises(IndexError) as _:
        m2_w = deepcopy(m)
        m2_w.update_part(-1, [1, 2, 3])
    with pytest.raises(IndexError) as _:
        m2_wo = deepcopy(m_wo)
        m2_wo.update_part(0, [1, 2, 3])
    with pytest.raises(IndexError) as _:
        m2_wo = deepcopy(m_wo)
        m2_wo.update_part(-1, [1, 2, 3])
    # invalid face index raises exception
    with pytest.raises(IndexError) as _:
        m3 = deepcopy(m)
        m3.update_part(1, [4, 5, 6])
    with pytest.raises(IndexError) as _:
        m3 = deepcopy(m)
        m3.update_part(1, [-1, 0, 1])
    # change existing part conform
    m4_1 = deepcopy(m)  # list of int
    m4_1.update_part(0, [3, 2, 1])
    assert not m4_1.dangling_face_check()
    assert len(m4_1.get_parts()) == 2
    assert m4_1.get_parts()[0][0] == 3
    m4_2 = deepcopy(m)  # tuple of int
    m4_2.update_part(0, (3, 2, 1))
    assert not m4_2.dangling_face_check()
    assert len(m4_2.get_parts()) == 2
    assert m4_2.get_parts()[0][0] == 3
    m4_3 = deepcopy(m)  # 1D - np.ndarray
    m4_3.update_part(0, np.array([3, 2, 1]))
    assert not m4_3.dangling_face_check()
    assert len(m4_3.get_parts()) == 2
    assert m4_3.get_parts()[0][0] == 3
    assert are_edges_equal(m4_3.get_edges(), m.get_edges())


def test_update_face():
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 2, 2], [3, 3, 3]]),
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 1], [1, 3, 4]],
        parts=[[1, 1, 1], [0, 1, 2]]
    )
    m_wo = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 2, 2], [3, 3, 3]]),
        faces=None,
        parts=None
    )
    # invalid face object raises an exception
    with pytest.raises(TypeError) as _:
        m1_wrong_nested = deepcopy(m)
        # noinspection PyTypeChecker
        m1_wrong_nested.update_face(0, np.array([[1, 2, 3]]))
    # invalid face index raises exception
    with pytest.raises(IndexError) as _:
        m2_w = deepcopy(m)
        m2_w.update_face(4, [1, 2, 3])
    with pytest.raises(IndexError) as _:
        m2_w = deepcopy(m)
        m2_w.update_face(-1, [1, 2, 3])
    with pytest.raises(IndexError) as _:
        m2_wo = deepcopy(m_wo)
        m2_wo.update_face(0, [1, 2, 3])
    with pytest.raises(IndexError) as _:
        m2_wo = deepcopy(m_wo)
        m2_wo.update_face(-1, [1, 2, 3])
    # invalid vertex index raises exception
    with pytest.raises(IndexError) as _:
        m3 = deepcopy(m)
        m3.update_face(1, [4, 5, 6])
    with pytest.raises(IndexError) as _:
        m3 = deepcopy(m)
        m3.update_face(1, [-1, 0, 1])
    # change existing part conform
    m4_1 = deepcopy(m)  # list of int
    m4_1.update_face(1, [5, 2, 0])
    assert not m4_1.dangling_vert_check()
    assert len(m4_1.get_faces()) == 4
    assert m4_1.get_faces()[1][0] == 5
    assert are_edges_equal(m4_1.get_edges(), [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (3, 4), (1, 4), (2, 5), (0, 5)])
    m4_2 = deepcopy(m)  # tuple of int
    m4_2.update_face(1, (5, 2, 0))
    assert not m4_2.dangling_vert_check()
    assert len(m4_2.get_faces()) == 4
    assert m4_2.get_faces()[1][0] == 5
    assert are_edges_equal(m4_1.get_edges(), [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (3, 4), (1, 4), (2, 5), (0, 5)])
    m4_3 = deepcopy(m)  # 1D - np.ndarray
    m4_3.update_face(1, np.array([5, 2, 0]))
    assert not m4_3.dangling_vert_check()
    assert len(m4_3.get_faces()) == 4
    assert m4_3.get_faces()[1][0] == 5
    assert are_edges_equal(m4_1.get_edges(), [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (3, 4), (1, 4), (2, 5), (0, 5)])


def test_update_vertex():
    m_3d = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 2, 2], [3, 3, 3]]),
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 1], [1, 3, 4]],
        parts=[[1, 1, 1], [0, 1, 2]]
    )
    m_2d = Mesh(
        verts=np.array([[1, 0], [0, 1], [0, 0]]),
        faces=None,
        parts=None
    )
    # invalid vertex object raises an exception
    with pytest.raises(TypeError) as _:
        m1_wrong_nested = deepcopy(m_3d)
        # noinspection PyTypeChecker
        m1_wrong_nested.update_vertex(0, np.array([[1, 2, 3]]))
    # invalid vertex index raises exception
    with pytest.raises(IndexError) as _:
        m2_w = deepcopy(m_3d)
        m2_w.update_vertex(6, [1, 2, 3])
    with pytest.raises(IndexError) as _:
        m2_w = deepcopy(m_3d)
        m2_w.update_vertex(-1, [1, 2, 3])
    # wrong dimensions result in exception
    with pytest.raises(InvalidMeshException) as _:
        m3_3d = deepcopy(m_3d)
        m3_3d.update_vertex(0, np.array([0, 1]))
    with pytest.raises(InvalidMeshException) as _:
        m3_2d = deepcopy(m_2d)
        m3_2d.update_vertex(0, [0, 1, 2])
    # change existing vertex conform
    m4_3d = deepcopy(m_3d)
    m4_3d.update_vertex(0, np.array([0, 1, 2]))
    assert len(m4_3d.get_vertices()) == 6
    assert m4_3d.get_vertices()[0][1] == 1
    m4_2d = deepcopy(m_2d)
    m4_2d.update_vertex(0, np.array([0, 1]))
    assert len(m4_2d.get_vertices()) == 3
    assert np.sum(np.abs(m4_2d.get_vertices()[0] - m4_2d.get_vertices()[1])) == 0


def test__add_and_remove_faces():
    triangle_mesh = create_pyramid(triangles_only=True)
    quad_mesh = create_pyramid(triangles_only=False)
    # remove faces
    triangle_mesh.remove_faces([4, 5])
    assert len(triangle_mesh.get_faces()) == 4
    assert len(triangle_mesh.get_parts()) == 0
    # add faces
    triangle_mesh.add_faces([np.array([0, 1, 2, 3])])
    assert len(triangle_mesh.get_faces()) == 5
    assert len(triangle_mesh.get_faces()[-1]) == 4
    assert len(triangle_mesh.get_parts()) == 0
    # re-connect faces as parts
    triangle_mesh.add_parts([np.array([0, 1, 2, 3, 4])])
    assert len(triangle_mesh.get_parts()) == 1
    # changed triangle only to quad pyramid mesh
    assert triangle_mesh == quad_mesh
    assert are_edges_equal(triangle_mesh.get_edges(), quad_mesh.get_edges())


def test_parts_w_o_faces():
    """creation of a mesh with parts without faces should not be possible"""
    with pytest.raises(InvalidMeshException) as _:
        Mesh(
            verts=[[1, 0], [0, 1], [0, 0], [1, 1]],
            faces=None,
            parts=[[1, 2, 3, 4]],
        )


def test_add_to_mesh():
    m = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 2, 2], [3, 3, 3]]),
        faces=[[0, 1, 2], [1, 2, 5], [2, 3, 1], [1, 3, 4]],
        parts=[[1, 1, 1], [0, 1, 2]]
    )
    # 2d and 3d mesh raises exception
    with pytest.raises(InvalidMeshException) as _:
        m1 = deepcopy(m)
        m1_2d = Mesh(
            verts=np.array([[1, 0], [0, 1], [0, 0]]),
            faces=None,
            parts=None
        )
        m1.add_to_mesh(m1_2d)
    # invalid face indices raise exception
    with pytest.raises(IndexError) as _:
        m2 = deepcopy(m)
        m2_f = Mesh(
            verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            faces=[[-7, 0, 0]]  # too small
        )
        m2.add_to_mesh(m2_f)
    with pytest.raises(IndexError) as _:
        m3 = deepcopy(m)
        m3_f = Mesh(
            verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            faces=[[3, 0, 0]]  # too big
        )
        m3.add_to_mesh(m3_f)
    # invalid part indices raise exception
    with pytest.raises(IndexError) as _:
        m4 = deepcopy(m)
        m4_f = Mesh(
            verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            faces=[[1, 2, -3], [-1, 0, 2]],
            parts=[[-5, 0, 0]]  # too small
        )
        m4.add_to_mesh(m4_f)
    with pytest.raises(IndexError) as _:
        m5 = deepcopy(m)
        m5_f = Mesh(
            verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            faces=[[1, 2, -3], [-1, 0, 2]],
            parts=[[2, 0, 0]]  # too big
        )
        m5.add_to_mesh(m5_f)
    # regular
    m6 = deepcopy(m)
    m6_1 = deepcopy(m)
    m6 += m6_1
    assert m6_1 == m
    assert len(m6.get_vertices()) == 12
    assert len(m6.get_faces()) == 8
    assert len(m6.get_parts()) == 4
    assert m6.get_faces()[-1][2] == 10
    assert are_edges_equal(m6.get_edges(),
                           m.get_edges() + [(6, 7), (7, 8), (6, 8), (8, 11), (7, 11), (8, 9), (7, 9), (7, 10), (9, 10)])
    m7 = deepcopy(m)
    m7_1 = Mesh(
        verts=np.array([[5, 6, 7], [8, 7, 6], [5, 5, 5]]),
        faces=[[0, 1, 2], [0, 0, 0], [0, 0, 0], [0, -1, -6]],
        parts=[[-4, -3, -1]]
    )
    m7.add_to_mesh(m7_1)
    assert m7.get_faces()[-1][2] == 0
    assert m7.get_faces()[-1][1] == len(m.get_vertices()) - 1
    assert m7.get_parts()[-1][0] == 0
    assert m7.get_parts()[-1][2] == len(m.get_faces()) - 1
    assert are_edges_equal(m7.get_edges(), m.get_edges() + [(6, 6), (6, 7), (7, 8), (6, 8), (5, 6), (0, 5), (0, 6)])


def test_scale_mesh():
    m = Mesh(
        verts=np.ones((10, 3)),
        faces=None
    )
    m1 = deepcopy(m)
    m1.scale_mesh(2)
    elems, count = np.unique(m1.get_vertices(), return_counts=True)
    assert len(elems) == 1
    assert 2 in elems
    assert np.sum(count) == 10 * 3
    m1.scale_mesh(0.5)
    assert m1 == m


def test_translate_mesh():
    m0 = Mesh(
        verts=np.zeros((10, 3)),
        faces=None
    )
    m1 = Mesh(
        verts=np.ones((10, 3)),
        faces=None
    )
    # translate up
    m = deepcopy(m0)
    m.translate_mesh(np.ones(3))
    assert m == m1
    # translate down
    m = deepcopy(m1)
    m.translate_mesh(np.array([-1, -1, -1]))
    assert m == m0
    # translate with different values per column
    m = deepcopy(m0)
    m.translate_mesh(np.array([1, 2, 3]))
    assert np.sum(m.get_vertices()) == 10 * (3 + 2 + 1)
    # 2d translation raises error
    with pytest.raises(ValueError) as _:
        m_err = deepcopy(m0)
        m_err.translate_mesh(np.array([[1, 2, 3], [1, 2, 3]]))
    # wrong dimensions raise error
    with pytest.raises(ValueError) as _:
        m_err = deepcopy(m0)
        m_err.translate_mesh(np.array([1, 2, 3, 4]))


def test_translate_vertex():
    m = Mesh(verts=np.zeros((10, 3)), faces=None)
    # invalid index raises error
    with pytest.raises(IndexError) as _:
        m_err_1 = deepcopy(m)
        m_err_1.translate_vertex(-1, np.ones(3))
    with pytest.raises(IndexError) as _:
        m_err_2 = deepcopy(m)
        m_err_2.translate_vertex(-1, np.ones(3))
    # 2d translation raises error
    with pytest.raises(ValueError) as _:
        m_err_3 = deepcopy(m)
        m_err_3.translate_mesh(np.array([[1, 2, 3], [1, 2, 3]]))
    # wrong dimensions raise error
    with pytest.raises(ValueError) as _:
        m_err_4 = deepcopy(m)
        m_err_4.translate_mesh(np.array([1, 2, 3, 4]))
    # correct translation of single vertex
    m1 = deepcopy(m)
    m1.translate_vertex(1, np.array([3, -1, 5]))
    assert np.array_equal(m1.get_vertices()[1], np.array([3, -1, 5]))
    assert np.array_equal(m1.get_vertices()[0], np.zeros(3))
    assert np.array_equal(m1.get_vertices()[2:], np.zeros((8, 3)))


def test_apply_rotation():
    m = Mesh(verts=np.identity(3), faces=None)
    # invalid axis raises error
    with pytest.raises(ValueError) as _:
        m_err_1 = deepcopy(m)
        m_err_1.apply_rotation(0, 4)
    with pytest.raises(ValueError) as _:
        m_err_2 = deepcopy(m)
        m_err_2.apply_rotation(0, -1)
    # turning by 0 or 360° = 2*pi degree keeps Mesh same
    m1 = deepcopy(m)
    m1.apply_rotation(0, 1)
    assert np.allclose(m1.get_vertices(), m.get_vertices())
    m2 = deepcopy(m)
    m2.apply_rotation(2 * np.pi, 2)
    assert np.allclose(m2.get_vertices(), m.get_vertices())
    # turning single vector around z-axis then around y-axis and finally x-axis, all 90°
    m3 = Mesh(verts=np.array([[1, 2, 3]]), faces=None)
    m3.apply_rotation(0.5 * np.pi, 2)
    assert np.allclose(m3.get_vertices(), np.array([[-2, 1, 3]]))
    m3.apply_rotation(0.5 * np.pi, 1)
    assert np.allclose(m3.get_vertices(), np.array([[3, 1, 2]]))
    m3.apply_rotation(0.5 * np.pi, 0)
    assert np.allclose(m3.get_vertices(), np.array([[3, -2, 1]]))
    # turning 2D mesh
    m_2d = Mesh(verts=np.array([[1, 2], [3, 4]]), faces=None)
    m_2d.apply_rotation(0.5 * np.pi)
    assert np.allclose(m_2d.get_vertices(), np.array([[-2, 1], [-4, 3]]))
    # turning identity by 90 deg around x then turning it back
    m4 = deepcopy(m)
    m4.apply_rotation(0.5 * np.pi, 0)
    assert np.allclose(m4.get_vertices(), np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    m4.apply_rotation(-0.5 * np.pi, 0)
    assert np.allclose(m4.get_vertices(), np.identity(3))


def test_snap_to_grid():
    m = Mesh(verts=np.arange(21).reshape((7, 3)), faces=None)
    # test wrong dims on params
    with pytest.raises(ValueError) as _:  # wrong dims on grid_size
        deepcopy(m).snap_to_grid((5, 10), (1, 1, 1))
    with pytest.raises(ValueError) as _:  # wrong dims on thresh
        deepcopy(m).snap_to_grid((1, 1, 1), (10, 1))
    # grid <= 0 gives an error
    with pytest.raises(ValueError) as _:  # negative gridsize
        deepcopy(m).snap_to_grid((1, -1, 1), (0, 0, 0))
    with pytest.raises(ValueError) as _:  # negative gridsize
        deepcopy(m).snap_to_grid((0, 1, 1), (0, 0, 0))
    # threshold has to be smaller than half grid_size
    with pytest.raises(ValueError) as _:  # to big threshold
        deepcopy(m).snap_to_grid((1, 1, 1), (0.5, 0, 0))
    with pytest.raises(ValueError) as _:  # all thresh 0
        deepcopy(m).snap_to_grid((1, 1, 1), (0, 0, 0))
    # threshold = 0 does not change grid
    m1 = deepcopy(m)
    m1.snap_to_grid((1, 1, 1), (0.1, 0, 0))
    assert m1 == m
    # regular snap works up and down
    m2 = deepcopy(m)
    m2.snap_to_grid((5, 10, 15), (1, 3, 5))
    assert np.array_equal(
        m2.get_vertices(),
        np.array([[0, 0, 0], [3, 4, 0], [5, 10, 8], [10, 10, 15], [12, 10, 15], [15, 16, 15], [18, 20, 15]])
    )
    # snap to negative numbers works
    m3 = deepcopy(m)
    m3.translate_mesh(np.array([-10, -11, -10]))
    m3.snap_to_grid((5, 6, 7), (1, 2, 3))
    assert np.array_equal(
        m3.get_vertices(),
        np.array([[-10, -12, -7], [-7, -6, -7], [-5, -6, 0], [0, 0, 0], [2, 0, 7], [5, 6, 7], [8, 6, 7]])
    )


def test_remove_duplicate_vertices():
    m = create_pyramid() + create_pyramid()
    m.remove_duplicate_vertices()
    assert len(m.find_vertex(np.array([1, 1, 0]))) == 1
    assert np.array_equal(m.get_vertices(), create_pyramid().get_vertices())
    assert len(m.get_faces()) == 2 * 6
    assert all(np.array_equal(a, b) for a, b in
               zip(m.get_faces(), create_pyramid().get_faces() + create_pyramid().get_faces()))
    assert are_edges_equal(m.get_edges(), create_pyramid().get_edges())


def test_remove_duplicate_faces():
    m = create_pyramid() + create_pyramid()
    m.remove_duplicate_vertices()
    assert len(m.get_parts()) == 2
    assert len(m.get_faces()) == 2 * 6
    m.remove_duplicate_faces()
    assert len(m.get_faces()) == 6
    assert all(np.array_equal(a, b) for a, b in zip(m.get_faces(), create_pyramid().get_faces()))
    assert all(np.array_equal(a, b) for a, b in
               zip(m.get_parts(), create_pyramid().get_parts() + create_pyramid().get_parts()))
    assert len(m.get_parts()) == 2
    assert are_edges_equal(m.get_edges(), create_pyramid().get_edges())


def test_remove_duplicate_parts():
    m = create_pyramid() + create_pyramid()
    m.remove_duplicate_vertices()
    assert len(m.get_parts()) == 2
    m.remove_duplicate_faces()
    assert len(m.get_parts()) == 2
    m.remove_duplicate_parts()
    assert all(np.array_equal(a, b) for a, b in zip(m.get_parts(), create_pyramid().get_parts()))
    assert len(m.get_parts()) == 1
    assert are_edges_equal(m.get_edges(), create_pyramid().get_edges())


def test_split_mesh_into_parts():
    # case only one object
    m_pyramid = create_pyramid()
    assert len(m_pyramid.split_mesh_into_objects()) == 1
    assert create_pyramid() in m_pyramid.split_mesh_into_objects()
    # case multiple objects - single vertex object, single part object and single face object
    m = Mesh(
        verts=np.arange(24).reshape((8, 3)),
        faces=[[1, 2, 3], [2, 3, 4], [1, 2, 4], [5, 6, 7]],
        parts=[[0, 1, 2]],
    )
    new_faces = m.split_mesh_into_objects()
    assert len(new_faces) == 3
    m1 = Mesh(verts=np.arange(3).reshape((1, 3)), faces=[], parts=[])
    m2 = Mesh(
        verts=np.arange(start=3, stop=15).reshape((4, 3)),
        faces=[[0, 1, 2], [1, 2, 3], [0, 1, 3]],
        parts=[[0, 1, 2]])
    m3 = Mesh(verts=np.arange(start=15, stop=24).reshape((3, 3)), faces=[[0, 1, 2]], parts=[])
    assert m1 in new_faces
    assert m2 in new_faces
    assert m3 in new_faces
