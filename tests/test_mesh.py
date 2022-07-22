"""
test mesh functionalities
"""
# python imports
from copy import deepcopy
# third-party imports
import numpy as np
import pytest
# local imports
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

    # not in is the same as != for all three
    tm4 = Mesh(verts=np.array([[1, 2, 3]]), faces=None, parts=None)
    tm5 = Mesh(verts=np.array([[1, 2, 3]]), faces=[], parts=[])
    tm6 = Mesh(verts=np.array([[1, 2, 3]]), faces=[], parts=None)
    tm7 = Mesh(verts=np.array([[1, 2, 3]]), faces=[[0, 1, 2]], parts=None)
    tm8 = Mesh(verts=np.array([[1, 2, 3]]), faces=[[0, 1, 2]], parts=[[0, 1, 2]])
    assert tm4 == tm5 == tm6
    assert tm7 not in (tm4, tm5, tm6)
    assert tm8 not in (tm4, tm5, tm6)
    assert tm7 != tm8


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
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        parts=[[0, 1, 2]]
    )
    # mesh stays the same except for parts
    m1 = deepcopy(m)
    m1.remove_parts([0])
    assert Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        parts=None
    ) == m1
    # removing non existent part raises an exception
    with pytest.raises(IndexError) as _:
        m2 = deepcopy(m)
        m2.remove_parts([1])
    # order matters and no multi indices
    m3 = deepcopy(m)
    m4 = Mesh(
        verts=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        faces=[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        parts=[[0, 1, 2], [1, 1, 1], [2, 2, 2], [1, 2, 1], [2, 1, 1]]
    )
    m4.remove_parts([1, 3, 2, 1, 4])
    assert m3 == m4


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
    # removing one dangling face does not change parts
    m2 = deepcopy(m)
    m2.remove_faces([3])
    assert len(m2.get_vertices()) == 4
    assert len(m2.get_faces()) == 3
    assert len(m2.get_parts()) == 2
    assert not m2.dangling_face_check()
    # removing one non-dangling face removes the part that uses it
    m3 = deepcopy(m)
    m3.remove_faces([2])
    assert len(m3.get_vertices()) == 4
    assert len(m3.get_faces()) == 3
    assert len(m3.get_parts()) == 1
    assert m3.dangling_face_check()
    # removing all faces works and order does not matter
    m4 = deepcopy(m)
    m4.remove_faces([0, 3, 1, 2])
    assert len(m4.get_vertices()) == 4
    assert len(m4.get_faces()) == 0
    assert len(m4.get_parts()) == 0
    # removing one face may result in multiple part deletions
    m5 = deepcopy(m)
    m5.remove_faces([1])
    assert len(m5.get_vertices()) == 4
    assert len(m5.get_faces()) == 3
    assert len(m5.get_parts()) == 0


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
    # removing unused vertex does not change mesh
    m2 = deepcopy(m)
    m2.remove_vertices([5])
    assert len(m2.get_vertices()) == 5
    assert len(m2.get_faces()) == 4
    assert len(m2.get_parts()) == 2
    # removing faces but not independent parts
    m3 = deepcopy(m)
    m3.remove_vertices([4])
    assert len(m3.get_vertices()) == 5
    assert len(m3.get_faces()) == 3
    assert len(m3.get_parts()) == 2
    assert not m3.dangling_face_check()
    # removing multiple vertices works and order does not matter
    m4 = deepcopy(m)
    m4.remove_vertices([0, 3, 1, 2, 5])  # not 4
    assert len(m4.get_vertices()) == 1
    assert len(m4.get_faces()) == 0
    assert len(m4.get_parts()) == 0
    # removing vertex results in removing faces and parts only if necessary
    m5 = deepcopy(m)
    m5.remove_vertices([0])
    assert len(m5.get_vertices()) == 5
    assert len(m5.get_faces()) == 3
    assert len(m5.get_parts()) == 1


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
    # add single to non-existing
    m3 = deepcopy(m)
    m3.add_faces([np.array([0, 1, 2])])
    assert len(m3.get_faces()) == 1
    # add single to existing - and size can be different
    m3.add_faces([np.array([1, 2, 3, 3])])
    assert len(m3.get_faces()) == 2
    # add multiple to non-existing
    m4 = deepcopy(m)
    m4.add_faces([np.array([0, 1, 2]), np.array([1, 2, 3])])
    assert len(m4.get_faces()) == 2
    # add multiple to existing
    m4.add_faces([np.array([1, 1, 1]), np.array([2, 2, 2, 2])])
    assert len(m4.get_faces()) == 4


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
    # add multiple
    m4 = deepcopy(m)
    m4.add_vertices(np.array([[0, 0, 0], [1,1,1], [1,2,3]]))
    assert len(m4.get_vertices()) == 4


def test_update_part():
    pass


def test_update_face():
    pass


def test_update_vertex():
    pass


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


def test_parts_w_o_faces():
    """creation of a mesh with parts without faces should not be possible"""
    with pytest.raises(InvalidMeshException) as _:
        Mesh(
            verts=[[1, 0], [0, 1], [0, 0], [1, 1]],
            faces=None,
            parts=[[1, 2, 3, 4]],
        )

# add_to_mesh (funktioniert auch noch nicht)
# __iadd__
# __add__

# add_faces
# remove_faces
# update_face
#   - update using list, np array, np multi dim array

# add_parts
# remove_parts
# update_part
#   - update using list, np array, np multi dim array

# dangling_face_check
# scale_mesh
# translate_mesh
# translate_vertex

# update_vertex
#   - update using list, np array, np multi dim array

# test creating mesh with [] and None for faces and parts
# test "anything" with faces and parts as []
