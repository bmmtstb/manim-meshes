"""
test mesh functionalities
"""
# third-party imports
import numpy as np
import pytest
# local imports
from manim_meshes.models.mesh import InvalidMeshException, Mesh
from manim_meshes.templates import create_pyramid


def test_dangling_vertices():
    # node 4 is not used -> has dangling vertices
    m = Mesh(
        verts=np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ]),
        faces=np.array([
            [0, 1, 2],
        ])
    )
    assert m.dangling_vert_check()

    # with second triangle - all nodes are used
    m = Mesh(
        verts=np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ]),
        faces=np.array([
            [0, 1, 2],
            [2, 3, 1],
        ])
    )
    assert not m.dangling_vert_check()


# def test_dangling_faces():
#     m = create_pyramid()
#     assert not m.dangling_face_check()
#
#     m = Mesh(
#         verts=np.array([
#             []
#         ]),
#         faces=np.array(),
#         parts=np.array(),
#     )
#     assert m.dangling_face_check()


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
            verts=[
                [1, 0],
                [0, 1],
                [0, 0, 1],
            ],
            faces=[[0, 1, 2]],
        )


def test_different_dims_on_faces():
    """2D and 3D vertices should not work"""
    try:
        Mesh(
            verts=[
                [1, 0],
                [0, 1],
                [0, 0],
                [1, 1],
            ],
            faces=[
                [0, 1, 2],
                [0, 1, 2, 3],
            ],
        )
    except InvalidMeshException as e:
        pytest.fail("Different sized faces should work." + str(e))
