"""
test mesh functionalities
"""
# third-party imports
import numpy as np
# local imports
from manim_meshes.models.mesh import Mesh



def test_dangling():
    # node 4 is not used
    m = Mesh(
        verts=np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]]),
        faces=np.array([[0, 1, 2]])
    )
    assert m.dangling_check()

    # all nodes are used
    m = Mesh(
        verts=np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]]),
        faces=np.array([[0, 1, 2], [2, 3, 1]])
    )
    assert not m.dangling_check()
