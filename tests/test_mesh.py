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
        verts=np.arange(0, 7, 1),
        faces=np.asarray([[1, 2, 3], [5, 2, 3], [1, 6, 3]])
    )
    assert not m.dangling_check()

    # all nodes are used
    m = Mesh(
        verts=np.arange(0, 7, 1),
        faces=np.asarray([[1, 2, 3], [5, 2, 4], [1, 6, 3]])
    )
    assert not m.dangling_check()
