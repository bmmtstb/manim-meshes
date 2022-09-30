"""
2D triangle manim mesh with functionality useful for delaunay meshes
"""
# python imports
# third-party imports
import manim as m
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
import numpy as np
# local imports
from manim_meshes.exceptions import InvalidMeshException
from manim_meshes.models.data_models.mesh import Mesh
from manim_meshes.models.manim_models.basic_mesh import Manim2DMesh


class TriangleManim2DMesh(Manim2DMesh, metaclass=ConvertToOpenGL):
    """2D Mesh implementation that has additional methods especially for triangles"""
    # pylint:disable=abstract-method
    def __init__(self, mesh: Mesh, *args, **kwargs) -> None:
        if any(len(face) != 3 for face in mesh.faces):
            raise InvalidMeshException("Mesh must only consist of triangles!")
        # init Manim2DMesh
        super().__init__(
            mesh=mesh,
            *args,
            **kwargs,
        )

    def edge_flip(self, scene: m.Scene, face_idx_1: int, face_idx_2: int, **kwargs) -> None:
        """
        Flips the edge shared by the given triangles. Raises an error if the faces are not triangles
        or do not share exactly one edge
        """
        face_arr_1 = self.mesh.faces[face_idx_1]
        face_arr_2 = self.mesh.faces[face_idx_2]
        if len(face_arr_1) != 3 or len(face_arr_2) != 3:
            raise ValueError("Faces must be triangles!")
        mask_1, mask_2 = np.isin(face_arr_1, face_arr_2), np.isin(face_arr_2, face_arr_1)
        if mask_1.sum() != 2:
            raise ValueError("Faces must share exactly one edge!")
        # currently ignores resulting winding order (should this be fixed?)
        v_1, v_2 = face_arr_1[~mask_1][0], face_arr_2[~mask_2][0]  # new shared edge
        v_3_a, v_3_b = face_arr_1[mask_1][0], face_arr_1[mask_1][1]  # new unshared vertices
        old_edge = self.get_edge(self.mesh.get_edge_index(tuple(sorted([v_3_a, v_3_b]))))
        self.mesh.update_face(face_idx_1, np.array([v_1, v_2, v_3_a]))
        self.mesh.update_face(face_idx_2, np.array([v_1, v_2, v_3_b]))
        anims = []
        if 'run_time' not in kwargs:
            kwargs['run_time'] = 1
        for face_idx in [face_idx_1, face_idx_2]:
            face = self.mesh.faces[face_idx]
            triangle = [self.mesh.get_3d_vertices()[i] for i in face]
            face = self.get_face(face_idx)
            new_face = face.copy()
            new_face.set_points_as_corners(
                [
                    triangle[0],
                    triangle[1],
                    triangle[2],
                    triangle[0]
                ],
            )
            anims.append(face.animate(**kwargs).become(new_face))
        new_edge = old_edge.copy()
        new_edge.set_points_as_corners([self.mesh.get_3d_vertices()[v_1], self.mesh.get_3d_vertices()[v_2]])
        anims.append(old_edge.animate(**kwargs).become(new_edge))
        scene.play(*anims)
