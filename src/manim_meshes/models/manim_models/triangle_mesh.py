"""
2D triangle manim mesh
"""
# python imports

# third-party imports
import manim as m
import numpy as np
# local imports
from manim_meshes.exceptions import InvalidMeshException
from manim_meshes.models.data_models.mesh import Mesh
from manim_meshes.models.manim_models.basic_mesh import Manim2DMesh

class TriangleManim2DMesh(Manim2DMesh):
    """
    "2D" mesh implementation
    printing Vertices in Manim is currently not supported for 2D vertices. Therefore, while printing 3D-vertices
    have to be used. But the Manim2DMesh class should support 2D vertices or 3D vertices with z-value == 0 on init

    This mesh is mainly for Educational purposes and has a few functions we needed for drawing basic
     mesh functionalities.
    """

    def __init__(self, mesh: Mesh, *args, **kwargs) -> None:
        if any(len(face) != 3 for face in mesh.get_faces()):
            raise InvalidMeshException("Mesh must only consist of triangles!")
        # init Manim2DMesh
        super().__init__(
            mesh=mesh,
            *args,
            **kwargs,
        )

    def get_circle(self, face_idx: int, **kwargs):
        """create a circum-circle around face with given idx"""
        face = self.mesh.get_faces()[face_idx]
        vertices = [self.mesh.get_3d_vertices()[i] for i in face]
        center, radius = self._get_triangle_circum_circle_params(*vertices)
        if 'stroke_width' not in kwargs:
            kwargs['stroke_width'] = 2
        circ = m.Circle(radius, **kwargs)
        circ.shift(center)
        return circ

    def edge_flip(self,scene: m.Scene, face_idx_1: int, face_idx_2: int, **kwargs):
        """
        Flips the edge shared by the given triangles. Raises an error if the faces are not triangles
        or do not share exactly one edge.
        """
        face_arr_1 = self.mesh.get_faces()[face_idx_1]
        face_arr_2 = self.mesh.get_faces()[face_idx_2]
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
            face = self.mesh.get_faces()[face_idx]
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

    def get_points_violating_delaunay(self, face_id: int):
        """given a triangle by id, get all points violating delaunay criterion"""
        points, indices = [], []
        face = self.mesh.get_faces()[face_id]
        center, radius = self._get_triangle_circum_circle_params(*[self.mesh.get_3d_vertices()[i] for i in face])

        # TODO: [improve to be faster] don't loop all vertices, only loop ones that are "close"
        for idx, point in enumerate(self.mesh.get_3d_vertices()):
            if idx not in face:
                distance = np.linalg.norm(center - point)
                if distance < radius:  # inside circle
                    dot = m.Dot(point, radius=0.03, color=m.RED)
                    dot.add_updater(lambda mo, mesh=self.mesh, index=idx: mo.move_to(mesh.get_3d_vertices()[index]))
                    points.append(dot)
                    indices.append(idx)
        return points, indices

    def is_point_violating_delaunay(self, vertex_idx: int, face_idx):
        """
            returns True if the vertex with index vertex_idx is violating the delaunay criterion
            w.r.t. the provided face (face_idx).
        """
        point = self.mesh.get_3d_vertices()[vertex_idx]
        face = self.mesh.get_faces()[face_idx]
        center, radius = self._get_triangle_circum_circle_params(*[self.mesh.get_3d_vertices()[i] for i in face])
        distance = np.linalg.norm(center - point)
        return distance < radius

    @staticmethod
    def _get_triangle_circum_circle_params(
            pt1: np.ndarray,
            pt2: np.ndarray,
            pt3: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        """
        Given the three corner-points of a triangle, calculate the parameters of the circum-circle
        :param: three corner-points of one triangle
        :returns: center point, radius
        """
        div = 2 * np.linalg.norm(np.cross(pt1 - pt2, pt2 - pt3)) ** 2
        alpha = np.linalg.norm(pt2 - pt3) ** 2 * (pt1 - pt2).dot(pt1 - pt3) / div
        beta = np.linalg.norm(pt1 - pt3) ** 2 * (pt2 - pt1).dot(pt2 - pt3) / div
        gamma = np.linalg.norm(pt1 - pt2) ** 2 * (pt3 - pt1).dot(pt3 - pt2) / div
        center = alpha * pt1 + beta * pt2 + gamma * pt3
        div = 2 * np.linalg.norm(np.cross(pt1 - pt2, pt2 - pt3))
        radius = np.linalg.norm(pt1 - pt2) * np.linalg.norm(pt2 - pt3) * np.linalg.norm(pt3 - pt1) / div
        return center, radius

    def align_points_with_larger(self, larger_mobject):
        """abstract from super - please the linter"""
        raise NotImplementedError
