"""
manim models for mesh objects
"""
# python imports
import copy
from typing import Tuple
# third-party imports
import manim as m
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
import numpy as np
# local imports
from manim_meshes.exceptions import InvalidMeshDimensionsException, InvalidMeshException, InvalidShapeException, \
    InvalidVertexIndex
from manim_meshes.helpers import remove_keys_from_dict
from manim_meshes.models.data_models.mesh import Mesh
from manim_meshes.models.manim_models.params import get_param_or_default, M2DM, M3DM


# class TrimeshObject(m.Polyhedron):
#     """
#     manim-meshes.models.TrimeshObject
#
#     Trimesh Object:
#     uses Python trimesh package to create a manim Polyhedron object to be rendered
#     """
#
#     def __init__(self, mesh: trimesh.Trimesh, *args, **kwargs):
#         # custom parameters
#         self.mesh: trimesh.Trimesh = mesh
#
#         # initialize Polyhedron
#         super().__init__(
#             vertex_coords=mesh.vertices,
#             faces_list=mesh.faces,
#             *args,
#             **kwargs,
#         )
#
#     def get_edges(self, *args, **kwargs) -> List[Tuple[int, int]]:
#         """
#         use trimesh to get edges
#         """
#         return [(edge[0], edge[1]) for edge in self.mesh.edges]
#
#     def create_faces(
#         self,
#         face_coords: List[List[List or np.ndarray]],
#     ) -> m.VGroup:
#         """Creates VGroup of faces from a list of face coordinates."""
#         face_group = m.VGroup()
#         for face in face_coords:
#             face_group.add(m.Polygon(*face, **self.faces_config))
#         return face_group
#
#     def update_mesh(self, mesh: trimesh.Trimesh):
#         """
#         use a new mesh
#         """
#         self.mesh = mesh
#         self.vertex_coords = mesh.vertices
#         self.faces_list = mesh.faces
#         # TODO reload other params?
#
#     def color_all_faces(self):
#         """
#         color all the faces of the polyhedron, make sure no neighboring faces have the same color
#         """
#         raise NotImplementedError

# pylint: disable=too-many-instance-attributes
class ManimMesh(m.VGroup, metaclass=ConvertToOpenGL):
    """
    another Mesh implementation, a little bit faster + looks better
    -> FIXME has no vertex dots, necessary?

    inspired by manim class 'Surface'
    """

    def __init__(self, scene: m.Scene, mesh: Mesh, *args, **kwargs) -> None:
        super().__init__(*args)
        self.mesh: Mesh = mesh
        self.scene: m.Scene = scene
        self.vertices: m.VGroup = m.VGroup()
        self.edges: m.VGroup = m.VGroup()
        self.faces: m.VGroup = m.VGroup()

        # set all the parameters
        self.display_vertices = get_param_or_default("display_vertices", kwargs, M3DM)
        self.display_edges = get_param_or_default("display_edges", kwargs, M3DM)
        self.display_faces = get_param_or_default("display_faces", kwargs, M3DM)
        self.clear_vertices = get_param_or_default("clear_vertices", kwargs, M3DM)
        self.clear_edges = get_param_or_default("clear_edges", kwargs, M3DM)
        self.clear_faces = get_param_or_default("clear_faces", kwargs, M3DM)
        self.edges_color = get_param_or_default("edges_color", kwargs, M3DM)
        self.edges_width = get_param_or_default("edges_width", kwargs, M3DM)
        self.faces_color = get_param_or_default("faces_color", kwargs, M3DM)
        self.faces_opacity = get_param_or_default("faces_opacity", kwargs, M3DM)
        self.verts_color = get_param_or_default("verts_color", kwargs, M3DM)

        self.pre_function_handle_to_anchor_scale_factor = (
            get_param_or_default("pre_function_handle_to_anchor_scale_factor", kwargs, M3DM)
        )
        self._setup()

    def _setup(self):
        """set all the necessary mesh parameters"""
        if self.display_faces:
            self._setup_faces()
        if self.display_edges:
            self._setup_edges()
        if self.display_vertices:
            self._setup_vertices()
        self.add(self.faces, self.edges, self.vertices)

    def _setup_vertices(self):
        """set the vertices as 3D manim objects"""
        if self.clear_vertices:
            self.vertices = m.VGroup()

        for v in self.mesh.get_3d_vertices():
            self.vertices.add(m.Sphere(v, radius=0.03, fill_color=self.verts_color, fill_opacity=1.,  stroke_width=0.))

    def _setup_edges(self):
        """set the edges as manim objects"""
        if self.clear_edges:
            self.edges.clear_points()

        vertices = self.mesh.get_3d_vertices()
        for edge_verts in self.mesh.get_edges():
            vert_1 = vertices[edge_verts[0]]
            vert_2 = vertices[edge_verts[1]]
            edge = m.ThreeDVMobject()
            edge.set_points_as_corners([vert_1, vert_2])
            self.edges.add(edge)
        # for all edges at once
        self.edges.set_fill(
            color=self.edges_color,
            opacity=1
        )
        self.edges.set_stroke(
            color=self.edges_color,
            width=self.edges_width,
            opacity=1.,
        )

    def _setup_faces(self):
        """
        set the current mesh up as manim objects
        should work for any sized face, not just triangles
        """
        if self.clear_faces:
            self.faces.clear_points()

        verts_3d = self.mesh.get_3d_vertices()
        for face_indices in self.mesh.get_faces():
            face_points = [verts_3d[i] for i in face_indices]
            face_points.append(verts_3d[face_indices[0]])
            new_face = m.ThreeDVMobject()
            new_face.set_points_as_corners(
                face_points
            )
            self.faces.add(new_face)
        self.faces.set_fill(
            color=self.faces_color,
            opacity=self.faces_opacity
        )
        self.faces.set_stroke(
            color=self.faces_color,
            width=0.,
            opacity=0.,
        )

    def get_face(self, face_idx):
        """get the faces with the given id"""
        return self.faces.submobjects[face_idx]

    def get_edge(self, edge_idx):
        """get the edge with the given id"""
        return self.edges.submobjects[edge_idx]

    def align_points_with_larger(self, larger_mobject):
        """abstract from super - please the linter"""
        raise NotImplementedError

    def _update_vertex(self, vertex_idx: int, pos: np.ndarray):
        """
        TODO
        """
        # update mesh
        self.mesh.update_vertex(vertex_idx, pos)
        # update faces
        for face_idx, face in enumerate(self.mesh.get_faces()):
            if vertex_idx in face:
                triangle = [self.mesh.get_3d_vertices()[i] for i in face]
                face = self.get_face(face_idx)
                face.set_points_as_corners(
                    [
                        triangle[0],
                        triangle[1],
                        triangle[2],
                        triangle[0]
                    ],
                )
        # update edges
        for edge in self.mesh.get_vertex_edges(vertex_idx):
            self._update_edge(edge)

    def _update_edge(self, edge: Tuple[int, ...]):
        e = self.get_edge(self.mesh.get_edge_index(edge))
        vert_1 = self.mesh.get_3d_vertices()[edge[0]]
        vert_2 = self.mesh.get_3d_vertices()[edge[1]]
        e.set_points_as_corners([vert_1, vert_2])

    def shift_vertex(self, vertex_idx: int, shift: np.ndarray, **kwargs):
        """shift vertex and update faces"""
        # expect everything has the same dimensions as mesh.dim
        start = self.mesh.get_vertices()[vertex_idx].copy()
        tracker = m.ValueTracker(0)
        tracker.add_updater(lambda mo: self._update_vertex(vertex_idx, start + tracker.get_value() * shift, **kwargs))
        self.scene.add(tracker)
        self.scene.play(tracker.animate(**kwargs).set_value(1))
        self.scene.remove(tracker)

    def move_vertices_to(self, new_positions: np.ndarray, **kwargs):
        """visually move all vertices to new positions and update faces"""
        if len(new_positions) != self.mesh.nof_vertices:
            raise InvalidShapeException("new_positions", len(new_positions), self.mesh.nof_vertices)
        for i, new_pos in enumerate(new_positions):
            self.move_vertex_to(vertex_idx=i, pos=new_pos, **kwargs)

    def move_vertex_to(self, vertex_idx: int, pos: np.ndarray, **kwargs):
        """visually move vertex to pos and update faces"""
        if vertex_idx < 0 or self.mesh.nof_vertices < vertex_idx:
            raise InvalidVertexIndex(vertex_idx, self.mesh.nof_vertices)
        # expect pos and curr_pos / mesh.dim to have the same dimensions
        if self.mesh.dim != len(pos):
            raise InvalidMeshDimensionsException(len(pos), self.mesh.dim, "pos")
        current_pos = self.mesh.get_vertices()[vertex_idx].copy()
        shift = pos - current_pos
        # use shift method to slowly move point to desired place
        self.shift_vertex(vertex_idx, shift, **kwargs)

    def move_to_grid(self, scene: m.Scene, grid_sizes: Tuple[float, ...], threshold: Tuple[float, ...], nof_steps: int):
        """slowly snap to a given grid, uses stepwise mesh.snap_to_grid()"""
        for step in range(nof_steps, 0, -1):
            # to be able to show the movement, the update needs to be calculated on a dummy mesh first
            old_mesh: Mesh = copy.copy(self.mesh)
            old_mesh.snap_to_grid(grid_sizes, threshold, step)
            # use new calculated positions but have still the old mesh
            self.move_vertices_to(self.scene, old_mesh.get_vertices())
            scene.wait(0.5)


class Manim2DMesh(ManimMesh):
    """
    "2D" mesh implementation
    printing Vertices in Manim is currently not supported for 2D vertices. Therefore, while printing 3D-vertices
    have to be used. But the Manim2DMesh class should support 2D vertices or 3D vertices with z-value == 0 on init

    This mesh is mainly for Educational purposes and has a few functions we needed for drawing basic
     mesh functionalities.
    """

    def __init__(self, scene: m.Scene, mesh: Mesh, *args, **kwargs) -> None:
        if mesh.dim == 3:
            if np.sum(np.abs(mesh.get_vertices()[:, 2] != 0)):
                raise InvalidMeshException("Mesh has z values != 0 and therefore is not 2D.")
        elif mesh.dim > 3:
            raise InvalidMeshException(
                f'Mesh is not in the correct format. Expected Dim 2 or 3 with z zero, was {mesh.dim}')
        # init ManimMesh
        super().__init__(
            mesh=mesh,
            scene=scene,
            display_vertices=get_param_or_default("display_vertices", kwargs, M2DM),
            display_edges=get_param_or_default("display_edges", kwargs, M2DM),
            display_faces=get_param_or_default("display_faces", kwargs, M2DM),
            clear_vertices=get_param_or_default("clear_vertices", kwargs, M2DM),
            clear_edges=get_param_or_default("clear_edges", kwargs, M2DM),
            clear_faces=get_param_or_default("clear_faces", kwargs, M2DM),
            faces_color=get_param_or_default("faces_color", kwargs, M2DM),
            faces_opacity=get_param_or_default("faces_opacity", kwargs, M2DM),
            edges_color=get_param_or_default("edges_color", kwargs, M2DM),
            edges_width=get_param_or_default("edges_width", kwargs, M2DM),
            verts_color=get_param_or_default("verts_color", kwargs, M2DM),
            pre_function_handle_to_anchor_scale_factor=get_param_or_default(
                "pre_function_handle_to_anchor_scale_factor", kwargs, M2DM),
            *args,
            **remove_keys_from_dict(kwargs, list(M2DM.keys())),
        )

    def _setup_vertices(self):
        """set the vertices as 2D manim objects"""
        if self.clear_vertices:
            self.vertices.clear_points()
            self.vertices = m.VGroup()

        for v in self.mesh.get_3d_vertices():
            self.vertices.add(m.Dot(v, radius=0.03, color=self.verts_color))
        self.add(self.vertices)

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

    def edge_flip(self, face_idx_1: int, face_idx_2: int, **kwargs):
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
        self.scene.play(*anims)

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
