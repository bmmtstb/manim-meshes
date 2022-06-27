"""
manim models for mesh objects
"""
# third-party imports
import trimesh
import manim as m
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
import numpy as np
# local imports
from manim_meshes.params import get_param_or_default, M2DM, M3DM, Parameters


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


class ManimMesh(m.VGroup, metaclass=ConvertToOpenGL):
    """
    another Mesh implementation, a little bit faster + looks better
    -> FIXME has no vertex dots, necessary?

    inspired by manim class 'Surface'
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        *args,
        params: Parameters = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mesh: trimesh.Trimesh = mesh

        # set all the parameters
        self.faces_stroke_color = get_param_or_default("faces_stroke_color", params, M3DM)
        self.faces_stroke_width = get_param_or_default("faces_stroke_width", params, M3DM)
        self.fill_color = get_param_or_default("fill_color", params, M3DM)
        self.fill_opacity = get_param_or_default("fill_opacity", params, M3DM)
        self.display_vertices = get_param_or_default("display_vertices", params, M3DM)
        self.display_edges = get_param_or_default("display_edges", params, M3DM)
        self.display_faces = get_param_or_default("display_faces", params, M3DM)

        self.pre_function_handle_to_anchor_scale_factor = (
            get_param_or_default("pre_function_handle_to_anchor_scale_factor", params, M3DM)
        )
        self._setup()

    def _setup(self):
        """set all the necessary mesh parameters"""
        if self.display_vertices:
            self._setup_vertices()
        if self.display_edges:
            self._setup_edges()
        if self.display_faces:
            self._setup_faces()

    def _setup_vertices(self):
        """set the vertices as manim objects"""
        vertices = m.VGroup()
        for vert_coord in np.asarray(self.mesh.vertices):
            vertices.add(m.Point(
                vert_coord,
                color=m.BLUE,
                stroke_width=self.stroke_width,
            ))
        self.add(*vertices)

    def _setup_edges(self):
        """set the edges as manim objects"""
        raise NotImplementedError

    def _setup_faces(self):
        """set the current mesh up as manim objects"""
        faces = m.VGroup()
        for face_indices in self.mesh.faces:
            triangle = [self.mesh.vertices[i] for i in face_indices]
            new_face = m.ThreeDVMobject()
            new_face.set_points_as_corners(
                [
                    triangle[0],
                    triangle[1],
                    triangle[2],
                    triangle[0]
                ],
            )
            faces.add(new_face)
        faces.set_fill(color=self.fill_color, opacity=self.fill_opacity)
        faces.set_stroke(
            # color=self.stroke_color,
            width=self.stroke_width,
            opacity=self.stroke_opacity,
        )
        self.add(*faces)

    def get_face(self, face_idx):
        """get the faces with the given id"""
        return self.submobjects[face_idx]


class Manim2DMesh(ManimMesh):
    """
    TODO
    """
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        params: Parameters = None,
        *args,
        **kwargs,
    ) -> None:
        if len(mesh.facets[0]) < len(mesh.faces):
            raise Exception('Mesh is not 2D!')
        # Todo: rotate mesh if mesh not already in z-plane
        super().__init__(
            mesh,
            fill_color=get_param_or_default("fill_color", params, M2DM),
            fill_opacity=get_param_or_default("fill_opacity", params, M2DM),
            stroke_color=get_param_or_default("stroke_color", params, M2DM),
            stroke_width=get_param_or_default("stroke_width", params, M2DM),
            pre_function_handle_to_anchor_scale_factor=get_param_or_default(
                "pre_function_handle_to_anchor_scale_factor", params, M2DM),
            **kwargs,
        )

    def get_circle(self, face_idx: int):
        """create a circum-circle around face with given id"""
        face = self.mesh.faces[face_idx]
        vertices = [self.mesh.vertices[i] for i in face]
        center, radius = self._get_triangle_circum_circle_params(*vertices)
        circ = m.Circle(radius, stroke_width=2)
        circ.move_to(center)
        return circ

    def get_points_violating_delaunay(self, face_idx_1: int):
        """given a triangle by id, get all points violating delaunay criterion"""
        points = []
        face_1 = self.mesh.faces[face_idx_1]
        center, radius = self._get_triangle_circum_circle_params(*[self.mesh.vertices[i] for i in face_1])
        # TODO: [improve to be faster] don't loop all vertices, only loop ones that are "close"
        for _, point in enumerate(self.mesh.vertices):
            point = np.asarray(point)
            if point not in face_1:
                distance = np.linalg.norm(center - point)
                if distance < radius:  # inside circle
                    points.append(m.Dot(point, radius=0.03, color=m.RED))
        return points

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
