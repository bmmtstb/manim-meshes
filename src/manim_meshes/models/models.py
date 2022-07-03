"""
manim models for mesh objects
"""
# third-party imports
import manim as m
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
import numpy as np
# local imports
from manim_meshes.models.mesh import Mesh
from manim_meshes.models.params import get_param_or_default, M2DM, M3DM


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
            mesh: Mesh,
            *args,
            **kwargs,
    ) -> None:
        """
        @keyword display_vertices
        """
        super().__init__(*args, **kwargs)
        self.mesh: Mesh = mesh

        # set all the parameters
        self.display_vertices = get_param_or_default("display_vertices", kwargs, M3DM)
        self.display_edges = get_param_or_default("display_edges", kwargs, M3DM)
        self.display_faces = get_param_or_default("display_faces", kwargs, M3DM)
        self.edges_fill_color = get_param_or_default("edges_fill_color", kwargs, M3DM)
        self.edges_fill_opacity = get_param_or_default("edges_fill_opacity", kwargs, M3DM)
        self.edges_stroke_color = get_param_or_default("edges_stroke_color", kwargs, M3DM)
        self.edges_stroke_opacity = get_param_or_default("edges_stroke_opacity", kwargs, M3DM)
        self.edges_stroke_width = get_param_or_default("edges_stroke_width", kwargs, M3DM)
        self.faces_fill_color = get_param_or_default("faces_fill_color", kwargs, M3DM)
        self.faces_fill_opacity = get_param_or_default("faces_fill_opacity", kwargs, M3DM)
        self.faces_stroke_color = get_param_or_default("faces_stroke_color", kwargs, M3DM)
        self.faces_stroke_opacity = get_param_or_default("faces_stroke_opacity", kwargs, M3DM)
        self.faces_stroke_width = get_param_or_default("faces_stroke_width", kwargs, M3DM)
        self.verts_fill_color = get_param_or_default("verts_fill_color", kwargs, M3DM)
        self.verts_fill_opacity = get_param_or_default("verts_fill_opacity", kwargs, M3DM)
        self.verts_stroke_color = get_param_or_default("verts_stroke_color", kwargs, M3DM)
        self.verts_stroke_opacity = get_param_or_default("verts_stroke_opacity", kwargs, M3DM)
        self.verts_stroke_width = get_param_or_default("verts_stroke_width", kwargs, M3DM)

        self.pre_function_handle_to_anchor_scale_factor = (
            get_param_or_default("pre_function_handle_to_anchor_scale_factor", kwargs, M3DM)
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
        # vertices = m.Group()
        # for vert_coord in np.asarray(self.mesh.vertices):
        #     # FIXME does not print right now
        #     vertices.add(m.Point(
        #         location=vert_coord,
        #         color=m.BLUE,
        #         # fill_opacity=self.verts_fill_opacity,
        #         # stroke_width=self.verts_stroke_width,
        #     ))
        # self.add(*vertices)

    def _setup_edges(self):
        """set the edges as manim objects"""
        # edges = m.VGroup()
        # vertices = np.asarray(self.mesh.vertices)
        # for edge_verts in np.asarray(self.mesh.edges):
        #     vert_1 = vertices[edge_verts[0]]
        #     vert_2 = vertices[edge_verts[1]]
        #     # FIXME which object
        # edges.set_fill(
        #     color=self.edges_fill_color,
        #     opacity=self.edges_fill_opacity
        # )
        # edges.set_stroke(
        #     width=self.edge_stroke_width,
        #     opacity=self.edge_stroke_opacity,
        # )
        # self.add(*edges)

    def _setup_faces(self):
        """set the current mesh up as manim objects"""
        faces = m.VGroup()
        for face_indices in self.mesh.get_faces():
            triangle = [self.mesh.get_vertices()[i] for i in face_indices]
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
        faces.set_fill(
            color=self.faces_fill_color,
            opacity=self.faces_fill_opacity
        )
        faces.set_stroke(
            color=self.faces_stroke_color,
            width=self.faces_stroke_width,
            opacity=self.faces_stroke_opacity,
        )
        self.add(*faces)

    def get_face(self, face_idx):
        """get the faces with the given id"""
        return self.submobjects[face_idx]

    def align_points_with_larger(self, larger_mobject):
        """abstract from super - please the linter"""
        raise NotImplementedError


class Manim2DMesh(ManimMesh):
    """
    TODO
    """

    def __init__(
            self,
            mesh: Mesh,
            *args,
            **kwargs,
    ) -> None:
        if any(mesh.get_vertices()[:, 2] != 0):
            raise Exception('Mesh is not 2D / z-coordinates not 0!')
        super().__init__(
            mesh,
            faces_fill_color=get_param_or_default("faces_fill_color", kwargs, M2DM),
            faces_fill_opacity=get_param_or_default("faces_fill_opacity", kwargs, M2DM),
            faces_stroke_color=get_param_or_default("faces_stroke_color", kwargs, M2DM),
            faces_stroke_width=get_param_or_default("faces_stroke_width", kwargs, M2DM),
            pre_function_handle_to_anchor_scale_factor=get_param_or_default(
                "pre_function_handle_to_anchor_scale_factor", kwargs, M2DM),
            *args,
            **kwargs,
        )

    def get_circle(self, face_id: int):
        """create a circum-circle around face with given id"""
        face = self.mesh.get_faces()[face_id]
        vertices = [self.mesh.get_vertices()[i] for i in face]
        center, radius = self._get_triangle_circum_circle_params(*vertices)
        circ = m.Circle(radius, stroke_width=2)
        circ.move_to(center)
        return circ

    def get_points_violating_delaunay(self, face_id: int):
        """given a triangle by id, get all points violating delaunay criterion"""
        points = []
        face = self.mesh.get_faces()[face_id]
        center, radius = self._get_triangle_circum_circle_params(*[self.mesh.get_vertices()[i] for i in face])
        # TODO: [improve to be faster] don't loop all vertices, only loop ones that are "close"
        for idx, point in enumerate(self.mesh.get_vertices()):
            if idx not in face:
                distance = np.linalg.norm(center - point)
                if distance < radius:  # inside circle
                    points.append(m.Dot(point, radius=0.03, color=m.RED))
        return points

    def move_vertex_to(self, vertex_id: int, position: np.ndarray):
        """move vertex and update faces"""
        # face_ids = []  # TODO
        #       for face_id in face_ids:
        #           triangle = [self.mesh.get_vertices()[i] for i in self.mesh.get_faces()[face_id]]
        #           face = self.get_face(face_id)
        #           face.set_points_as_corners(
        #               [
        #                   triangle[0],
        #                   triangle[1],
        #                   triangle[2],
        #                   triangle[0]
        #               ],
        #           )
        raise NotImplementedError

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
