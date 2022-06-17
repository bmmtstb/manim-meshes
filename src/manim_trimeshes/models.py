"""
manim models for trimesh objects
"""
# python imports
from typing import *
# third-party imports
import trimesh
from colour import Color
from manim import *
import numpy as np
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL


class TrimeshObject(Polyhedron):
    """
    manim-trimeshes.models.TrimeshObject

    Trimesh Object:
    uses Python trimesh package to create a manim Polyhedron object to be rendered
    """

    def __init__(self, mesh: trimesh.Trimesh, *args, **kwargs):
        # custom parameters
        self.mesh: trimesh.Trimesh = mesh

        # initialize Polyhedron
        super().__init__(
            vertex_coords=mesh.vertices,
            faces_list=mesh.faces,
            *args,
            **kwargs,
        )

    def get_edges(self, *args, **kwargs) -> List[Tuple[int, int]]:
        """
        use trimesh to get edges
        """
        return [(edge[0], edge[1]) for edge in self.mesh.edges]

    def create_faces(
        self,
        face_coords: List[List[List or np.ndarray]],
    ) -> VGroup:
        """Creates VGroup of faces from a list of face coordinates."""
        face_group = VGroup()
        for face in face_coords:
            face_group.add(Polygon(*face, **self.faces_config))
        return face_group

    def update_mesh(self, mesh: trimesh.Trimesh):
        """
        use a new mesh
        """
        self.mesh = mesh
        self.vertex_coords = mesh.vertices
        self.faces_list = mesh.faces
        # TODO reload other params?

    def color_all_faces(self):
        """
        color all the faces of the polyhedron, make sure no neighboring faces have the same color
        """
        raise NotImplementedError


class PointCloudObject(Group):
    """use a mesh to display a point-cloud"""
    def __init__(self, mesh: trimesh.Trimesh, *args, **kwargs):
        self.mesh: trimesh.Trimesh = mesh
        self.mesh_points = \
            [Point(p, color=BLUE, stroke_width=2, **kwargs) for p in self.mesh.vertices]
        super().__init__(*self.mesh_points, *args, **kwargs)


    def get_point(self, point_idx):
        return self.mesh_points[point_idx]

    def align_points_with_larger(self, larger_mobject):
        pass


class ManimMesh(VGroup, metaclass=ConvertToOpenGL):
    """
    another Mesh implementation, a little bit faster + looks better
    -> FIXME has no vertex dots, necessary?

    inspired by manim class 'Surface'
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        fill_color: Color = BLUE_D,
        fill_opacity: float = 0.4,
        stroke_color: Color = LIGHT_GREY,
        stroke_width: float = 0.3,
        pre_function_handle_to_anchor_scale_factor: float = 0.00001,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mesh: trimesh.Trimesh = mesh
        self.fill_color = fill_color
        self.fill_opacity = fill_opacity
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.pre_function_handle_to_anchor_scale_factor = (
            pre_function_handle_to_anchor_scale_factor
        )
        self._setup()

    def _setup(self):
        """set the current mesh up as manim objects"""
        faces = VGroup()
        for f in self.mesh.faces:
            triangle = [self.mesh.vertices[i] for i in f]
            face = ThreeDVMobject()
            face.set_points_as_corners(
                [
                    triangle[0],
                    triangle[1],
                    triangle[2],
                    triangle[0]
                ],
            )
            faces.add(face)
        faces.set_fill(color=self.fill_color, opacity=self.fill_opacity)
        faces.set_stroke(
            color=self.stroke_color,
            width=self.stroke_width,
            opacity=self.stroke_opacity,
        )
        self.add(*faces)

    def get_face(self, face_idx):
        """get the faces with the given id"""
        return self.submobjects[face_idx]


# TODO: move to "Triangle-class"
def get_triangle_circum_circle_params(
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


class Manim2DMesh(ManimMesh):
    """
    TODO
    """
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        fill_color: Color = BLUE_D,
        fill_opacity: float = 0.4,
        stroke_color: Color = LIGHT_GREY,
        stroke_width: float = 0.3,
        pre_function_handle_to_anchor_scale_factor: float = 0.00001,
        **kwargs,
    ) -> None:
        if len(mesh.facets[0]) < len(mesh.faces):
            raise Exception('Mesh is not 2D!')
        # Todo: rotate mesh if mesh not already in z-plane
        super().__init__(
            mesh,
            fill_color,
            fill_opacity,
            stroke_color,
            stroke_width,
            pre_function_handle_to_anchor_scale_factor,
            **kwargs,
        )

    def get_circle(self, face_idx: int):
        face = self.mesh.faces[face_idx]
        vertices = [self.mesh.vertices[i] for i in face]
        center, radius = get_triangle_circum_circle_params(*vertices)
        circ = Circle(radius, stroke_width=2)
        circ.move_to(center)
        return circ

    def get_points_violating_delaunay(self, face_idx_1: int):
        """given a triangle by id, get all points violating delaunay criterion"""
        points = []
        face_1 = self.mesh.faces[face_idx_1]
        center, radius = get_triangle_circum_circle_params(*[self.mesh.vertices[i] for i in face_1])
        # TODO: don't loop all vertices, only loop ones that are "close"
        for _, point in enumerate(self.mesh.vertices):
            point = np.asarray(point)
            if point not in face_1:
                distance = np.linalg.norm(center - point)
                if distance < radius:  # inside circle
                    points.append(Dot(point, radius=0.03, color=RED))
        return points
