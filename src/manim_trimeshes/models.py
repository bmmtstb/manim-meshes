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


# another Mesh implementation, a little bit faster + looks better (but has no vertex dots, even necessary...?)
# inspired by class 'Surface'
class ManimMesh(VGroup, metaclass=ConvertToOpenGL):

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
        """set the current mesh up as manim obejcts"""
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
