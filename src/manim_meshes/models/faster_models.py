"""
faster meshes (work in progress)
"""
# FIXME: actually please the linter by correctly implementing everything
# pylint: skip-file
# pylint: disable-all

import numpy as np
import manim as m
import moderngl
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim_meshes.models.mesh import Mesh
from manim_meshes.templates import create_model


class FastManimMesh(OpenGLMobject):
    """ More efficient mesh implementation.
        Uses custom shaders and stores vertices and faces in a single VAO

        NOTE: requires to manipulate the manim lib
        -> copy directory 'mesh' (under manim_meshes/shaders/) to manim/renderer/shaders/
    """
    shader_dtype = [
        ("point", np.float32, (3,)),
        ("color", np.float32, (4,)),
    ]
    shader_folder = "mesh"

    def __init__(
            self,
            mesh,
            color=m.GREY,
            opacity=1.0,
            gloss=0.3,
            shadow=0.4,
            render_primitive=moderngl.TRIANGLES,
            depth_test=True,
            shader_folder=None,
            **kwargs,
    ):
        self.mesh = mesh
        super().__init__(
            color=color,
            opacity=opacity,
            gloss=gloss,
            shadow=shadow,
            shader_folder=shader_folder if shader_folder is not None else "mesh",
            render_primitive=render_primitive,
            depth_test=depth_test,
            **kwargs,
        )
        self.triangle_indices = np.hstack(mesh.get_faces())

    def init_points(self):
        self.set_points(self.mesh.get_vertices())

    def get_triangle_indices(self):
        return self.triangle_indices

    # For shaders
    def get_shader_data(self):
        shader_data = np.zeros(len(self.points), dtype=self.shader_dtype)
        if "points" not in self.locked_data_keys:
            shader_data["point"] = self.points
        self.fill_in_shader_color_info(shader_data)
        return shader_data

    def fill_in_shader_color_info(self, shader_data):
        self.read_data_to_shader(shader_data, "color", "rgbas")
        return shader_data

    def get_shader_vert_indices(self):
        return self.get_triangle_indices()


class VManimMesh(m.ThreeDVMobject):
    """ Mesh implementation which renders the whole mesh as a single
        vectorized mobject (instead of one vmobject per face, see ManimMesh)

        Slower than FastManimMesh, faster than ManimMesh.

        NOTE: has some artefacts (random looking lines across the image), probably
        due to some floating point issues in the stroke shader (stroke_opacity=0 -> no issues)
    """
    def __init__(
            self,
            mesh: Mesh,
            *args,
            fill_color=m.BLUE,
            fill_opacity=0.3,
            stroke_color=m.BLUE,
            stroke_opacity=1,
            stroke_width=0.1,
            **kwargs,
    ) -> None:
        super().__init__(*args, fill_color=fill_color, fill_opacity=fill_opacity,
                         stroke_color=stroke_color, stroke_opacity=stroke_opacity,
                         stroke_width=stroke_width, **kwargs)
        self.mesh: Mesh = mesh
        self._setup_faces()

    def _setup_faces(self):
        for face_indices in self.mesh.get_faces():
            triangle = [self.mesh.get_vertices()[i] for i in face_indices]
            self.start_new_path(triangle[0])
            self.add_points_as_corners(
                np.array([
                    triangle[1],
                    triangle[2],
                    triangle[0]
                ])
            )

    # overrides method from OpenGLVMobject
    def get_triangulation(self):
        if not self.needs_new_triangulation:
            return self.triangulation

        points = self.points

        if len(points) <= 1:
            self.triangulation = np.zeros(0, dtype="i4")
            self.needs_new_triangulation = False
            return self.triangulation

        indices = np.arange(len(points), dtype=int)
        inner_tri_indices = indices[0::3]

        tri_indices = np.hstack([indices, inner_tri_indices])
        self.triangulation = tri_indices
        self.needs_new_triangulation = False
        return tri_indices

# some test scenes

# preview: manim --renderer=opengl -p faster_models.py FastMeshTest
class FastMeshTest(m.ThreeDScene):
    """render FastManimMesh"""

    def construct(self):
        self.camera.set_phi(90 * m.DEGREES)
        mesh = create_model(name="armadillo")
        mesh.apply_rotation(90 * m.DEGREES, 0)
        mesh.apply_scale(0.03)
        manim_mesh_obj = FastManimMesh(mesh=mesh)
        self.add(manim_mesh_obj)
        self.play(
            m.Rotate(
                manim_mesh_obj,
                angle=2 * m.PI,
                about_point=m.ORIGIN,
                rate_func=m.linear,
                run_time=5
            )
        )

# preview: manim --renderer=opengl -p faster_models.py VManimMeshTest
class VManimMeshTest(m.ThreeDScene):
    """render VManimMesh"""

    def construct(self):
        self.camera.set_phi(90 * m.DEGREES)
        mesh = create_model(name="squirrel")
        mesh.apply_scale(0.1)
        mesh.apply_translation(np.array([0,0,-2.2]))
        obj = VManimMesh(mesh)
        self.add(obj)
        self.play(
            m.Rotate(
                obj,
                angle=2 * m.PI,
                about_point=m.ORIGIN,
                rate_func=m.linear,
                run_time=5
            )
        )
