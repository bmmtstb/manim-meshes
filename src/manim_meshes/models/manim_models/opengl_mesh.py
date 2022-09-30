"""
faster meshes (WORK IN PROGRESS) by using OpenGL more efficiently
"""
# FIXME: actually please the linter by correctly implementing everything
# pylint: skip-file
# pylint: disable-all

import numpy as np
import manim as m
from manim.mobject.opengl.opengl_mobject import OpenGLMobject

from manim_meshes.helpers import remove_keys_from_dict
from manim_meshes.models.data_models.mesh import Mesh
from manim_meshes.params import get_param_or_default, OGLM
from manim_meshes.templates import create_model


class FastManimMesh(OpenGLMobject):
    """ More efficient mesh implementation.
        Uses custom shaders and stores vertices and faces in a single VAO
        Useful to render meshes with many vertices / faces. Currently only supports displaying the mesh,
        no mesh manipulations.

        NOTE: requires to manipulate the manim lib
        -> copy directory 'mesh' (under manim_meshes/shaders/) to manim/renderer/shaders/

        HINT: the mesh must only consist of triangles
    """

    shader_dtype = [
        ("point", np.float32, (3,)),
        ("color", np.float32, (4,)),
    ]
    shader_folder = "mesh"

    def __init__(
            self,
            mesh: Mesh,
            shader_folder=None,
            **kwargs,
    ):
        """ Initialization. mesh must only consist of triangles """
        if any(len(face) != 3 for face in mesh.faces):
            raise ValueError("mesh must only consist of triangles!")
        self.mesh = mesh
        super().__init__(
            shader_folder=shader_folder if shader_folder is not None else "mesh",
            # default params
            **{key: get_param_or_default(key, kwargs, OGLM) for key in OGLM},
            # regular kwargs
            **remove_keys_from_dict(kwargs, list(OGLM.keys())),
        )
        self.triangle_indices = np.hstack(mesh.faces)

    def init_points(self) -> None:
        self.set_points(self.mesh.vertices)

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

