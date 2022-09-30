"""
Parameters can get out of hand for the meshes, store defaults and casting in separate functions
"""
# python imports
from typing import Any
# third-party imports
from colour import Color
import manim as m
import moderngl
# local imports
from manim_meshes.exceptions import BadParameterException
from manim_meshes.types import DefaultParameters, Parameters

# map from param name to type and default value

# basic_manim_3d_mesh_default_params
BM3DM: DefaultParameters = {
    "display_vertices":                           (bool, False),
    "display_edges":                              (bool, True),
    "display_faces":                              (bool, True),
    "clear_vertices":                             (bool, True),
    "clear_edges":                                (bool, True),
    "clear_faces":                                (bool, True),
    "edges_color":                                (Color, Color(m.BLUE)),
    "edges_width":                                (float, 0.1),
    "faces_color":                                (Color, Color(m.BLUE_D)),
    "faces_opacity":                              (float, 0.4),
    "verts_color":                                (Color, Color(m.GREEN)),
    "verts_size":                                 (float, 0.04),
    "pre_function_handle_to_anchor_scale_factor": (float, 0.00001),
}

# basic_manim_2d_mesh_default_params
BM2DM: DefaultParameters = {
    "display_vertices":                           (bool, False),
    "display_edges":                              (bool, True),
    "display_faces":                              (bool, True),
    "clear_vertices":                             (bool, True),
    "clear_edges":                                (bool, True),
    "clear_faces":                                (bool, True),
    "edges_color":                                (Color, Color(m.LIGHT_GREY)),
    "edges_width":                                (float, 1.5),
    "faces_color":                                (Color, Color(m.BLUE_E)),
    "faces_opacity":                              (float, 1.),
    "verts_color":                                (Color, Color(m.GREEN)),
    "verts_size":                                 (float, 0.02),
    "pre_function_handle_to_anchor_scale_factor": (float, 0.00001),
}

# opengl_mesh_default_params
OGLM: DefaultParameters = {
    "color": m.GREY,
    "depth_test": True,
    "gloss": 0.3,
    "opacity": 1.0,
    "render_primitive": moderngl.TRIANGLES,
    "shadow": 0.4,
}


def get_param_or_default(
        value: str,
        params: Parameters,
        default: DefaultParameters
) -> Any:
    """get value from params or get default value"""
    # get value from user given parameters
    if params and value in params:
        if value in default:
            if issubclass(type(params[value]), default[value][0]) or \
                    isinstance(type(params[value]), default[value][0]):
                return params[value]
            try:
                return default[value][0](params[value])
            except (ValueError, TypeError) as e:
                raise BadParameterException(f'Value {value} does not have correct type '
                                            f'{default[value][0]} and can not be cast.') from e
        raise BadParameterException(f'Value {value} is not an expected default value.')
    # get value from default parameters
    if value in default:
        return default[value][1]
    # should this be raised?
    raise BadParameterException(f'Value {value} is not in params and not in default params.')
