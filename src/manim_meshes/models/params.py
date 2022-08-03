"""
Parameters get out of hand for the meshes, store defaults and casting in separate functions
"""
# python imports
from typing import Any, Dict, Tuple
from colour import Color
# third-party imports
import manim as m

Parameters = Dict[str, Any]
DefaultParameters = Dict[str, Tuple[type, Any]]


class BadParameterException(Exception):
    """Default Class for Parameter Exceptions"""


# map from param name to type and default value
# manim_3d_mesh_default_params
M3DM: DefaultParameters = {
    "display_vertices":                           (bool, False),
    "display_edges":                              (bool, True),
    "display_faces":                              (bool, True),
    "edges_fill_color":                           (Color, Color(m.WHITE)),
    "edges_fill_opacity":                         (float, 0.4),
    "edges_stroke_color":                         (Color, Color(m.WHITE)),
    "edges_stroke_opacity":                       (float, 0.3),
    "edges_stroke_width":                         (float, 0.3),
    "faces_fill_color":                           (Color, Color(m.BLUE_D)),
    "faces_fill_opacity":                         (float, 0.4),
    "faces_stroke_color":                         (Color, Color(m.LIGHT_GREY)),
    "faces_stroke_opacity":                       (float, 0.3),
    "faces_stroke_width":                         (float, 0.3),
    "verts_fill_color":                           (Color, Color(m.GREEN)),
    "verts_fill_opacity":                         (float, 0.4),
    "verts_stroke_color":                         (Color, Color(m.GREEN)),
    "verts_stroke_opacity":                       (float, 1.0),
    "verts_stroke_width":                         (float, 1.0),
    "pre_function_handle_to_anchor_scale_factor": (float, 0.00001),
}

# manim_2d_mesh_default_params
M2DM: DefaultParameters = {
    "display_vertices":                           (bool, False),
    "display_edges":                              (bool, True),
    "display_faces":                              (bool, True),
    "faces_fill_color":                           (Color, Color(m.BLUE_E)),
    "faces_fill_opacity":                         (float, 1),
    "faces_stroke_color":                         (Color, Color(m.BLUE_E)),
    "faces_stroke_width":                         (float, 0.0),
    "edges_fill_color":                           (Color, Color(m.LIGHT_GREY)),
    "edges_fill_opacity":                         (float, 0),
    "edges_stroke_color":                         (Color, Color(m.LIGHT_GREY)),
    "edges_stroke_opacity":                       (float, 1.0),
    "edges_stroke_width":                         (float, 2),
    "pre_function_handle_to_anchor_scale_factor": (float, 0.00001),
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
    raise BadParameterException(f'Value {value} is not in params and not in default params.')
