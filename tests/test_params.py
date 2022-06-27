"""
test for params.py
"""
# python imports
from typing import Dict
from colour import Color
# third-party imports
import manim as m

# local imports
from manim_meshes.params import BadParameterException, DefaultParameters, get_param_or_default

TEST_PARAMS: DefaultParameters = {
    "bool":   (bool, False),
    "string": (str, "testing"),
    "int":    (int, 42),
    "float":  (float, 3.14159),
    "colour": (Color, Color(m.LIGHT_GREY)),
}


def test_existing_parameter():
    """given an existing parameter"""
    param = {
        "bool": True,
        "int":  69.0,
    }
    assert get_param_or_default("bool", param, TEST_PARAMS)
    assert 69 == get_param_or_default("int", param, TEST_PARAMS)


def test_non_existing_parameter():
    """given an non existing parameter"""
    param = {
        "bool": True,
        "int":  69.0,
    }
    assert "testing" == get_param_or_default("string", param, TEST_PARAMS)
    assert 3.14159 == get_param_or_default("float", param, TEST_PARAMS)


def test_faulty_parameter_dict():
    """given an non existing parameter"""
    param = {
        "faulty": False,
    }
    try:
        get_param_or_default("faulty", param, TEST_PARAMS)
        assert False
    except BadParameterException:
        assert True


def test_faulty_parameter_string():
    """given an non existing parameter"""
    param = {}
    try:
        get_param_or_default("faulty", param, TEST_PARAMS)
        assert False
    except BadParameterException:
        assert True


def test_colors():
    """test color functions including casting"""
    param = {
        "colour": m.BLUE_D
    }
    assert Color(m.BLUE_D) == get_param_or_default("colour", param, TEST_PARAMS)
    param = {}
    assert Color(m.LIGHT_GREY) == get_param_or_default("colour", param, TEST_PARAMS)
