"""
test the test helpers and the manim_meshes helpers
"""
# third-party imports
import pytest
import numpy as np
# local imports
from manim_meshes.helpers import is_twice_nested_iterable

# TODO test assert_exception decorator


objs = [
    (np.ones((4, 3)), True),
    (np.ones((1, 3)), True),
    (np.ones((10, 1)), False),
    (np.ones((1, 3, 1)), False),
    ([[1, 2, 3]], True),
    ([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], False),
    ([[1], [2], [3]], False),
    ([{"a": 1, "b": 2, "c": 3}], False),
    ([np.ones(3)], True),
    (tuple([tuple([1, 2, 3])]), True),
]


@pytest.mark.parametrize("item", objs)
def test_is_twice_nested_iterable(item):
    obj, b = item
    assert b == is_twice_nested_iterable(obj)
