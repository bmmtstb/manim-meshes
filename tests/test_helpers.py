"""
test the test helpers and the manim_meshes helpers
"""
# third-party imports
import pytest
import numpy as np
# local imports
from manim_meshes.helpers import find_in_vararray, is_in_vararray, is_vararray_equal, is_twice_nested_iterable, \
    fix_references

vararrays = [
    ([], [], True),  # empty
    ([np.array([0, 1, 2])], [np.arange(3)], True),  # differently initialised still same
    ([np.array([0, 1, 2]), np.ones(1)], [np.arange(3)], False),  # unequal matching in both directions
    ([np.array([0, 1, 2])], [np.arange(3), np.ones(1)], False),  # unequal matching in both directions
    ([np.array([2, 1, 0])], [np.arange(3)], False),  # FIXME: how do we decide this case?
    ([np.arange(3), np.ones(3)], [np.array([1, 1, 1]), np.array([0, 1, 2])], True),  # sorting does not matter
]


@pytest.mark.parametrize("vararrs", vararrays)
def test_is_vararray_equal(vararrs):
    obj_1, obj_2, b = vararrs
    assert b == is_vararray_equal(obj_1, obj_2)


objs = [
    ([], True),
    ((), True),
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


references = [
    # pre_obj, post_obj, indices to remove, return value
    ([np.array([0, 1, 2])], [], [2], [0]),  # remove all of single
    ([np.array([0, 1, 3]), np.array([0, 1, 2])], [], [2, 3], [1, 0]),  # remove all of multiple
    ([np.array([0, 1, 3]), np.array([0, 1, 2])], [np.array([0, 1, 2])], [3], [0]),  # remove partly, no updates
    ([np.array([0, 1, 3]), np.array([0, 1, 2])], [np.array([0, 1, 2])], [2], [1]),  # remove partly, update reference
    ([np.array([0, 1, 3])], [np.array([0, 1, 3])], [4], []),  # remove nothing, no updates
    ([np.array([2, 3, 4, 5, 6]), np.array([1, 2, 3])],  # remove nothing, update references, more than 3 values
     [np.array([1, 2, 3, 4, 5]), np.array([0, 1, 2])], [0], []),
]


@pytest.mark.parametrize("refs", references)
def test_fix_references(refs):
    pre_obj, post_obj, r_indices, new_indices = refs
    assert set(new_indices) == set(fix_references(pre_obj, r_indices))
    assert is_vararray_equal(pre_obj, post_obj)


vararray = [
    ([], np.arange(3), True, False),  # empty
    ([np.array([0, 1, 2])], np.arange(3), False, True),  # equality
    ([np.array([2, 0, 1])], np.arange(3), False, False),  # no rolling
    ([np.array([2, 0, 1])], np.arange(3), True, True),  # rolling
    ([np.array([2, 1, 0])], np.arange(3), True, False),  # shifted not rolled
]


@pytest.mark.parametrize("vararr", vararray)
def test_is_in_vararray(vararr):
    arr, item, roll, contained = vararr
    assert contained == is_in_vararray(arr, item, rolling=roll)


vararray = [
    ([], np.ones(3), True, []),  # empty
    ([np.array([0, 1, 2])], np.arange(3), False, [0]),  # equality
    ([np.array([2, 0, 1])], np.arange(3), False, []),  # no rolling
    ([
         np.array([0, 1, 2]),
         np.array([1, 2, 0]),
         np.array([0, 0, 0]),
         np.array([2, 0, 1])],
     np.arange(3), True, [0, 1, 3]),  # rolling
    ([np.array([2, 1, 0])], np.arange(3), True, []),  # shifted not rolled
]


@pytest.mark.parametrize("vararr", vararray)
def test_find_in_vararray(vararr):
    arr, item, roll, contained = vararr
    assert contained == find_in_vararray(arr, item, rolling=roll)
