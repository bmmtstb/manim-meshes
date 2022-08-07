"""
some basic helpers
"""
# python imports
from typing import Any, Dict, List, Tuple, Union
# third-party imports
import numpy as np

from manim_meshes.types import Edges, VarArray


def is_in_vararray(array: VarArray, item: np.ndarray, rolling: bool = True) -> bool:
    """
    return whether item is in array, possibility to check for rolling
    """
    if rolling:
        alternatives = [np.roll(item, i) for i in range(len(item))]
        return any(any(np.array_equal(alt, a) for a in array) for alt in alternatives)
    # non rolling
    return any(np.array_equal(item, a) for a in array)


def find_in_vararray(array: VarArray, item: np.ndarray, rolling: bool = True, start: int = 0) -> List[int]:
    """
    return list of indices where array == item or a clockwise rolled / shifted alternative
    possibility to start loop at different index
    """
    if rolling:
        alternatives = [np.roll(item, i) for i in range(len(item))]
        return [idx for idx, curr_item in enumerate(array[start:], start=start)
                if any(np.array_equal(a, curr_item) for a in alternatives)]
    # non rolling
    return [idx for idx, curr_item in enumerate(array[start:], start=start) if np.array_equal(curr_item, item)]


def is_vararray_equal(array1: VarArray, array2: VarArray, rolling: bool = True) -> bool:
    """
    check whether l1 contains l2 and l2 contains l1 resulting in whether they are equal
    possibility to additionally check with rolling
    """
    return all(is_in_vararray(array=array1, item=value2, rolling=rolling) for value2 in array2) and\
           all(is_in_vararray(array=array2, item=value1, rolling=rolling) for value1 in array1)


def is_twice_nested_iterable(obj: Any, min_lens: Tuple[int, int] = (1, 3)) -> bool:
    """
    check whether obj is Array-Like exactly twice
    -> e.g. 2-dim np.ndarray, List[List[int|float]] or similar
    :params obj: Array-Like object to be checked
    :params lens: min lengths that the respective layer has to have, default 1 object at least 3 sub-objects
    """
    # easy case np.ndarray with correct specs
    if isinstance(obj, np.ndarray) and len(obj.shape) == 2:
        return obj.shape[0] >= min_lens[0] and obj.shape[1] >= min_lens[1]
    if obj in ([], ()):
        return True

    if isinstance(obj, (list, tuple, np.ndarray)) and \
            len(obj) >= min_lens[0]:
        # obj is iterable
        return all(
            # either list / tuple with values inside
            # or np.ndarray with shape length 1
            ((isinstance(sub_obj, (list, tuple)) and all(isinstance(v, (int, float)) for v in sub_obj)) or
             (isinstance(sub_obj, np.ndarray) and len(sub_obj.shape) == 1)) and \
            len(sub_obj) >= min_lens[1] for sub_obj in obj
        )

    return False


def are_edges_equal(edges1: Edges, edges2: Edges) -> bool:
    """
    check if two lists of edges are equal, no rolling, order does not matter
    -> currently every edge is sorted
    """
    return all(e1 in edges2 for e1 in edges1) and all(e2 in edges1 for e2 in edges2)


def fix_references(original: VarArray, indices: Union[np.ndarray, List[int]]) -> List[int]:
    """
    given a VarArray remove all sub arrays that reference an index of indices
    then shift all the indices after removed indices to link to the correct indices
    original and indices is edited in place (not returned) for higher speed on larger meshes

    :returns: a list of indices where original was modified
    """
    # reverse sort indices in place to delete back to front and change given indices accordingly
    indices[:] = list(set(indices))
    indices.sort(reverse=True)

    # get list of indices where original references removed indices
    sub_removed = []
    for i, part in enumerate(original):
        if any(idx in part for idx in indices):
            sub_removed.append(i)

    # delete all parts that contain at least one of the indices using the precomputed list
    # make sure to delete back to front
    sub_removed.sort(reverse=True)
    for s_r_index in sub_removed:
        del original[s_r_index]

    # change indices of all sub_objects due to parent-deletions in place!
    # (indices is reversely sorted)
    for idx in indices:
        original[:] = [np.subtract(arr, 1, out=arr, where=arr > idx) for arr in original]

    return sub_removed


def remove_keys_from_dict(d: dict, keys: List[str]) -> Dict[str, Any]:
    """given a dictionary remove all keys if they are present"""
    for key in keys:
        try:
            del d[key]
        except KeyError:
            pass
    return d if d is not None else {}
