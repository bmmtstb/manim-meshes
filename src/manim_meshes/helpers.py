"""
some basic helpers
"""
# python imports
from typing import Any, Dict, List, Tuple, Union
# third-party imports
import numpy as np

from manim_meshes.types import VarArray


def is_vararray_equal(l1: VarArray, l2: VarArray) -> bool:
    """check whether l1 contains l2 and l2 contains l1 resulting in whether they are equal"""
    return all(any(np.array_equal(f1, x) for x in l1) for f1 in l2) and \
           all(any(np.array_equal(f2, x) for x in l2) for f2 in l1)


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
