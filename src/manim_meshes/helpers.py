"""
some basic helpers
"""
# python imports
from typing import Any, Tuple
# third-party imports
import numpy as np


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
