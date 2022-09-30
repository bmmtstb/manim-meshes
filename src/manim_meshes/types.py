"""
custom type hints for all the models
"""
# python imports
from typing import List, Union, Tuple, Dict, Any
# third-party imports
import numpy as np

# VarArrays are Lists of possibly different sized np.ndarrays
# due to the possible different sizes, a nested numpy array can not be used
VarArray = List[np.ndarray]

# vertex can be an array or an iterable of ints or floats
Vertex = Union[np.ndarray, List[Union[int, float]], Tuple[Union[int, float], ...]]
# vertices has to be a two-dimensional np.ndarray
Vertices = np.ndarray

# all faces reference vertices by id and therefore have to be integers
Face = Union[np.ndarray, List[int], Tuple[int, ...]]
Faces = VarArray

# all parts reference faces by id and therefore have to be integers
Part = Union[np.ndarray, List[int], Tuple[int, ...]]
Parts = VarArray

# an edge is a line from one vertex_id to another
Edge = Tuple[int, int]
Edges = List[Edge]

Parameters = Dict[str, Any]
DefaultParameters = Dict[str, Tuple[type, Any]]
