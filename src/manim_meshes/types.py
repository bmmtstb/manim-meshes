"""
custom type hints
"""
# python imports
from typing import List, Union, Tuple
# third-party imports
import numpy as np

VarArray = List[np.ndarray]

Vertex = Union[np.ndarray, List[int], Tuple[int, ...]]
Vertices = np.ndarray

Face = Union[np.ndarray, List[int], Tuple[int, ...]]
Faces = VarArray

Part = Union[np.ndarray, List[int], Tuple[int, ...]]
Parts = VarArray

Edge = Tuple[int, ...]
Edges = List[Edge]
