"""
custom exceptions
"""
# python imports
from typing import Any, Tuple, Union


class InvalidMeshException(Exception):
    """something with the mesh is generally wrong"""


class InvalidRequestException(InvalidMeshException):
    """a request was made that is not defined"""


class MeshIndexException(IndexError):
    """invalid index"""


class InvalidTypeException(TypeError):
    """A mesh function did get a faulty type"""


class InvalidMeshDimensionsException(Exception):
    """Something with the Mesh Dimensions is not as expected"""
    def __init__(self, actual: Union[int, Tuple[Any, Any]], expected: Union[int, Tuple[Any, Any]], name: str = ""):
        if name == "":
            super().__init__(f'Dimensions is expected to be {expected} but was {actual}.')
        else:
            super().__init__(f'Dimensions of {name} is expected to be {expected} but was {actual}.')


class InvalidShapeException(Exception):
    """A new parameter has invalid shape"""
    def __init__(self, name: str, actual: int, expected: int):
        super().__init__(f'Size of {name} is expected to be {expected} but was {actual}.')


class BadParameterException(Exception):
    """Default Class for Parameter Exceptions"""


class FaultyVarArrayException(Exception):
    """The given object is no VarArray"""
