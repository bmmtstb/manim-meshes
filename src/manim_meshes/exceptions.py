"""
custom exceptions
"""


class InvalidMeshException(Exception):
    """something with the mesh is wrong"""


class InvalidMeshDimensionsException(Exception):
    """Something with the Mesh Dimensions is not as expected"""
    def __init__(self, actual: int, expected: int, name: str = ""):
        if name == "":
            super().__init__(f'Dimensions is expected to be {expected} but was {actual}.')
        else:
            super().__init__(f'Dimensions of {name} is expected to be {expected} but was {actual}.')


class InvalidVertexIndex(Exception):
    """Vertex index is out of range"""
    def __init__(self, index: int, length: int):
        super().__init__(f'Index {index} is out of bounds for vertices of length {length}.')


class InvalidFaceIndex(Exception):
    """Face index is out of range"""
    def __init__(self, index: int, length: int):
        super().__init__(f'Index {index} is out of bounds for faces of length {length}.')


class InvalidPartIndex(Exception):
    """Part index is out of range"""
    def __init__(self, index: int, length: int):
        super().__init__(f'Index {index} is out of bounds for parts of length {length}.')


class InvalidShapeException(Exception):
    """A new parameter has invalid shape"""
    def __init__(self, name: str, actual: int, expected: int):
        super().__init__(f'Size of {name} is expected to be {expected} but was {actual}.')
