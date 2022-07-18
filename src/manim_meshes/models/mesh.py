"""
Mesh structure
"""
# third-party imports
import warnings
from typing import List, Union

import numpy as np

# we need to have support for a list that contains different sizes of arrays, because objects may
# contain e.g. triangles and squares
from manim_meshes.exceptions import InvalidMeshException
from manim_meshes.helpers import is_vararray_equal, fix_references, is_twice_nested_iterable
from manim_meshes.types import Vertex, Vertices, Face, Faces, Part, Parts


class Mesh:
    """
    Basic mesh structure based on vertices and faces. For 3D objects parts is additionally used.
    """

    def __init__(self, verts, faces, parts=None):
        """
        Initialize mesh to have correct structure for all variables
        :param verts: array-like list vertices, all with the same dimensions
        :type verts: Array-like
        :param faces: a list of vertex ids that form a face as lists
        :type faces: Array-like or None
        :param parts: list of face ids that form a 3D object as list
        :type parts: Array-like or None
        """
        # check verts, faces and parts for correct types
        if faces is not None and not is_twice_nested_iterable(faces):
            raise InvalidMeshException("Faces have to be twice nested enumerates.")
        if parts is not None and faces is None:
            raise InvalidMeshException("Parts can not be defined while faces is None.")
        if parts is not None and not is_twice_nested_iterable(parts):
            raise InvalidMeshException("Parts have to be twice nested enumerates or None.")

        # indirectly check vertices
        try:
            conv_vertices = np.array(verts, dtype=float)
            # if array or inner array could not be broadcast, one of the exceptions should be raised
            if len(conv_vertices.shape) != 2:
                raise InvalidMeshException("Could not broadcast to array. Dimensional mismatch for vertices. "
                                           "All vertices should have the same number of dimensions.")
        except (np.VisibleDeprecationWarning, TypeError, ValueError) as e:
            raise InvalidMeshException("Dimensional mismatch for vertices. All vertices should have the same number of "
                                       "dimensions.") from e
        # set class variables
        self._vertices: Vertices = conv_vertices
        self._faces: Faces = [np.array(f, dtype=int) for f in faces] if faces is not None else []
        self._parts: Parts = [np.array(p, dtype=int) for p in parts] if parts is not None else []

        # warn user for creating dangling meshes
        if faces is not None and self.dangling_vert_check():
            warnings.warn('Mesh contains dangling vertices.')
        if parts is not None and faces is not None and self.dangling_face_check():
            warnings.warn('Mesh contains dangling faces.')

    def __iadd__(self, other: 'Mesh') -> None:
        if isinstance(other, Mesh):
            self.add_to_mesh(other)
        else:
            raise NotImplementedError

    def __add__(self, other: 'Mesh') -> 'Mesh':
        if isinstance(other, Mesh):
            self.add_to_mesh(other)
            return self
        raise NotImplementedError

    def __eq__(self, other: 'Mesh') -> bool:
        # check for Mesh instance, everything else is not defined
        if isinstance(other, Mesh):
            if is_vararray_equal(self.get_faces(), other.get_faces()) and \
                    is_vararray_equal(self.get_parts(), other.get_parts()) and \
                    np.array_equal(self.get_vertices(), other.get_vertices()):
                return True
            return False
        raise InvalidMeshException(f'Not equal is not defined for mesh and {type(other)}')

    def __ne__(self, other: 'Mesh') -> bool:
        """Overrides the default implementation (kind of unnecessary in Python 3)"""
        if isinstance(other, Mesh):
            return not self.__eq__(other)
        raise InvalidMeshException(f'Not equal is not defined for mesh and {type(other)}')

    def get_vertices(self) -> Vertices:
        return self._vertices

    def get_faces(self) -> Faces:
        return self._faces

    def get_parts(self) -> Parts:
        return self._parts

    def remove_vertices(self, indices: Union[np.ndarray, List[int]]) -> None:
        """remove multiple vertices"""
        indices = list(set(indices))
        indices.sort(reverse=True)
        raise NotImplementedError

    def update_vertex(self, idx: int, new_vert: Vertex) -> None:
        """update the position of the vertex at given index"""
        if len(self._vertices) <= idx:
            raise IndexError(f'Vertex index {idx} out of range for vertices of length {len(self._vertices)}')
        if self._vertices.shape[1] != new_vert.shape[0]:
            raise InvalidMeshException(f'Current indices have dimension {self._vertices.shape[1]}, while'
                                       f'new vertex has dimension {new_vert.shape[0]} .')
        try:
            self._vertices[idx] = np.array(new_vert)
        except (np.VisibleDeprecationWarning, TypeError) as e:
            raise InvalidMeshException("Could not update Vertex. Dimensional mismatch for vertices."
                                       "All vertices should have the same number of dimensions.") from e

    def add_faces(self, new_faces: Faces) -> None:
        """adds new faces"""
        if not is_twice_nested_iterable(new_faces):
            raise InvalidMeshException("new_faces should be twice nested iterable.")

        for new_face in new_faces:
            if any(v >= len(self._vertices) for v in new_face):
                raise InvalidMeshException('Vertex index not defined')

        if isinstance(new_faces, list):
            self._faces += new_faces
        elif isinstance(new_faces, (np.ndarray, tuple)):
            for val in new_faces:
                self._faces.append(val)
        else:
            raise TypeError(f'unknown type for new_face {type(new_faces)}')

    def remove_faces(self, indices: Union[np.ndarray, List[int]]) -> None:
        """removes the faces with given indices and clean-up parts"""
        if any(len(self._faces) <= idx for idx in indices):
            raise IndexError('Face index out of range')

        # use indices to update self._parts
        fix_references(self._parts, indices)
        # remove faces at all the indices
        for index in indices:
            del self._faces[index]

        # warn on dangling vertices and faces
        if self.dangling_vert_check():
            warnings.warn('Dangling vertices due to face removal')
        if self.dangling_face_check():
            warnings.warn('Dangling faces due to removed parts by face removal')

    def update_face(self, idx: int, new_face: Face) -> None:
        """update face with index idx to take new vertices"""
        if len(self._faces) <= idx:
            raise IndexError(f'Face index {idx} out of range.')
        if any(v_idx >= len(self._vertices) for v_idx in new_face):
            raise IndexError('Vertex index out of range.')
        self._faces[idx] = np.array(new_face)
        if self.dangling_vert_check():
            warnings.warn('Dangling vertices due to face update')

    def add_parts(self, new_parts: Parts) -> None:
        """adds new parts"""
        for new_part in new_parts:
            if any(f >= len(self._faces) for f in new_part):
                raise InvalidMeshException('Face index not defined')
        self._parts = np.vstack((self._parts, np.narray(new_parts)))

    def remove_parts(self, indices: Union[np.ndarray, List[int]]) -> None:
        """removes the parts with given indices"""
        if any(len(self._parts) <= idx for idx in indices):
            raise IndexError('Part index out of range')
        # remove indices back to front
        indices = list(set(indices))
        indices.sort(reverse=True)
        for idx in indices:
            del self._parts[idx]
        # post remove
        if self.dangling_face_check():
            warnings.warn('Dangling faces due to part removal')

    def update_part(self, idx: int, new_part: Part) -> None:
        """update part with index idx to take new faces"""
        if len(self._parts) <= idx:
            raise IndexError(f'Part index {idx} out of range.')
        if any(f_idx > len(self._faces) for f_idx in new_part):
            raise IndexError('Face index out of range.')
        self._parts[idx] = np.array(new_part)
        if self.dangling_face_check():
            warnings.warn('Dangling faces due to part update')

    def add_to_mesh(self, other: 'Mesh') -> None:
        """
        add another mesh to current mesh
        To get consistent indices of the new mesh, the indices of other will be shifted by len(self.vertices).
        Therefore, if you want to use the old indices, either subtract the len or use negative indices to start with.
        """
        # check if vertices have the same dimension
        if self._vertices.shape[1] != other.get_vertices().shape[1]:
            raise InvalidMeshException("Can not concatenate meshes with vertices of different dimensionality.")

        # append vertices, faces and parts
        self._vertices = np.vstack((self._vertices, other.get_vertices()))
        # shift faces indices by the amount of vertices of the current mesh and append them
        self._faces += [face + len(self._vertices) for face in other.get_faces()]
        # shift parts indices by the amount of faces of the current mesh and append them
        self._parts += [part + len(self._faces) for part in other.get_parts()]

    def dangling_vert_check(self) -> bool:
        """check whether there are any dangling nodes - vertices that are not part of a face"""
        if len(self._faces) == 0:
            return len(self._vertices) != 0
        unique = np.unique(np.concatenate(self._faces).ravel())
        return any(v_idx not in unique for v_idx in range(len(self._vertices)))

    def dangling_face_check(self) -> bool:
        """check whether there are any dangling faces - faces that are not part of a part"""
        if len(self._parts) == 0:
            return len(self._faces) != 0
        if len(self._parts) > 0 and len(self._faces) == 0:
            return False
        unique = np.unique(np.concatenate(self._parts).ravel())
        return any(f_idx not in unique for f_idx in range(len(self._faces)))

    def is_face_ccw(self, face_id: int) -> bool:
        """check if face is counter-clockwise"""
        # FIXME is this even possible to check without going through the whole mesh?
        raise NotImplementedError

    def is_mesh_ccw(self) -> bool:
        """check full mesh """
        # FIXME see: is_face_ccw
        return all(self.is_face_ccw(f_id) for f_id in range(len(self._faces)))

    def scale_mesh(self, scaling: float) -> None:
        """scales all vertices """
        self._vertices *= float(scaling)

    def translate_mesh(self, translation: np.ndarray) -> None:
        """translates all vertices"""
        self._vertices += np.array(translation)

    def translate_vertex(self, v_id: int, translation: np.ndarray) -> None:
        """translate a single vertex"""
        self._vertices[v_id] += translation

    def apply_rotation(self, angle, axis) -> None:
        """
        rotates all vertices around an axis
        :param angle: rotation angle in radians
        :param axis: rotation axis, x (axis=0), y (axis=1) or z (axis=2)
        """
        rotation_matrix = np.eye(3)
        tmp = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        if axis == 0:
            rotation_matrix[1:, 1:] = tmp
        elif axis == 1:
            rotation_matrix[2::-2, 2::-2] = tmp
        elif axis == 2:
            rotation_matrix[:-1, :-1] = tmp
        else:
            raise ValueError('Parameter \'axis\' must be 0, 1 or 2')
        self._vertices = self._vertices @ rotation_matrix.T
