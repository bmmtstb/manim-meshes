"""
Mesh structure
"""
# third-party imports
import warnings
import numpy as np

# we need to have support for a list that contains different sizes of arrays, because objects may
# contain e.g. triangles and squares
from manim_meshes.exceptions import InvalidMeshException
from manim_meshes.helpers import is_twice_nested_iterable
from manim_meshes.types import VarArray


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
        :type faces: Array-like
        :param parts: list of face ids that form a 3D object as list
        :type parts: Array-like
        """
        # check verts, faces and parts for correct types
        if not is_twice_nested_iterable(faces):
            raise InvalidMeshException("Faces have to be twice nested enumerates.")
        if parts is not None and not is_twice_nested_iterable(parts):
            raise InvalidMeshException("Parts have to be twice nested enumerates or None.")

        # indirectly check vertices
        try:
            conv_vertices = np.array(verts)
            # if array or inner array could not be broadcast, one of the exceptions should be raised
            if len(conv_vertices.shape) != 2:
                raise InvalidMeshException("Could not broadcast to array. Dimensional mismatch for vertices. "
                                           "All vertices should have the same number of dimensions.")
        except (np.VisibleDeprecationWarning, TypeError) as e:
            raise InvalidMeshException("Dimensional mismatch for vertices. All vertices should have the same number of "
                                       "dimensions.") from e
        # set class variables
        self._vertices: np.ndarray = conv_vertices
        self._faces: VarArray = [np.array(f) for f in faces]
        self._parts: VarArray = [np.array(p) for p in parts] if parts is not None else []

        # warn user for creating dangling meshes
        if self.dangling_vert_check():
            warnings.warn('Mesh contains dangling vertices.')
        if parts and self.dangling_face_check():
            warnings.warn('Mesh contains dangling faces.')

    def get_vertices(self) -> np.ndarray:
        return self._vertices

    def get_faces(self) -> VarArray:
        return self._faces

    def get_parts(self) -> VarArray:
        return self._parts

    def update_vertex(self, idx: int, new_vert: np.ndarray) -> None:
        """update the position of the vertex at given index"""
        if len(self._vertices) <= idx:
            raise IndexError(f'Vertex index {idx} out of range')
        if self._vertices.shape[1] != new_vert.shape[1]:
            raise InvalidMeshException(f'Current indices have dimension {self._vertices.shape[1]}, while'
                                       f'new vertex has dimension {new_vert.shape[1]} .')
        self._vertices[idx] = np.array(new_vert)

    def add_faces(self, new_faces: VarArray):
        """adds new faces"""
        for new_face in new_faces:
            if any(v >= len(self._vertices) for v in new_face):
                raise InvalidMeshException('Vertex index not defined')
        self._faces = np.vstack((self._faces, np.narray(new_faces)))

    def remove_faces(self, indices: np.ndarray):
        """removes the faces with given indices"""
        if any(len(self._faces) <= idx for idx in indices):
            raise IndexError('Face index out of range')
        self._faces = np.delete(self._faces, indices)
        # delete parts
        mask = np.ones_like(self._parts, dtype=bool)
        for idx in indices:
            mask[np.argwhere(self._parts == idx)[:, 0]] = False
        self._parts = self._parts[mask]
        if self.dangling_vert_check():
            warnings.warn('Dangling vertices due to face removal')

    def update_face(self, idx: int, new_face: np.ndarray) -> None:
        """update face with index idx to take new vertices"""
        if len(self._faces) <= idx:
            raise IndexError(f'Face index {idx} out of range.')
        if any(v_idx >= len(self._vertices) for v_idx in new_face):
            raise IndexError('Vertex index out of range.')
        self._faces[idx] = np.array(new_face)
        if self.dangling_vert_check():
            warnings.warn('Dangling vertices due to face update')

    def add_parts(self, new_parts: np.ndarray):
        """adds new parts"""
        for new_part in new_parts:
            if any(f >= len(self._faces) for f in new_part):
                raise InvalidMeshException('Face index not defined')
        self._parts = np.vstack((self._parts, np.narray(new_parts)))

    def remove_parts(self, indices: np.ndarray):
        """removes the parts with given indices"""
        if any(len(self._parts) <= idx for idx in indices):
            raise IndexError('Part index out of range')
        self._parts = np.delete(self._parts, indices)
        if self.dangling_face_check():
            warnings.warn('Dangling faces due to part removal')

    def update_part(self, idx: int, new_part: np.ndarray) -> None:
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
        # start by shifting others indices
        other._faces += len(self._vertices)
        other._parts += len(self._vertices)
        # append vertices
        self._vertices = np.vstack((self._vertices, other.get_vertices()))

    def dangling_vert_check(self) -> bool:
        """check if there are any dangling nodes - vertices that are not part of a face"""
        unique = np.unique(np.concatenate(self._faces).ravel())
        return any(v_idx not in unique for v_idx in range(len(self._vertices)))

    def dangling_face_check(self) -> bool:
        """check if there are any dangling faces - faces that are not part of a part"""
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

    def apply_scale(self, scaling: float):
        """scales all vertices """
        self._vertices *= float(scaling)

    def apply_translation(self, translation: np.ndarray):
        """translates all vertices"""
        self._vertices += np.array(translation)
