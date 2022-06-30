"""
Mesh structure
"""
# python imports

# third-party imports
import warnings
import numpy as np


class InvalidMeshException(Exception):
    """something with the mesh is wrong"""


class Mesh:
    """
    Basic mesh structure based on vertices and faces. For 3D objects parts is additionally used.
    """

    def __init__(self, verts, faces, parts=None):
        """
        :param verts:
        :param faces:
        :param parts:
        """
        self._vertices: np.ndarray = np.array(verts)
        self._faces: np.ndarray = np.array(faces)
        self._parts: np.ndarray = np.array(parts)
        if self.dangling_check():
            warnings.warn('Mesh contains dangling vertices')
        if parts and self.dangling_face_check():
            warnings.warn('Mesh contains dangling faces')

    def get_vertices(self) -> np.ndarray:
        return self._vertices

    def get_faces(self) -> np.ndarray:
        return self._faces

    def get_parts(self):
        return self._parts

    def update_vertex(self, idx: int, new_vert: np.ndarray) -> None:
        """update the position of the vertex at given index"""
        if len(self._vertices) <= idx:
            raise IndexError(f'Vertex index {idx} out of range')
        self._vertices[idx] = np.array(new_vert)

    def add_faces(self, new_faces: np.ndarray):
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
        if self.dangling_check():
            warnings.warn('Dangling vertices due to face removal')

    def update_face(self, idx: int, new_face: np.ndarray) -> None:
        """update face with index idx to take new vertices"""
        if len(self._faces) <= idx:
            raise IndexError(f'Face index {idx} out of range.')
        if any(v_idx >= len(self._vertices) for v_idx in new_face):
            raise IndexError('Vertex index out of range.')
        self._faces[idx] = np.array(new_face)
        if self.dangling_check():
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

    def dangling_check(self) -> bool:
        """check if there are any dangling nodes - vertices that are not part of a face"""
        unique = np.unique(self._faces)
        return any(v_idx not in unique for v_idx in range(len(self._vertices)))

    def dangling_face_check(self) -> bool:
        """check if there are any dangling faces - faces that are not part of a part"""
        unique = np.unique(self._parts)
        return any(f_idx not in unique for f_idx in range(len(self._faces)))


# *** not even possible to check? ***

    def is_face_ccw(self, face_id: int) -> bool:
        """check if face is counter-clockwise"""
        raise NotImplementedError

    def is_mesh_ccw(self) -> bool:
        """check full mesh """
        return all(self.is_face_ccw(f) for f in self._faces)

    def apply_scale(self, scaling: float):
        """scales all vertices """
        self._vertices *= scaling

    def apply_translation(self, translation: np.ndarray):
        """translates all vertices"""
        self._vertices += np.array(translation)
