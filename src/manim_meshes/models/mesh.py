"""
Mesh structure
"""
# third-party imports
import warnings
from typing import List, Tuple, Union

import numpy as np

# we need to have support for a list that contains different sizes of arrays, because objects may
# contain e.g. triangles and squares
from manim_meshes.exceptions import InvalidMeshException
from manim_meshes.helpers import is_vararray_equal, fix_references, is_twice_nested_iterable
from manim_meshes.types import Vertex, Vertices, Face, Faces, Part, Parts, VarArray, Edges


class Mesh:
    """
    Basic mesh structure based on vertices and faces. For 3D objects parts is additionally used.
    """

    def __init__(self, verts, faces, parts=None, dangling: bool = False):
        """
        Initialize mesh to have correct structure for all variables
        :param verts: array-like list vertices, all with the same dimensions
        :type verts: Array-like [N x d]
        :param faces: a list of vertex ids that form a face as lists
        :type faces: 2D Array-like or None, can e.g. be list of different sized np.ndarray
        :param parts: list of face ids that form a 3D object as list
        :type parts: 2D Array-like or None, can e.g. be list of different sized np.ndarray
        :param dangling: whether to check regularly for dangling nodes and faces
        :type dangling: bool
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
        self._edges = self.extract_edges()
        self._test_for_dangling = dangling
        self.dim = conv_vertices.shape[1]

        # warn user for creating dangling meshes
        if self._test_for_dangling and faces is not None and self.dangling_vert_check():
            warnings.warn('Mesh contains dangling vertices.')
        if self._test_for_dangling and parts is not None and faces is not None and self.dangling_face_check():
            warnings.warn('Mesh contains dangling faces.')

    def __add__(self, other) -> 'Mesh':
        if isinstance(other, Mesh):
            self.add_to_mesh(other)
            return self
        raise NotImplementedError

    def __iadd__(self, other: 'Mesh') -> 'Mesh':
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

    def make_vertices_3d(self) -> None:
        """transforms currents mesh vertices to be 3D, works if dim is <= 3"""
        if self.dim < 3:
            self._vertices = np.pad(self._vertices, ((0, 0), (0, 3 - self.dim)))
        elif self.dim > 3:
            raise InvalidMeshException(f'Can not Broadcast from {self.dim}-D Mesh to 3D Mesh.')

    def get_faces(self) -> Faces:
        return self._faces

    def get_parts(self) -> Parts:
        return self._parts

    def get_edges(self) -> Edges:
        return self._edges

    def get_edge_index(self, edge):
        """return index of given edge"""
        return self._edges.index(edge)

    def get_vertex_edges(self, vertex_idx):
        """return all edges containing vertex_idx"""
        vertex_edges = []
        for edge in self._edges:
            if vertex_idx in edge:
                vertex_edges.append(edge)
        return vertex_edges

    def find_vertex(self, vertex: np.ndarray, start: int = 0) -> List[int]:
        """
        return list of indices where self._vertices == vertex
        possibility to start loop at different index
        """
        if len(vertex) != self.dim:
            return []
        return [i for i, v in enumerate(self._vertices[start:], start=start)
                if np.array_equal(vertex, v)]

    @staticmethod
    def _find_rolling_alternative(array: VarArray, item: np.ndarray, start: int = 0) -> List[int]:
        """
        return list of indices where array == item or a clockwise rolled / shifted alternative
        possibility to start loop at different index
        """
        alternatives = [np.roll(item, i) for i in range(len(item))]
        return [idx for idx, curr_item in enumerate(array[start:], start=start)
                if any(np.array_equal(a, curr_item) for a in alternatives)]

    def find_face(self, face: np.ndarray, start: int = 0) -> List[int]:
        """return all indices where face is found in self._faces"""
        return self._find_rolling_alternative(array=self._faces, item=face, start=start)

    def find_part(self, part: np.ndarray, start: int = 0) -> List[int]:
        """return all indices where part is found in self._parts"""
        # Fixme: are two parts equal even if the faces are randomly sorted, not clockwise?
        #  [1,2,3] ?=? [1,3,2] or is it "rolling" like faces?
        return self._find_rolling_alternative(array=self._parts, item=part, start=start)

    def add_vertices(self, new_vertices: Vertices) -> None:
        """add given vertices to current ones"""
        if not isinstance(new_vertices, np.ndarray):
            raise InvalidMeshException(f'new_vertices has invalid type {new_vertices}')
        if len(new_vertices.shape) != 2 or new_vertices.shape[1] != self.dim:
            raise InvalidMeshException("new Vertices do not have the same dimensions as current ones.")
        self._vertices = np.vstack((self._vertices, new_vertices))

    def remove_vertices(self, indices: Union[np.ndarray, List[int]]) -> None:
        """remove multiple vertices"""
        if any(len(self._vertices) <= idx or idx < 0 for idx in indices):
            raise IndexError('Vertex index out of range')
        # use indices to update self._faces
        faces_to_remove = fix_references(self._faces, indices)
        # remove vertices at all the indices
        self._vertices = np.delete(self._vertices, indices, axis=0)
        # remove faces and possibly parts
        fix_references(self._parts, faces_to_remove)

    def update_vertex(self, idx: int, new_vert: Vertex) -> None:
        """update the position of the vertex at given index"""
        if len(self._vertices) <= idx or idx < 0:
            raise IndexError(f'Vertex index {idx} out of range for vertices of length {len(self._vertices)}')
        if isinstance(new_vert, np.ndarray) and len(new_vert.shape) != 1:
            raise TypeError(f'Vertex {new_vert} has incorrect shape, expected 1D-like array.')
        if self._vertices.shape[1] != len(new_vert):
            raise InvalidMeshException(f'Current indices have dimension {self._vertices.shape[1]}, while '
                                       f'new vertex has dimension {len(new_vert)} .')
        try:
            self._vertices[idx] = np.array(new_vert)
        except (np.VisibleDeprecationWarning, TypeError, ValueError) as e:
            raise InvalidMeshException("Could not update Vertex. Dimensional mismatch for vertices."
                                       "All vertices should have the same number of dimensions.") from e

    def add_faces(self, new_faces: Faces) -> None:
        """adds new faces"""
        if not is_twice_nested_iterable(new_faces):
            raise InvalidMeshException("new_faces should be twice nested iterable.")

        for new_face in new_faces:
            if any(0 > v or v >= len(self._vertices) for v in new_face):
                raise IndexError('Vertex index not defined')

        if isinstance(new_faces, list):
            self._faces += new_faces
        elif isinstance(new_faces, (np.ndarray, tuple)):
            for val in new_faces:
                self._faces.append(val)
        else:
            raise TypeError(f'unknown type for new_face {type(new_faces)}')

    def remove_faces(self, indices: Union[np.ndarray, List[int]]) -> None:
        """removes the faces with given indices and clean-up parts"""
        if any(len(self._faces) <= idx or idx < 0 for idx in indices):
            raise IndexError('Face index out of range')

        # use indices to update self._parts
        fix_references(self._parts, indices)
        # remove faces at all the indices
        for index in indices:
            del self._faces[index]

        # warn on dangling vertices and faces
        if self._test_for_dangling and self.dangling_vert_check():
            warnings.warn('Dangling vertices due to face removal')
        if self._test_for_dangling and self.dangling_face_check():
            warnings.warn('Dangling faces due to removed parts by face removal')

    def update_face(self, idx: int, new_face: Face) -> None:
        """update face with index idx to take new vertices"""
        if len(self._faces) <= idx or idx < 0:
            raise IndexError(f'Face index {idx} out of range.')
        if isinstance(new_face, np.ndarray) and len(new_face.shape) != 1:
            raise TypeError(f'Face {new_face} has incorrect shape, expected 1D-like array.')
        if any(0 > v_idx or v_idx >= len(self._vertices) for v_idx in new_face):
            raise IndexError('Vertex index out of range.')
        # update face
        self._faces[idx] = np.array(new_face)
        if self._test_for_dangling and self.dangling_vert_check():
            warnings.warn('Dangling vertices due to face update')
        # update edges (FIXME: only update new/deleted edges, not everything)
        self._edges = self.extract_edges()

    def add_parts(self, new_parts: Parts) -> None:
        """adds new parts"""
        if not is_twice_nested_iterable(new_parts):
            raise InvalidMeshException('new_parts is not a valid nested iterable')
        for new_part in new_parts:
            if any(0 > f or f >= len(self._faces) for f in new_part):
                raise IndexError(f'Face index out of range for new part: {new_part}')
        if isinstance(new_parts, list):
            self._parts += [np.array(part) for part in new_parts]
        elif isinstance(new_parts, (np.ndarray, tuple)):
            for val in new_parts:
                self._parts.append(np.array(val))
        else:
            raise TypeError(f'unknown type for new_part {type(new_parts)}')

    def remove_parts(self, indices: Union[np.ndarray, List[int]]) -> None:
        """removes the parts with given indices"""
        if any(len(self._parts) <= idx or idx < 0 for idx in indices):
            raise IndexError('Part index out of range')
        # remove indices back to front
        indices[:] = list(set(indices))
        indices.sort(reverse=True)
        for idx in indices:
            del self._parts[idx]
        # post remove
        if self._test_for_dangling and self.dangling_face_check():
            warnings.warn('Dangling faces due to part removal')

    def update_part(self, idx: int, new_part: Part) -> None:
        """update part with index idx to take new faces"""
        if len(self._parts) <= idx or idx < 0:
            raise IndexError(f'Part index {idx} out of range.')
        if isinstance(new_part, np.ndarray) and len(new_part.shape) != 1:
            raise TypeError(f'Part {new_part} has incorrect shape, expected 1D-like array.')
        if any(0 > f_idx or f_idx >= len(self._faces) for f_idx in new_part):
            raise IndexError('Face index out of range.')
        # update part
        self._parts[idx] = np.array(new_part)
        if self._test_for_dangling and self.dangling_face_check():
            warnings.warn('Dangling faces due to part update')

    def add_to_mesh(self, other: 'Mesh') -> None:
        """
        add another mesh to current mesh
        To get consistent indices of the new mesh, the indices of other will be shifted by len(self.vertices).
        Therefore, if you want to use the old indices, either subtract the len or use negative indices to start with.
        """
        # Mesh has to be a correct mesh therefore many checks can be omitted
        # check if vertices have the same dimension
        if self._vertices.shape[1] != other.get_vertices().shape[1]:
            raise InvalidMeshException("Can not concatenate meshes with vertices of different dimensionality.")

        # save shift factor
        pre_nof_vertices = len(self.get_vertices())
        pre_nof_faces = len(self.get_faces())

        # add vertices
        self.add_vertices(other.get_vertices())
        # shift faces indices by the amount of vertices of the current mesh, check them and finally append them
        shifted_faces = [face + pre_nof_vertices for face in other.get_faces()]
        if any(min(sf) < 0 or max(sf) >= len(self._vertices) for sf in shifted_faces):
            raise IndexError("A face index is out of bounds.")
        self._faces += shifted_faces
        # shift parts indices by the amount of faces of the current mesh and append them
        shifted_parts = [part + pre_nof_faces for part in other.get_parts()]
        if any(min(sp) < 0 or max(sp) >= len(self._faces) for sp in shifted_parts):
            raise IndexError("A part index is out of bounds.")
        self._parts += shifted_parts

        # warn on dangling vertices and faces
        if self._test_for_dangling and self.dangling_vert_check():
            warnings.warn('Dangling vertices due to face removal')
        if self._test_for_dangling and self.dangling_face_check():
            warnings.warn('Dangling faces due to removed parts by face removal')

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

    def scale_mesh(self, scaling: float) -> None:
        """scales all vertices """
        self._vertices *= float(scaling)

    def translate_mesh(self, translation: np.ndarray) -> None:
        """translates all vertices by a given vector"""
        if not isinstance(translation, np.ndarray):
            raise TypeError(f'Translation should be a np.ndarray, but was a {type(translation)} instead')
        if len(translation.shape) != 1 or translation.shape[0] != self.dim:
            raise ValueError(f'Translation has wrong dimensions expected {self.dim} was {translation.shape}')
        self._vertices += np.array(translation)

    def translate_vertex(self, v_id: int, translation: np.ndarray) -> None:
        """translate a single vertex by a given vector"""
        if 0 > v_id or v_id >= len(self._vertices):
            raise IndexError(f'Index {v_id} out of bounds for vertices of shape {self._vertices.shape}')
        if not isinstance(translation, np.ndarray):
            raise TypeError(f'Translation should be a np.ndarray, but was a {type(translation)} instead')
        if len(translation.shape) != 1 or translation.shape[0] != self.dim:
            raise ValueError(f'Translation has wrong dimensions expected {self.dim} was {translation.shape}')
        self._vertices[v_id] += translation

    def apply_rotation(self, angle, axis=None) -> None:
        """
        rotates all vertices
        implemented only for 2D and 3D
        2D is equal to rotation around (non-existent) z axis
        3D rotates all vertices around the given axis
        :param angle: rotation angle in radians
        :param axis: for 3D - rotation axis, x (axis=0), y (axis=1) or z (axis=2)
        """
        # define basic rotation matrix
        rot_2d = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
        if self.dim == 2:
            self._vertices = self._vertices @ rot_2d.T  # transposed because we got row vectors not col vecs
        elif self.dim == 3:  # 3d uses the 2D matrix in different columns
            rotation_matrix = np.eye(3)

            if axis == 0:
                rotation_matrix[1:, 1:] = rot_2d
            elif axis == 1:
                rotation_matrix[2::-2, 2::-2] = rot_2d
            elif axis == 2:
                rotation_matrix[:-1, :-1] = rot_2d
            else:
                raise ValueError('In 3D parameter \'axis\' must be 0, 1 or 2')
            self._vertices = self._vertices @ rotation_matrix.T  # transposed because we got row vectors not col vecs
        else:
            raise NotImplementedError("No implementation for n-Dimensional vector rotation")

    def snap_to_grid(self, grid_sizes: Tuple[float, ...], threshold: Tuple[float, ...], steps: int = 1) -> None:
        """
        given vertices of a mesh, move vertices to exact locations if they are close-by.
        e.g. if there is some value 0.999 or 1.001, it would be shifted towards 1.000 if the grid size is 1,
        on the other hand 0.699 would *not* be shifted to 0.700, only iff grid size changed
        Threshold is the amount of movement allowed to snap into the next grid position

        If you want only one dimension to be snapped, use threshold 0 on all other axes
        :param grid_sizes: grid resolution / size in every axis direction
        :type grid_sizes: tuple with the same size as self.dim
        :param threshold: defines the threshold of every axis
        :type threshold: tuple with the same size as self.dim
        :param steps: the number of steps to take before hitting the grid (for animation)
        :type steps: positive integer
        """
        if len(grid_sizes) != self.dim:
            raise ValueError(f'Grid sizes dim is incorrect, was {len(grid_sizes)} expected {self.dim}.')
        if len(threshold) != self.dim:
            raise ValueError(f'Threshold dim is incorrect, was {len(threshold)} expected {self.dim}.')
        if any(v <= 0 for v in grid_sizes):
            raise ValueError("invalid value for grid_sizes. Has to be greater than zero.")
        if any(2 * threshold[i] >= grid_sizes[i] for i in range(self.dim)):
            raise ValueError("threshold can not be bigger than half of grid_size for same dimension.")
        if all(t == 0 for t in threshold):
            raise ValueError("one value in threshold has to be != 0")
        if steps <= 0:
            raise ValueError(f'steps has to be a positive integer, but was {steps}')
        # look at every dimension separately
        for d in range(self.dim):
            curr_vals = self._vertices[:, d]
            mod = curr_vals % grid_sizes[d]
            # set everything that is less than threshold to the difference to the next step
            differences = np.where((mod > 0) & (mod <= threshold[d]), -mod, mod)
            # set everything that is close to the next step to the (positive) missing difference to the next step
            differences = np.where(
                (differences > 0) & (differences >= grid_sizes[d] - threshold[d]),
                grid_sizes[d] - differences,
                differences
            )
            # set everything that is bigger than Threshold to zero
            differences = np.where(differences > threshold[d], 0, differences)
            # add differences to vertices to snap every modified value
            # possibility to move stepwise for later animation
            self._vertices[:, d] += (differences / steps)

    def remove_duplicate_vertices(self) -> None:
        """remove exact duplicates in vertices"""
        # get unique vertices, indices, and the inverse references
        new_verts, indices, inverse = np.unique(self._vertices, axis=0, return_index=True, return_inverse=True)
        # sort indices, to keep current sorting for faces, then set unique vertices
        permutation = indices.argsort()
        self._vertices = new_verts[permutation]
        # switch index of every face that contains a value that is duplicate
        for i_inv, i_idx in enumerate(inverse):
            # for every index in inverse (which are original indices)
            # replace it with the value indices[i_idx] for every face
            for face in self._faces:
                np.place(face, face == i_inv, indices[i_idx])

    def remove_duplicate_faces(self) -> None:
        """remove duplicates in faces, indices only have to be in the correct order, not at the exact places"""
        to_delete = []
        for i_face, face in enumerate(self._faces):
            if i_face in to_delete:  # skip indices marked for deletion
                continue
            others = self.find_face(face, start=i_face + 1)
            to_delete += others
            for o_f_idx in others:
                # change parts to use first index of the same face
                for part in self._parts:
                    np.place(part, part == o_f_idx, i_face)
        # delete all duplicates from faces
        for del_f_idx in sorted(list(set(to_delete)), reverse=True):
            del self._faces[del_f_idx]

    def remove_duplicate_parts(self) -> None:
        """remove duplicates in parts, indices only have to be in the correct order, not at the exact places"""
        to_delete = []
        for i_part, part in enumerate(self._parts):
            if i_part in to_delete:  # skip indices marked for deletion
                continue
            others = self.find_part(part, start=i_part + 1)
            to_delete += others
            # no further nested type
        for del_p_idx in sorted(list(set(to_delete)), reverse=True):
            # exact duplicates can be removed
            del self._parts[del_p_idx]

    def remove_duplicates(self) -> None:
        """remove all duplicates in the current mesh"""
        self.remove_duplicate_vertices()
        self.remove_duplicate_faces()
        self.remove_duplicate_parts()

    def extract_edges(self) -> Edges:
        """returns all edges of the mesh as list of sorted 2-tuples of vertex indices, e.g. [(1,2), (2,3)]"""
        edges = []
        for face in self._faces:
            last_vertex = face[-1]
            for _, vertex_idx in enumerate(face):
                edge = tuple(sorted([last_vertex, vertex_idx]))
                last_vertex = vertex_idx
                if edge not in edges:
                    edges.append(edge)

        return edges

    # def is_face_ccw(self, face_id: int) -> bool:
    #     """check if face is counter-clockwise"""
    #     # FIXME is this even possible to check without going through the whole mesh?
    #     raise NotImplementedError
    #
    # def is_mesh_ccw(self) -> bool:
    #     """check full mesh """
    #     # FIXME see: is_face_ccw
    #     return all(self.is_face_ccw(f_id) for f_id in range(len(self._faces)))
