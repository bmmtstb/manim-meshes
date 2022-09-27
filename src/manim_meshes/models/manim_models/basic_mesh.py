"""
manim models for mesh objects
contains a 3D and a 2D version

additionally there is the mesh for only triangles in triangle_mesh.py
"""
# python imports
import copy
from typing import Tuple
# third-party imports
import manim as m
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
import numpy as np
# local imports
from manim_meshes.exceptions import InvalidMeshDimensionsException, InvalidMeshException, InvalidShapeException
from manim_meshes.helpers import remove_keys_from_dict
from manim_meshes.models.data_models.mesh import Mesh
from manim_meshes.params import get_param_or_default, BM2DM, BM3DM
from manim_meshes.types import Vertices


# pylint: disable=too-many-instance-attributes
class ManimMesh(m.Group, metaclass=ConvertToOpenGL):
    """
    another Mesh implementation, a bit faster + looks better

    inspired by manim class 'Surface'
    possible kwargs: [see BM3DM]
        display_vertices: whether to display the vertices
        display_edges: whether to display the edges
        display_faces: whether to display the faces
        clear_vertices: whether to clear the vertices after WHAT?
        clear_edges: whether to clear the edges after WHAT?
        clear_faces: whether to clear the faces after WHAT?
        edges_color: color of the edges
        edges_width: width of the lines of the edges
        faces_color: color of the faces
        faces_opacity: opacity of the faces
        verts_color: color of the vertices
        pre_function_handle_to_anchor_scale_factor: ?
    """

    def __init__(self, mesh: Mesh, *args, **kwargs) -> None:
        """
        initialize super Group and set all the params
        vertices, edges and faces are groups, so we can easily access them later on
        finally setup everything that needs to be rendered
        """
        super().__init__(*args)
        self.mesh: Mesh = mesh
        self.vertices: m.Group = m.Group()
        self.edges: m.VGroup = m.VGroup()
        self.faces: m.VGroup = m.VGroup()

        # set all the parameters
        for param_name in BM3DM:
            self.__setattr__(param_name, get_param_or_default(param_name, kwargs, BM3DM))

        self.setup()

    def setup(self):
        """set all the necessary mesh parameters"""
        if self.display_faces:
            self.setup_faces()
        if self.display_edges:
            self.setup_edges()
        if self.display_vertices:
            self.setup_vertices()
        # add all the objects to the scene renderer
        self.add(self.faces, self.edges, self.vertices)

    def setup_vertices(self) -> m.Group:
        """set the vertices as 3D manim objects"""
        if self.clear_vertices:
            self.vertices = m.Group()

        for v in self.mesh.get_3d_vertices():
            self.vertices.add(m.Sphere(v, radius=self.verts_size, color=self.verts_color))
        return self.vertices

    def setup_edges(self) -> m.VGroup:
        """set the edges as manim objects"""
        if self.clear_edges:
            self.edges.clear_points()

        vertices = self.mesh.get_3d_vertices()
        for edge_verts in self.mesh.edges:
            vert_1 = vertices[edge_verts[0]]
            vert_2 = vertices[edge_verts[1]]
            edge = m.ThreeDVMobject()
            edge.set_points_as_corners([vert_1, vert_2])
            self.edges.add(edge)
        # for all edges at once
        self.edges.set_fill(
            color=self.edges_color,
            opacity=1.0,
        )
        self.edges.set_stroke(
            color=self.edges_color,
            width=self.edges_width,
            opacity=1.0,
        )
        return self.edges

    def setup_faces(self) -> m.VGroup:
        """
        set the current mesh up as manim objects
        should work for any sized face, not just triangles
        """
        if self.clear_faces:
            self.faces.clear_points()

        verts_3d = self.mesh.get_3d_vertices()
        for face_indices in self.mesh.faces:
            face_points = [verts_3d[i] for i in face_indices]
            face_points.append(verts_3d[face_indices[0]])
            new_face = m.ThreeDVMobject()
            new_face.set_points_as_corners(
                face_points
            )
            self.faces.add(new_face)
        self.faces.set_fill(
            color=self.faces_color,
            opacity=self.faces_opacity
        )
        self.faces.set_stroke(
            color=self.faces_color,
            width=0.,
            opacity=0.,
        )
        return self.faces

    def get_vertex(self, vertex_idx) -> m.mobject:
        """get the vertex with the given id"""
        return self.vertices.submobjects[vertex_idx]

    def get_face(self, face_idx) -> m.mobject:
        """get the face with the given id"""
        return self.faces.submobjects[face_idx]

    def get_edge(self, edge_idx) -> m.mobject:
        """get the edge with the given id"""
        return self.edges.submobjects[edge_idx]

    def add_vertices(self, new_vertices: Vertices, scene: m.Scene) -> None:
        """fade in some additional vertices"""
        self.mesh.add_vertices(new_vertices)
        # fade out current ones, fade in all after add
        scene.play(m.FadeOut(self.vertices), m.FadeIn(self.setup_vertices()))

    def align_points_with_larger(self, larger_mobject):
        """abstract from super - please the linter"""
        raise NotImplementedError

    def _update_vertex(self, vertex_idx: int, pos: np.ndarray):
        """
        TODO
        """
        # update mesh
        self.mesh.update_vertex(vertex_idx, pos)
        if self.display_vertices:
            # update vertex
            vertex = self.get_vertex(vertex_idx)
            vertex.move_to(np.pad(pos, (0, 3 - len(pos))))
        if self.display_faces:
            # update faces
            for face_idx, face in enumerate(self.mesh.faces):
                if vertex_idx in face:
                    triangle = [self.mesh.get_3d_vertices()[i] for i in face]
                    face = self.get_face(face_idx)
                    face.set_points_as_corners(
                        [
                            triangle[0],
                            triangle[1],
                            triangle[2],
                            triangle[0]
                        ],
                    )
        if self.display_edges:
            # update edges
            for edge in self.mesh.get_vertex_edges(vertex_idx):
                self._update_edge(edge)

    def _update_edge(self, edge: Tuple[int, int]):
        """
        TODO
        """
        e = self.get_edge(self.mesh.get_edge_index(edge))
        vert_1 = self.mesh.get_3d_vertices()[edge[0]]
        vert_2 = self.mesh.get_3d_vertices()[edge[1]]
        e.set_points_as_corners([vert_1, vert_2])

    def shift_vertex(self, scene: m.Scene, vertex_idx: int, shift: np.ndarray, **kwargs):
        """
        shift vertex by id and update faces
        expect shift to have the same dimensions as mesh.dim

        shift_vertex_runtime: runtime in seconds for current call
        """
        start = self.mesh.vertices[vertex_idx].copy()
        tracker = m.ValueTracker(0)
        tracker.add_updater(lambda mo: self._update_vertex(vertex_idx, start + tracker.get_value() * shift, **kwargs))
        scene.add(tracker)
        scene.play(
            tracker.animate(**kwargs).set_value(1),
            run_time=kwargs["shift_vertex_runtime"] if "shift_vertex_runtime" in kwargs else 1.0
        )
        scene.remove(tracker)

    def shift_vertices(self, scene: m.Scene, shift: np.ndarray, **kwargs):
        """
        shift multiple vertices from self.mesh.vertices by shift using the manim tracker and updater
        make sure to update all points simultaneously, not one after the other

        shift_vertices_runtime: runtime in seconds for current call
        """
        start = copy.deepcopy(self.mesh.vertices)
        tracker = m.ValueTracker(0)
        for vertex_idx in range(len(self.mesh.vertices)):
            tracker.add_updater(
                # make sure at the moment when lambda is called, it still has the correct bound loop variable
                lambda mo, bound_v_id=vertex_idx: self._update_vertex(
                    vertex_idx=bound_v_id,
                    pos=start[bound_v_id] + tracker.get_value() * shift[bound_v_id],
                    **remove_keys_from_dict(kwargs, ["shift_vertices_runtime"])
                ))
        scene.add(tracker)
        scene.play(
            tracker.animate(**remove_keys_from_dict(kwargs, ["shift_vertices_runtime"])).set_value(1),
            run_time=kwargs["shift_vertices_runtime"] if "shift_vertices_runtime" in kwargs else 1.0
        )
        scene.remove(tracker)

    def move_vertices_to(self, scene: m.Scene, new_positions: np.ndarray, **kwargs):
        """visually move all vertices to new positions and update faces. In the end update self.mesh as well"""
        if len(new_positions) != len(self.mesh.vertices):
            raise InvalidShapeException("new_positions", len(new_positions), len(self.mesh.vertices))
        shift: np.ndarray = new_positions - self.mesh.vertices
        self.shift_vertices(scene, shift=shift, **kwargs)

    def move_vertex_to(self, scene: m.Scene, vertex_idx: int, pos: np.ndarray, **kwargs):
        """visually move vertex to pos and update faces"""
        # expect pos and curr_pos / mesh.dim to have the same dimensions
        if self.mesh.dim != len(pos):
            raise InvalidMeshDimensionsException(len(pos), self.mesh.dim, "pos")
        shift = pos - self.mesh.vertices[vertex_idx]
        # use shift method to slowly move point to desired place
        self.shift_vertex(scene, vertex_idx, shift, **kwargs)

    def move_to_grid(
            self, scene: m.Scene, grid_sizes: Tuple[float, ...], threshold: Tuple[float, ...],
            nof_steps: int = 1, **kwargs
        ) -> None:
        """slowly snap to a given grid, uses stepwise mesh.snap_to_grid()"""
        # to be able to show the movement, the update needs to be calculated on a dummy mesh first
        new_verts = self.mesh.snap_to_grid(grid_sizes, threshold, steps=nof_steps, update_verts=False)
        # use new calculated positions but have still the old mesh
        self.move_vertices_to(scene, new_verts, **kwargs)


class Manim2DMesh(ManimMesh):
    """
    "2D" mesh implementation
    printing Vertices in Manim is currently not supported for 2D vertices. Therefore, while printing the appropriate
    3D-vertices are used. Everything else should accept plain 2D values. Therefore, this Manim2DMesh class should
    support 2D vertices or 3D vertices with z-value == 0 on initialization

    This mesh is mainly for Educational purposes and has a few functions we needed for drawing basic
    mesh functionalities. It is performant up to a point and should not be used for larger meshes.
    """

    def __init__(self, mesh: Mesh, *args, **kwargs) -> None:
        if mesh.dim == 3:
            if np.sum(np.abs(mesh.vertices[:, 2] != 0)):
                raise InvalidMeshException("Mesh has z values != 0 and therefore is not 2D.")
        elif mesh.dim > 3:
            raise InvalidMeshException(
                f'Mesh is not in the correct format. Expected Dim 2 or 3 with z zero, was {mesh.dim}')
        # init ManimMesh
        super().__init__(
            mesh=mesh,
            *args,
            **{key: get_param_or_default(key, kwargs, BM2DM) for key in BM2DM},
            **remove_keys_from_dict(kwargs, list(BM2DM.keys())),
        )

    def setup_vertices(self) -> m.Group:
        """set the vertices as 3D manim objects"""
        if self.clear_vertices:
            self.vertices = m.Group()

        for v in self.mesh.get_3d_vertices():
            self.vertices.add(m.Dot(v, radius=self.verts_size, color=self.verts_color))
        return self.vertices

    def get_dots(self, indices):
        """
        returns manim Dot objects for each vertex with index in indices
        """
        dots = []
        vertices = self.mesh.get_3d_vertices()
        for idx in indices:
            dot = m.Dot(vertices[idx], radius=0.03, color=m.RED)
            dot.add_updater(lambda mo, mesh=self.mesh, index=idx: mo.move_to(mesh.get_3d_vertices()[index]))
            dots.append(dot)
        return dots

    def align_points_with_larger(self, larger_mobject):
        """abstract from super - please the linter"""
        raise NotImplementedError
