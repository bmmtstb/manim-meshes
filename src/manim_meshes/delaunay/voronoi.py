"""
functions to display voronoi diagram and create delaunay meshes as its dual
"""
# python imports
import numpy as np
# third-party imports
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module
import manim as m
# local imports
from manim_meshes.models.manim_models.triangle_mesh import TriangleManim2DMesh


class VoronoiDelaunay:
    """
    Class providing methods to visualize the voronoi diagram of a 2D point set and its dual
    delaunay triangulation
    """

    def __init__(self, scene: m.Scene, triangle_mesh: TriangleManim2DMesh) -> None:
        """
        initialize class, triangle_mesh must must be already added to scene (scene.add)
        """
        self.scene: m.Scene = scene
        self.triangle_mesh: TriangleManim2DMesh = triangle_mesh
        verts = self.triangle_mesh.mesh.get_3d_vertices()
        self.voronoi = Voronoi(verts[:, :2])

    def create_voronoi(self):
        """ Creates voronoi diagram of the vertices in triangle_mesh.
        Returns voronoi vertices and lines as VGroups for rendering.
        -> tuple (vertices: VGroup(Dot), lines: VGroup(Line))

        ~> based on source code of scipy.spatial.voronoi_plot_2d
        """
        verts = self.triangle_mesh.mesh.get_3d_vertices()
        vert_group = m.VGroup()
        line_group = m.VGroup()

        center = verts.mean(axis=0)
        ptp_bound = verts.ptp(axis=0)
        voronoi_vertices = np.pad(self.voronoi.vertices, ((0, 0), (0, 1)))

        # add voronoi lines
        for point_indices, segment in zip(self.voronoi.ridge_points, self.voronoi.ridge_vertices):
            segment = np.asarray(segment)
            if np.all(segment >= 0):  # finite segment
                line = m.Line(voronoi_vertices[segment[0]], voronoi_vertices[segment[1]],
                              stroke_width=self.triangle_mesh.edges_width, color=m.WHITE)
                line_group.add(line)
            else:  # infinite segment
                i = segment[segment >= 0][0]  # finite end

                t = verts[point_indices[1]] - verts[point_indices[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0], 0])  # normal

                midpoint = verts[point_indices].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = voronoi_vertices[i] + direction * ptp_bound.max()

                line = m.Line(voronoi_vertices[i], far_point,
                              stroke_width=self.triangle_mesh.edges_width, color=m.WHITE)
                line_group.add(line)

        # add voronoi vertices
        for vert in voronoi_vertices:
            dot = m.Dot(vert, radius=self.triangle_mesh.verts_size, color=m.WHITE)
            vert_group.add(dot)

        return vert_group, line_group

    def get_circum_circle(self, voronoi_vertex_index):
        """ Returns circum-circle around the triangle, which corresponds to the voronoi_vertex with index
        voronoi_vertex_index as manim Circle object"""
        verts = self.triangle_mesh.mesh.get_3d_vertices()
        voronoi_vertices = np.pad(self.voronoi.vertices, ((0, 0), (0, 1)))
        for point_indices, segment in zip(self.voronoi.ridge_points, self.voronoi.ridge_vertices):
            if voronoi_vertex_index in segment:
                vert_a = voronoi_vertices[voronoi_vertex_index]
                vert_b = verts[point_indices[0]]
                circle = m.Circle(radius=np.linalg.norm(vert_b - vert_a), stroke_width=2, color=m.ORANGE)
                circle.shift(vert_a)
                return circle
        return None  # should never get here

    def create_triangle(self, voronoi_vertex_index):
        """ Create and add triangle to mesh, which corresponds to the voronoi_vertex with index voronoi_vertex_index"""

        triangle_indices = set()
        for point_indices, segment in zip(self.voronoi.ridge_points, self.voronoi.ridge_vertices):
            if voronoi_vertex_index in segment:
                triangle_indices.add(point_indices[0])
                triangle_indices.add(point_indices[1])
        _, _ = self.triangle_mesh.add_face(np.array(list(triangle_indices)))
