"""
functions to create delaunay meshes by divide and conquer
"""
# python imports
from typing import List
import numpy as np
# third-party imports
import manim as m
# local imports

from manim_meshes.delaunay.delaunay_criterion import get_triangle_circum_circle_params
from manim_meshes.models.manim_models.triangle_mesh import TriangleManim2DMesh


def get_clockwise_angle(a, b):
    """ Returns clockwise angle between 2D vectors a and b """
    dot = a[0] * b[0] + a[1] * b[1]  # proportional to cos
    det = a[0] * b[1] - b[0] * a[1]  # proportional to sin
    angle = -np.arctan2(det, dot)  # atan2(sin, cos)
    if angle < 0:  # counterclockwise
        angle = 2 * np.pi + angle
    return angle


def get_counter_clockwise_angle(a, b):
    """ Returns counter-clockwise angle between 2D vectors a and b """
    dot = a[0] * b[0] + a[1] * b[1]  # proportional to cos
    det = a[0] * b[1] - b[0] * a[1]  # proportional to sin
    angle = np.arctan2(det, dot)  # atan2(sin, cos)
    if angle < 0:  # clockwise
        angle = 2 * np.pi + angle
    return angle


class DivideAndConquer:
    """
    Class providing methods to visualize the divide and conquer algorithm, which generates a delaunay triangulation
    for a 2D point set

    Algorithm based on http://www.geom.uiuc.edu/~samuelp/del_project.html, Guibas and Stolfi
    """

    def __init__(self, scene: m.Scene, triangle_mesh: TriangleManim2DMesh) -> None:
        """
        initialize class, triangle_mesh must contain only vertices (len(triangle_mesh.mesh.faces) == 0)
        and must be already added to scene (scene.add)
        """
        self.scene: m.Scene = scene
        self.triangle_mesh: TriangleManim2DMesh = triangle_mesh

    def split_points(self, vert_indices, dash_length=0.2, line_width=1, speed=1.):
        """ Splits set of vertices given by vert_indices into two equally sized sets. The objects are sorted
            and then split by their x-coordinate. Calculates a dashed line between both sets.

            Returns indices of resulting sets and DashedLine manim object
            ~> tuple (indices_left: List, indices_right: List, line: DashedLine)

            speed: animation speed, lower = faster """

        verts_3d = self.triangle_mesh.mesh.get_3d_vertices()
        vertices = verts_3d[vert_indices]

        # sort and get split index
        sort_indices = [x for x, y in sorted(enumerate(vertices), key=lambda x: x[1][0])]
        sorted_verts = [vertices[x] for x in sort_indices]
        sorted_vert_indices = [vert_indices[x] for x in sort_indices]
        split_index = len(sorted_verts) // 2

        # draw split line
        x_mid = (sorted_verts[split_index - 1][0] + sorted_verts[split_index][0]) / 2.
        y_max = np.max(verts_3d[:, 1])
        y_min = np.min(verts_3d[:, 1])
        # z > 0 -> always draw in front
        split_line = m.DashedLine(start=np.array([x_mid, y_min, 0.01]), end=np.array([x_mid, y_max, 0.01]),
                                  stroke_width=line_width, dash_length=dash_length)
        self.scene.play(m.Create(split_line, run_time=0.5 * speed))
        self.scene.wait(0.15 * speed)

        # return indices of resulting sets
        return sorted_vert_indices[:split_index], sorted_vert_indices[split_index:], split_line

    def triangulate_leq_3(self, vert_indices: List):
        """ Triangulation of a vertices set given by vert_indices with no more than 3 points
            (len(vertices) must be lower-equal 3) ~> draws new triangle or segment if possible """

        if len(vert_indices) > 3:
            raise ValueError("len(vert_indices) must be lower-equal 3")
        if len(vert_indices) > 1:
            new_objects = []
            if len(vert_indices) == 2:
                vert_indices = vert_indices.copy()
                vert_indices.append(vert_indices[0])
            new_face, new_edges = self.triangle_mesh.add_face(np.array(vert_indices))
            new_objects.append(new_face)
            new_objects.extend(new_edges)
            # update hack (corrects drawing order)
            self.scene.renderer.update_frame(self.scene)

    def _right_candidate(self, base_lr, rr_edges, run_time=0.3):
        """ Find and return right potential candidate (or None if not found)
        to build triangle for merging, deletes RR edges if necessary """
        endpoints = [edge[0] if base_lr[1] != edge[0] else edge[1] for edge in rr_edges if base_lr[1] in edge]
        vertices = self.triangle_mesh.mesh.get_3d_vertices()
        angles = np.array(
            [get_clockwise_angle(vertices[base_lr[0]] - vertices[base_lr[1]], vertices[endpoint] - vertices[base_lr[1]])
             for endpoint in endpoints])
        order = np.argsort(angles)
        for i in range(len(endpoints)):
            potential_candidate = endpoints[order[i]]
            next_potential_candidate = endpoints[order[i + 1]] if i != len(endpoints) - 1 else None
            c, r = get_triangle_circum_circle_params(vertices[base_lr[0]], vertices[base_lr[1]],
                                                     vertices[potential_candidate])
            if angles[order[i]] < np.pi:  # angle less than 0 degree
                if next_potential_candidate is None or not np.linalg.norm(c - vertices[endpoints[order[i + 1]]]) < r:
                    return endpoints[order[i]]  # next_potential_candidate not within circle defined by base_lr and
                    # potential_candidate

                # delete RR edge to potential_candidate
                faces = self.triangle_mesh.mesh.faces
                face_idx_to_delete = None
                for face_idx, face in enumerate(faces):
                    if endpoints[order[i]] in face and base_lr[1] in face:
                        face_idx_to_delete = face_idx
                        break
                face, edges = self.triangle_mesh.remove_face(face_idx_to_delete)
                self.scene.play(m.Uncreate(face, run_time=run_time), m.FadeOut(*edges, run_time=run_time))
                rr_edges.remove(tuple(sorted((base_lr[1], potential_candidate))))
        return None

    def _left_candidate(self, base_lr, ll_edges, run_time=0.3):
        """ Find and return left potential candidate (or None if not found)
        to build triangle for merging, deletes LL edges if necessary """
        endpoints = [edge[0] if base_lr[0] != edge[0] else edge[1] for edge in ll_edges if base_lr[0] in edge]
        vertices = self.triangle_mesh.mesh.get_3d_vertices()
        angles = np.array([get_counter_clockwise_angle(vertices[base_lr[1]] - vertices[base_lr[0]],
                                                       vertices[endpoint] - vertices[base_lr[0]])
                           for endpoint in endpoints])
        order = np.argsort(angles)
        for i in range(len(endpoints)):
            potential_candidate = endpoints[order[i]]
            next_potential_candidate = endpoints[order[i + 1]] if i != len(endpoints) - 1 else None
            c, r = get_triangle_circum_circle_params(vertices[base_lr[0]], vertices[base_lr[1]],
                                                     vertices[potential_candidate])
            if angles[order[i]] < np.pi:
                if next_potential_candidate is None or not np.linalg.norm(c - vertices[endpoints[order[i + 1]]]) < r:
                    return endpoints[order[i]]

                # delete LL edge to potential_candidate
                faces = self.triangle_mesh.mesh.faces
                face_idx_to_delete = None
                for face_idx, face in enumerate(faces):
                    if endpoints[order[i]] in face and base_lr[0] in face:
                        face_idx_to_delete = face_idx
                        break
                face, edges = self.triangle_mesh.remove_face(face_idx_to_delete)
                ll_edges.remove(tuple(sorted((base_lr[0], potential_candidate))))
                self.scene.play(m.FadeOut(face, *edges, run_time=run_time))
        return None

    def merge_sets(self, indices_left: List, indices_right: List, split_line: m.DashedLine, speed=1.):
        """ Merges two delaunay triangulated vertex sets, given by indices (indices_left, indices_right)
        to combined delaunay triangulation, returns vertex indices of combined set.

        speed: animation speed, lower = faster """

        # remove split line
        self.scene.play(m.Uncreate(split_line), run_time=0.5 * speed)
        self.scene.wait(0.15 * speed)

        base_lr = self._find_base_lr(indices_left, indices_right)
        # check edge[0] != edge[1]: class Mesh does not support plain edges without a face, thus they are drawn as
        # faces, e.g. [0,1,0] for edge (0,1) ~> introduces also invalid edge (0,0)
        rr_edges = set(edge for edge in self.triangle_mesh.mesh.extract_edges() if edge[0] != edge[1]
                       and edge[0] in indices_right and edge[1] in indices_right)
        ll_edges = set(edge for edge in self.triangle_mesh.mesh.extract_edges() if edge[0] != edge[1]
                       and edge[0] in indices_left and edge[1] in indices_left)
        while True:
            r_candidate = self._right_candidate(base_lr, rr_edges, 0.3 * speed)
            l_candidate = self._left_candidate(base_lr, ll_edges)
            if r_candidate is None and l_candidate is None:
                break  # merge complete
            if r_candidate is not None and l_candidate is not None:
                # choose candidate by criterion, see http://www.geom.uiuc.edu/~samuelp/del_project.html
                vertices = self.triangle_mesh.mesh.get_3d_vertices()
                c, r = get_triangle_circum_circle_params(vertices[base_lr[0]], vertices[base_lr[1]],
                                                         vertices[l_candidate])
                if np.linalg.norm(c - vertices[r_candidate]) < r:
                    l_candidate = None
                else:
                    r_candidate = None

            candidate = r_candidate if r_candidate is not None else l_candidate
            indices = indices_right if r_candidate is not None else indices_left
            if len(indices) == 2:  # delete segment (fake face, see comment about check edge[0] != edge[1])
                face = self.triangle_mesh.mesh.find_face(np.array([indices[0], indices[1],
                                                                   indices[0]]))
                if len(face) != 0:  # found
                    face_idx_to_delete = face[0]
                    _, _ = self.triangle_mesh.remove_face(face_idx_to_delete)
            # add new face
            _, _ = self.triangle_mesh.add_face(np.array([base_lr[0], base_lr[1], candidate]))
            base_lr = (base_lr[0], candidate) if r_candidate is not None else (candidate, base_lr[1])
            self.scene.wait(0.15 * speed)
        merged_indices = indices_left.copy()
        merged_indices.extend(indices_right)
        return merged_indices

    def _find_base_lr(self, indices_left: List, indices_right: List):
        """ Finds and returns the base_lr (l,r) , where l and r are vertex indices, between two vertex sets
            given by indices (indices_a, indices_b)"""

        def next_on_left_hull(cur_idx):
            verts = self.triangle_mesh.mesh.get_3d_vertices()
            endpoints = [edge[0] if cur_idx != edge[0] else edge[1]
                         for edge in self.triangle_mesh.mesh.get_vertex_edges(cur_idx) if edge[0] != edge[1]]
            angles = np.array([get_counter_clockwise_angle(verts[endpoint] - verts[cur_idx],
                                                           np.array([0, 1, 0]) - verts[cur_idx])
                               for endpoint in endpoints])
            order = np.argsort(angles)
            return endpoints[order[0]]

        def next_on_right_hull(cur_idx):
            verts = self.triangle_mesh.mesh.get_3d_vertices()
            endpoints = [edge[0] if cur_idx != edge[0] else edge[1]
                         for edge in self.triangle_mesh.mesh.get_vertex_edges(cur_idx) if edge[0] != edge[1]]
            angles = np.array([get_clockwise_angle(verts[endpoint] - verts[cur_idx],
                                                           np.array([0, 1, 0]) - verts[cur_idx])
                               for endpoint in endpoints])
            order = np.argsort(angles)
            return endpoints[order[0]]

        def on_right(tangent, point):
            return ((tangent[1][0] - tangent[0][0]) * (point[1] - tangent[0][1]) -
                    (tangent[1][1] - tangent[0][1]) * (point[0] - tangent[0][0])) < 0

        verts = self.triangle_mesh.mesh.get_3d_vertices()
        x_max_left = max((v[0], i) for i, v in enumerate(verts[indices_left]))[1]
        x_min_right = min((v[0], i) for i, v in enumerate(verts[indices_right]))[1]
        left = indices_left[x_max_left]
        right = indices_right[x_min_right]

        # move tangent 'down'
        next_left = next_on_left_hull(left)
        while on_right((verts[left], verts[right]), verts[next_left]):
            left = next_left
            next_left = next_on_left_hull(left)
        next_right = next_on_right_hull(right)
        while on_right((verts[left], verts[right]), verts[next_right]):
            right = next_right
            next_right = next_on_right_hull(right)

        return left, right

    def divide_and_conquer_recursive(self, speed=1.):
        """ Runs complete (recurive) algorithm to create a delaunay triangulation by divide and conquer.
         Expects self.triangle_mesh to be without defined faces / triangles

        speed: animation speed, lower = faster """

        if len(self.triangle_mesh.mesh.faces) != 0:
            raise ValueError("self.triangle_mesh.mesh.faces must be empty to apply the divide and conquer algorithm!")

        vert_indices = list(range(len(self.triangle_mesh.mesh.vertices)))
        self._divide_and_conquer_recursive(vert_indices, speed)

    def _divide_and_conquer_recursive(self, vert_indices: List, speed=1.):
        """ Recursive internal implementation used by method divide_and_conquer_recursive()

            speed: animation speed, lower = faster """

        if len(vert_indices) <= 3:
            self.triangulate_leq_3(vert_indices)
            self.scene.wait(0.15 * speed)
            return vert_indices

        indices_left, indices_right, line = self.split_points(vert_indices)
        vert_indices_left = self._divide_and_conquer_recursive(indices_left, speed=speed)
        vert_indices_right = self._divide_and_conquer_recursive(indices_right, speed=speed)
        return self.merge_sets(vert_indices_left, vert_indices_right, line, speed=speed)
