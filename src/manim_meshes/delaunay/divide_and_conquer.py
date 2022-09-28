"""
functions to create delaunay meshes by divide and conquer
"""
# python imports
from random import randint
import numpy as np
# third-party imports
import manim as m
# local imports
from manim_meshes.delaunay.delaunay_criterion import get_triangle_circum_circle_params
from manim_meshes.models.manim_models.triangle_mesh import TriangleManim2DMesh


def get_color():
    """ returns a random color """
    return '#%06X' % randint(0, 0xFFFFFF)


def split_points(scene: m.Scene, dots, dot_indices, color_a, color_b):
    """ Splits set of manim Dot objects into two sets with different colors.
        Returns resulting sets and corresponding indices"""

    sort_indices = [x for x, y in sorted(enumerate(dots), key=lambda x: x[1].get_x())]
    sorted_dots = [dots[x] for x in sort_indices]
    sorted_dot_indices = [dot_indices[x] for x in sort_indices]
    split_index = len(sorted_dots) // 2
    dots_a = sorted_dots[:split_index]
    dots_b = sorted_dots[split_index:]
    anims = []
    for dot in dots_a:
        anims.append(dot.animate.set_color(color_a))
    for dot in dots_b:
        anims.append(dot.animate.set_color(color_b))
    scene.play(*anims)
    return dots_a, dots_b, sorted_dot_indices[:split_index], sorted_dot_indices[split_index:]


def split_points_recursive(scene: m.Scene, dots, dot_indices, old_color):
    """ Splits set of manim Dot objects recursively into subsets until each set consists of less than 3 dots.
    Colors each set differently. Returns resulting indices for each set."""
    if len(dots) <= 3:
        return [dot_indices]
    indices = []
    color = get_color()
    dots_a, dots_b, dot_indices_a, dot_indices_b = split_points(scene, dots, dot_indices, old_color, color)
    indices_a = split_points_recursive(scene, dots_a, dot_indices_a, old_color)
    indices_b = split_points_recursive(scene, dots_b, dot_indices_b, color)
    indices.extend(indices_a)
    indices.extend(indices_b)
    return indices


def triangulate_simple(scene: m.Scene, triangle_mesh, set_dot_indices):
    """ Triangulation of a Dot set wth no more than 3 points ~> creates triangle or segment if possible"""
    new_objects = []
    for dot_indices in set_dot_indices:
        if len(dot_indices) > 3:
            raise ValueError("dot_indices must contain no more than 3 indices!")
        elif len(dot_indices) > 1:
            if len(dot_indices) == 2:
                dot_indices = dot_indices.copy()
                dot_indices.append(dot_indices[0])
            new_face, new_edges = triangle_mesh.add_face(np.array(dot_indices),
                                                         triangle_mesh.vertices.submobjects[dot_indices[0]].get_color())
            new_objects.append(new_face)
            new_objects.extend(new_edges)
    scene.play(m.FadeIn(*new_objects))
    # update hack
    scene.remove(triangle_mesh)
    scene.add(triangle_mesh)
    scene.wait(0.1)


def get_clockwise_angle(a, b):
    " Returns clockwise angle between 2D vectors a and b "
    dot = a[0] * b[0] + a[1] * b[1]  # proportional to cos
    det = a[0] * b[1] - b[0] * a[1]  # proportional to sin
    angle = -np.arctan2(det, dot)  # atan2(sin, cos)
    if angle < 0:  # counterclockwise
        angle = 2 * np.pi + angle
    return angle


def get_counter_clockwise_angle(a, b):
    " Returns counter-clockwise angle between 2D vectors a and b "
    dot = a[0] * b[0] + a[1] * b[1]  # proportional to cos
    det = a[0] * b[1] - b[0] * a[1]  # proportional to sin
    angle = np.arctan2(det, dot)  # atan2(sin, cos)
    if angle < 0:  # clockwise
        angle = 2 * np.pi + angle
    return angle


def right_candidate(scene: m.Scene, triangle_mesh, base_lr, rr_edges):
    """ Find right potential candidate dots to build triangle for merging """
    endpoints = [edge[0] if base_lr[1] != edge[0] else edge[1] for edge in rr_edges if base_lr[1] in edge]
    vertices = triangle_mesh.mesh.get_3d_vertices()
    angles = np.array(
        [get_clockwise_angle(vertices[base_lr[0]] - vertices[base_lr[1]], vertices[endpoint] - vertices[base_lr[1]]) for
         endpoint in endpoints])
    order = np.argsort(angles)
    for i in range(len(endpoints)):
        potential_candidate = endpoints[order[i]]
        next_potential_candidate = endpoints[order[i + 1]] if i != len(endpoints) - 1 else None
        c, r = get_triangle_circum_circle_params(vertices[base_lr[0]], vertices[base_lr[1]],
                                                 vertices[potential_candidate])
        if angles[order[i]] < np.pi:
            if next_potential_candidate is None or not np.linalg.norm(c - vertices[endpoints[order[i + 1]]]) < r:
                return endpoints[order[i]]
            else:  # delete RR edge to potential_candidate
                faces = triangle_mesh.mesh.faces
                face_idx_to_delete = None
                for face_idx, face in enumerate(faces):
                    if endpoints[order[i]] in face and base_lr[1] in face:
                        face_idx_to_delete = face_idx
                        break
                face, edges = triangle_mesh.remove_face(face_idx_to_delete)
                scene.play(m.FadeOut(face, run_time=0.2), m.FadeOut(*edges, run_time=0.2))
                # update hack
                scene.remove(triangle_mesh)
                scene.add(triangle_mesh)
                rr_edges.remove(tuple(sorted((base_lr[1], potential_candidate))))
    return None


def left_candidate(scene: m.Scene, triangle_mesh, base_lr, ll_edges):
    """ Find left potential candidate dots to build triangle for merging """
    endpoints = [edge[0] if base_lr[0] != edge[0] else edge[1] for edge in ll_edges if base_lr[0] in edge]
    vertices = triangle_mesh.mesh.get_3d_vertices()
    angles = np.array([get_counter_clockwise_angle(vertices[base_lr[1]] - vertices[base_lr[0]],
                                                   vertices[endpoint] - vertices[base_lr[0]]) for endpoint in
                       endpoints])
    order = np.argsort(angles)
    for i in range(len(endpoints)):
        potential_candidate = endpoints[order[i]]
        next_potential_candidate = endpoints[order[i + 1]] if i != len(endpoints) - 1 else None
        c, r = get_triangle_circum_circle_params(vertices[base_lr[0]], vertices[base_lr[1]],
                                                 vertices[potential_candidate])
        if angles[order[i]] < np.pi:
            if next_potential_candidate is None or not np.linalg.norm(c - vertices[endpoints[order[i + 1]]]) < r:
                return endpoints[order[i]]
            else:  # delete LL edge to potential_candidate
                faces = triangle_mesh.mesh.faces
                face_idx_to_delete = None
                for face_idx, face in enumerate(faces):
                    if endpoints[order[i]] in face and base_lr[0] in face:
                        face_idx_to_delete = face_idx
                        break
                face, edges = triangle_mesh.remove_face(face_idx_to_delete)
                scene.play(m.FadeOut(face, run_time=0.2), m.FadeOut(*edges, run_time=0.2))
                # update hack
                scene.remove(triangle_mesh)
                scene.add(triangle_mesh)
                ll_edges.remove(tuple(sorted((base_lr[0], potential_candidate))))
    return None


def merge_sets(scene: m.Scene, triangle_mesh, indices_a, indices_b, color):
    """ Merges two delaunay triangulated Dot sets to combined delaunay triangulation """

    base_lr = find_base_lr(triangle_mesh, indices_a, indices_b)
    rr_edges = set(edge for edge in triangle_mesh.mesh.extract_edges() if edge[0] != edge[1]
                   and edge[0] in indices_b and edge[1] in indices_b)
    ll_edges = set(edge for edge in triangle_mesh.mesh.extract_edges() if edge[0] != edge[1]
                   and edge[0] in indices_a and edge[1] in indices_a)
    start = True
    while True:
        r_candidate = right_candidate(scene, triangle_mesh, base_lr, rr_edges)
        l_candidate = left_candidate(scene, triangle_mesh, base_lr, ll_edges)
        if r_candidate is None and l_candidate is None:
            break
        if r_candidate is not None and l_candidate is not None:
            vertices = triangle_mesh.mesh.get_3d_vertices()
            c, r = get_triangle_circum_circle_params(vertices[base_lr[0]], vertices[base_lr[1]], vertices[l_candidate])
            if np.linalg.norm(c - vertices[r_candidate]) < r:
                l_candidate = None
            else:
                r_candidate = None

        anims = []
        if start:
            for idx, face in enumerate(triangle_mesh.mesh.faces):
                if any(x in face for x in indices_b):
                    face = triangle_mesh.faces.submobjects[idx]
                    anims.append(face.animate.set_fill(color).set_stroke(color))
                if any(x in face for x in indices_a):
                    face = triangle_mesh.faces.submobjects[idx]
                    anims.append(face.animate.set_fill(color).set_stroke(color))
            for idx in indices_b:
                dot = triangle_mesh.vertices.submobjects[idx]
                anims.append(dot.animate.set_color(color))
            for idx in indices_a:
                dot = triangle_mesh.vertices.submobjects[idx]
                anims.append(dot.animate.set_color(color))
            start = False
            scene.play(*anims)
            scene.remove(triangle_mesh)
            scene.add(triangle_mesh)

        anims.clear()
        if r_candidate is not None:
            if len(indices_b) == 2:  # delete segment
                f = triangle_mesh.mesh.find_face(np.array([indices_b[0], indices_b[1], indices_b[0]]))
                if len(f) != 0:
                    face_idx_to_delete = f[0]
                    face, edges = triangle_mesh.remove_face(face_idx_to_delete)
                    anims.append(m.FadeOut(face))
                    anims.append(m.FadeOut(*edges))
            face, edges = triangle_mesh.add_face(np.array([base_lr[0], base_lr[1], r_candidate]), color)
            base_lr = (base_lr[0], r_candidate)

        else:
            if len(indices_a) == 2:  # delete segment
                f = triangle_mesh.mesh.find_face(np.array([indices_a[0], indices_a[1], indices_a[0]]))
                if len(f) != 0:
                    face_idx_to_delete = f[0]
                    face, edges = triangle_mesh.remove_face(face_idx_to_delete)
                    anims.append(m.FadeOut(face))
                    anims.append(m.FadeOut(*edges))
            face, edges = triangle_mesh.add_face(np.array([base_lr[0], base_lr[1], l_candidate]), color)
            base_lr = (l_candidate, base_lr[1])

        anims.extend([m.FadeIn(face), m.FadeIn(*edges)])
        scene.play(*anims)
        scene.remove(triangle_mesh)
        scene.add(triangle_mesh)


def find_base_lr(triangle_mesh, indices_a, indices_b):
    """ Finds and returns the base_lr"""
    verts_a = triangle_mesh.mesh.vertices[indices_a]
    verts_b = triangle_mesh.mesh.vertices[indices_b]
    _, idx_a = min((val[1], idx) for (idx, val) in enumerate(verts_a))
    _, idx_b = min((val[1], idx) for (idx, val) in enumerate(verts_b))
    return indices_a[idx_a], indices_b[idx_b]


def divide_and_conquer(scene: m.Scene, triangle_mesh: TriangleManim2DMesh):
    """ Creates a delaunay triangulation by a simple divide and conquer algorithm.
     Expects a TriangleManim2DMesh without defined faces / triangles as input"""

    if len(triangle_mesh.faces) != 0:
        raise ValueError("TriangleManim2DMesh.faces must be empty to apply the divide and conquer algorithm!")

    dots = triangle_mesh.vertices.submobjects
    indices = split_points_recursive(scene, dots, range(len(dots)), triangle_mesh.verts_color)
    triangulate_simple(scene, triangle_mesh, indices)
    color = get_color()
    while len(indices) > 1:
        # Fixme in general
        merge_sets(scene, triangle_mesh, indices[-2], indices[-1], color)
        indices[-2].extend(indices[-1])
        del indices[-1]

