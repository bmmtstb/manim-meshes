"""
create a few sample test scenes to check efficiency of renderer
"""
# third-party imports

import manim as m

# local imports

from manim_meshes.models import ManimMesh, Manim2DMesh
from manim_meshes.templates import create_pyramid, create_model, create_coplanar_triangles


# class PyramidScene(ThreeDScene):
#     """4 sided pyramid"""
#
#     def construct(self):
#         self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
#         mesh = create_pyramid()
#         trimesh_obj = TrimeshObject(mesh=mesh)
#         self.add(trimesh_obj)


# class HandleScene(ThreeDScene):
#     """?"""
#     def construct(self):
#         self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
#         mesh = create_model(name="Handle")
#         trimesh_obj = TrimeshObject(mesh=mesh)
#         self.add(trimesh_obj)


# class ConeScene(ThreeDScene):
#     """Display a cone"""
#     def construct(self):
#         self.set_camera_orientation(phi=70 * DEGREES, zoom=0.40)
#         mesh = create_model(name="tail_topper")
#         mesh.apply_scale(scaling=0.3)
#         mesh.apply_translation([0, -8, 0])
#         trimesh_obj = TrimeshObject(mesh=mesh)
#         self.add(trimesh_obj)


# class SquirrelScene(ThreeDScene):
#     """Display a cute squirrel"""
#     def construct(self):
#         self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
#         mesh = create_model(name="squirrel")
#         trimesh_obj = TrimeshObject(mesh=mesh)
#         self.add(trimesh_obj)


# class LandScene(ThreeDScene):
#     """?"""
#     def construct(self):
#         self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
#         mesh = create_model(name="Land")
#         trimesh_obj = TrimeshObject(mesh=mesh)
#         self.add(trimesh_obj)


# class OctocatScene(ThreeDScene):
#     """Display the Octo-Cat"""
#     def construct(self):
#         self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
#         mesh = create_model(name="Octocat-v1")
#         trimesh_obj = TrimeshObject(mesh=mesh)
#         self.add(trimesh_obj)


# 32GB RAM OVERFLOW
# class ArmadilloScene(ThreeDScene):
#     def construct(self):
#         self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
#         mesh = create_model(name="armadillo")
#         trimesh_obj = TrimeshObject(mesh=mesh)
#         self.add(trimesh_obj)


##### ManimMesh #####
class ConeScene2(m.ThreeDScene):
    """Display a cone"""

    def construct(self):
        self.set_camera_orientation(phi=70 * m.DEGREES, zoom=0.40)
        mesh = create_model(name="tail_topper")
        mesh.apply_scale(scaling=0.3)
        mesh.apply_translation([0, -8, 0])
        manim_mesh_obj = ManimMesh(mesh=mesh)
        self.add(manim_mesh_obj)


class SuzanneScene(m.ThreeDScene):
    """suzanne mesh"""

    def construct(self):
        self.camera.set_phi(90 * m.DEGREES)
        mesh = create_model(name="suzanne")
        manim_mesh_obj = ManimMesh(mesh=mesh)
        self.add(manim_mesh_obj)
        self.play(
            m.Rotate(
                manim_mesh_obj,
                angle=2 * m.PI,
                about_point=m.ORIGIN,
                rate_func=m.linear,
                run_time=5
            )
        )


class PyramidScene2(m.ThreeDScene):
    """pyramid mesh, changes a face color"""

    def construct(self):
        self.set_camera_orientation(70 * m.DEGREES, 30 * m.DEGREES)
        mesh = create_pyramid()
        manim_mesh_obj = ManimMesh(mesh=mesh)
        self.add(manim_mesh_obj)
        self.play(
            manim_mesh_obj.get_face(0).animate.set_fill(m.RED, 0.8)
        )


class TriangleScene(m.ThreeDScene):
    """pyramid as point cloud, changes a point color"""

    def construct(self):
        mesh = create_coplanar_triangles()

        mesh_2d = Manim2DMesh(mesh=mesh)
        self.add(mesh_2d)
        triangle_1 = mesh_2d.get_face(0)
        self.play(triangle_1.animate.set_fill(m.GREEN, 0.6))
        circle = mesh_2d.get_circle(0)
        self.play(m.Create(circle))
        points = mesh_2d.get_points_violating_delaunay(0)
        for point in points:
            self.play(m.FadeIn(point))
        self.wait()
