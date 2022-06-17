"""
create a few sample test scenes to check efficiency of renderer
"""
# third-party imports

from manim import *

# local imports

from manim_trimeshes.models import TrimeshObject, PointCloudObject, ManimMesh, Manim2DMesh
from manim_trimeshes.templates import create_pyramid, create_model, create_coplanar_triangles


class PyramidScene(ThreeDScene):
    """4 sided pyramid"""

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        mesh = create_pyramid()
        trimesh_obj = TrimeshObject(mesh=mesh)
        self.add(trimesh_obj)


class PyramidPointsScene(ThreeDScene):
    """4 sided pyramid as point cloud"""

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        mesh = create_pyramid()
        trimesh_obj = PointCloudObject(mesh=mesh)
        self.add(trimesh_obj)
        self.play(
            Rotate(
                trimesh_obj,
                angle=2 * PI,
                about_point=ORIGIN,
                rate_func=linear,
                run_time=5
            ),
        )


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


class ConePointsScene(ThreeDScene):
    """Display a cone"""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, zoom=0.40)
        mesh = create_model(name="tail_topper")
        mesh.apply_scale(scaling=0.2)
        mesh.apply_translation([0, -8, 0])
        points_obj = PointCloudObject(mesh=mesh)
        self.add(points_obj)
        self.play(
            Rotate(
                points_obj,
                angle=2 * PI,
                about_point=ORIGIN,
                rate_func=linear,
                run_time=5
            ),
        )


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

class ConeScene2(ThreeDScene):
    """Display a cone"""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, zoom=0.40)
        mesh = create_model(name="tail_topper")
        mesh.apply_scale(scaling=0.3)
        mesh.apply_translation([0, -8, 0])
        manim_mesh_obj = ManimMesh(mesh=mesh)
        self.add(manim_mesh_obj)


class ConePointsScene2(ThreeDScene):
    """Display a cone"""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, zoom=0.40)
        mesh = create_model(name="tail_topper")
        mesh.apply_scale(scaling=0.3)
        mesh.apply_translation([0, -8, 0])
        points_obj = PointCloudObject(mesh=mesh)
        self.add(points_obj)


class SuzanneScene(ThreeDScene):
    """suzanne mesh"""

    def construct(self):
        self.camera.set_phi(90 * DEGREES)
        mesh = create_model(name="suzanne")
        manim_mesh_obj = ManimMesh(mesh=mesh)
        self.add(manim_mesh_obj)
        self.play(
            Rotate(
                manim_mesh_obj,
                angle=2 * PI,
                about_point=ORIGIN,
                rate_func=linear,
                run_time=5
            )
        )


class SuzannePointsScene(ThreeDScene):
    """suzanne as point cloud"""

    def construct(self):
        self.camera.set_phi(90 * DEGREES)
        mesh = create_model(name="suzanne")
        points_obj = PointCloudObject(mesh=mesh)
        self.add(points_obj)
        self.play(
            Rotate(
                points_obj,
                angle=2 * PI,
                about_point=ORIGIN,
                rate_func=linear,
                run_time=5
            )
        )


class PyramidScene2(ThreeDScene):
    """pyramid mesh, changes a face color"""

    def construct(self):
        self.set_camera_orientation(70 * DEGREES, 30 * DEGREES)
        mesh = create_pyramid()
        manim_mesh_obj = ManimMesh(mesh=mesh)
        self.add(manim_mesh_obj)
        self.play(
            manim_mesh_obj.get_face(0).animate.set_fill(RED, 0.8)
        )


class PyramidPointsScene2(ThreeDScene):
    """pyramid as point cloud, changes a point color"""

    def construct(self):
        self.set_camera_orientation(70 * DEGREES, 30 * DEGREES)
        mesh = create_pyramid()
        points_obj = PointCloudObject(mesh=mesh)
        self.add(points_obj)
        self.play(
            points_obj.get_point(0).animate.set_color(YELLOW)
        )


class TriangleScene(ThreeDScene):
    """pyramid as point cloud, changes a point color"""

    def construct(self):
        mesh = create_coplanar_triangles()

        mesh_2d = Manim2DMesh(mesh=mesh)
        self.add(mesh_2d)
        t1 = mesh_2d.get_face(0)
        self.play(t1.animate.set_fill(GREEN, 0.6))
        c = mesh_2d.get_circle(0)
        self.play(Create(c))
        ps = mesh_2d.get_points_violating_delaunay(0)
        for p in ps:
            self.play(FadeIn(p))
        self.wait()
