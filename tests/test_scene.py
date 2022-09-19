"""
create a few sample test scenes to check efficiency of renderer
"""
# pylint: skip-file
# pylint: disable-all

# third-party imports
import manim as m
# local imports
import numpy as np

from manim_meshes.models.data_models.mesh import Mesh
from manim_meshes.models.manim_models.basic_mesh import ManimMesh, Manim2DMesh
from manim_meshes.templates import create_grid, create_pyramid, create_model, create_coplanar_triangles


# #### ManimMesh #####
class ConeScene(m.ThreeDScene):
    """Display a cone"""

    def construct(self):
        self.set_camera_orientation(phi=70 * m.DEGREES, zoom=0.40)
        mesh = create_model(name="tail_topper")
        mesh.scale_mesh(scaling=0.3)
        mesh.translate_mesh(np.array([0, -8, 0]))
        manim_mesh_obj = ManimMesh(
            scene=self,
            mesh=mesh,
            display_vertices=True,
        )
        self.add(manim_mesh_obj)


class SuzanneScene(m.ThreeDScene):
    """suzanne mesh"""

    def construct(self):
        self.camera.set_phi(90 * m.DEGREES)
        mesh = create_model(name="suzanne")
        manim_mesh_obj = ManimMesh(scene=self, mesh=mesh)
        self.add(manim_mesh_obj)
        self.play(
            m.Rotate(
                manim_mesh_obj,
                angle=3 * m.PI,
                about_point=m.ORIGIN,
                rate_func=m.linear,
                run_time=21
            )
        )


class PyramidScene(m.ThreeDScene):
    """pyramid mesh, changes a face color"""

    def construct(self):
        self.set_camera_orientation(70 * m.DEGREES, 30 * m.DEGREES)
        mesh = create_pyramid()
        manim_mesh_obj = ManimMesh(scene=self, mesh=mesh, display_vertices=True)
        self.add(manim_mesh_obj)
        self.play(
            manim_mesh_obj.get_face(0).animate.set_fill(m.RED, 1)
        )


# run in manim-meshes
# preview: manim -p --renderer=opengl tests/test_scene.py TriangleScene
# use --write_to_file instead of -p to render to file
class TriangleScene(m.ThreeDScene):
    """simple 2D mesh scene, visualizes delaunay criterion"""

    # pylint: disable=too-many-statements
    def construct(self):
        text = m.Text('Delaunay Example').scale(0.5).to_corner(m.UL)
        text.set_color(m.WHITE)
        text.fix_in_frame()
        self.add(text)
        mesh = create_coplanar_triangles()
        mesh_2d = Manim2DMesh(scene=self, mesh=mesh)
        self.add(mesh_2d)
        triangle = mesh_2d.get_face(0)
        self.play(triangle.animate.set_fill(m.YELLOW_D, None))  # mark triangle
        circle = mesh_2d.get_circle(0)  # circumcircle around triangle
        self.play(m.Create(circle))
        points, indices = mesh_2d.get_points_violating_delaunay(0)  # vertex indices and manim point objects
        assert len(points) == 1  # should be one in this example
        self.play(m.FadeIn(points[0]))
        # add updater which colors the circle and point green if it is not violating the delaunay criteria
        # w.r.t. triangle (face index 0)
        points[0].add_updater(lambda mo: (mo.set_color(m.GREEN), circle.set_color(m.GREEN)) \
            if not mesh_2d.is_point_violating_delaunay(indices[0], 0) else None)
        # use mesh_2d.move_vertex_to and mesh_2d.shift_vertex instead of e.g. self.play(points[0].animate.move_to)
        # -> otherwise the faces will not be updated
        mesh_2d.shift_vertex(indices[0], 0.35 * m.DL[:2])
        points[0].remove_updater(points[0].non_time_updaters[-1])  # remove last updater
        self.play(m.FadeOut(points[0]), m.Uncreate(circle))
        self.play(triangle.animate.set_fill(mesh_2d.faces_color, None))  # unmark triangle
        # check delaunay for each triangle (except first ~> already checked above)
        for f in range(1, len(mesh_2d.mesh.get_faces())):
            circ = mesh_2d.get_circle(f)  # circumcircle around triangle
            self.play(m.Create(circ, run_time=0.4))
            points, _ = mesh_2d.get_points_violating_delaunay(f)
            if len(points) == 0:  # no violating points, mark circle green
                self.play(circ.animate(run_time=0.4).set_color(m.GREEN))
            self.play(m.Uncreate(circ, run_time=0.4))
        # demonstrate edge flip
        self.wait(0.5)
        triangle_a = mesh_2d.get_face(2)
        triangle_b = mesh_2d.get_face(3)
        self.play(triangle_a.animate.set_fill(m.YELLOW_D, None), triangle_b.animate.set_fill(m.YELLOW_D, None))
        mesh_2d.edge_flip(2, 3)
        circle_a = mesh_2d.get_circle(2)  # circumcircle around triangle
        circle_b = mesh_2d.get_circle(3)  # circumcircle around triangle
        self.play(m.Create(circle_a), m.Create(circle_b), self.camera.animate.shift(m.DOWN))
        points_a, _ = mesh_2d.get_points_violating_delaunay(2)
        points_b, _ = mesh_2d.get_points_violating_delaunay(3)
        all_points = []
        all_points.extend(points_a)
        all_points.extend(points_b)
        # show points
        for point in all_points:
            self.play(m.FadeIn(point, run_time=0.3), point.animate().scale(2))
            self.play(point.animate(run_time=0.3).scale(1 / 3))
        # remove circles and points
        anims_remove = []
        for point in all_points:
            anims_remove.append(m.FadeOut(point))
        anims_remove.append(m.Uncreate(circle_a))
        anims_remove.append(m.Uncreate(circle_b))
        self.play(*anims_remove)
        # flip
        mesh_2d.edge_flip(2, 3)
        # get new circles
        circle_a = mesh_2d.get_circle(2)  # circumcircle around triangle
        circle_b = mesh_2d.get_circle(3)  # circumcircle around triangle
        self.play(m.Create(circle_a), m.Create(circle_b))
        # mark circle green, then remove them
        self.play(circle_a.animate.set_color(m.GREEN), circle_b.animate.set_color(m.GREEN))
        self.play(m.Uncreate(circle_a), m.Uncreate(circle_b))
        # unmark triangles
        self.play(triangle_a.animate.set_fill(mesh_2d.faces_color, None),
                  triangle_b.animate.set_fill(mesh_2d.faces_color, None))


class SnapToGridScene(m.ThreeDScene):
    """
    A basic example of a 2D grid with some points that are close to but not on the grid.
    Explaining the overall process of grid-snapping
    """

    def construct(self):
        self.set_camera_orientation(0, 0)
        grid_mesh = Manim2DMesh(
            scene=self,
            mesh=create_grid([(-3, 3, 7), (-3, 3, 7)]),
            display_vertices=False,
            faces_stroke_width=0.3,
            faces_stroke_color=m.BLUE_D,
        )
        self.add(grid_mesh)
        # generate "random" points
        vertex_mesh = Manim2DMesh(
            scene=self,
            mesh=Mesh(
                verts=np.array([[1, 1], [0.2, 0.1], [0.5, 0.7], [-1, 1.1], [-2.5, -2.3], [-1.1, 0.2], [2.1, 1.7]]),
                faces=None,
            ),
            display_vertices=True,
            clear_vertices=True,
        )
        self.add(vertex_mesh)
        self.wait(0.5)
        vertex_mesh.move_to_grid(
            scene=self,
            grid_sizes=(1, 1),
            threshold=(0.3, 0.3),
            nof_steps=10,
        )
