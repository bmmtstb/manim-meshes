"""
create a few sample test scenes to check efficiency of renderer
"""
# pylint: skip-file
# pylint: disable-all
# python imports
from copy import deepcopy
# third-party imports
import manim as m
import numpy as np
# local imports
from manim_meshes.delaunay.delaunay_criterion import get_point_indices_violating_delaunay, is_point_violating_delaunay, \
    get_circum_circle
from manim_meshes.delaunay.divide_and_conquer import DivideAndConquer
from manim_meshes.delaunay.voronoi import VoronoiDelaunay
from manim_meshes.models.data_models.mesh import Mesh
from manim_meshes.models.manim_models.basic_mesh import ManimMesh, Manim2DMesh
from manim_meshes.models.manim_models.opengl_mesh import FastManimMesh
from manim_meshes.models.manim_models.triangle_mesh import TriangleManim2DMesh
from manim_meshes.templates import create_grid, create_pyramid, create_model, create_coplanar_triangles, \
    create_coplanar_points


# some test scenes

# preview: manim --renderer=opengl -p faster_models.py FastMeshTest
class FastManimMeshScene(m.ThreeDScene):
    """render / display FastManimMesh"""

    def construct(self):
        self.camera.set_phi(90 * m.DEGREES)
        mesh = create_model(name="armadillo")
        mesh.apply_rotation(90 * m.DEGREES, m.RIGHT)
        mesh.scale_mesh(0.03)
        fast_manim_mesh = FastManimMesh(mesh=mesh, color=m.GREEN_E)
        self.add(fast_manim_mesh)


# #### ManimMesh #####
class ConeScene(m.ThreeDScene):
    """Display a cone"""

    def construct(self):
        self.set_camera_orientation(phi=70 * m.DEGREES)
        mesh = create_model(name="tail_topper")
        mesh.scale_mesh(scaling=0.2)
        mesh.translate_mesh(np.array([0, -5, 0]))
        manim_mesh_obj = ManimMesh(
            mesh=mesh
        )
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
                angle=3 * m.PI,
                about_point=m.ORIGIN,
                rate_func=m.linear,
                run_time=21
            )
        )


# run in manim-meshes
# preview: manim -p --renderer=opengl tests/test_scene.py PyramidScene
# use --write_to_movie instead of -p to render to file
class PyramidScene(m.ThreeDScene):
    """pyramid mesh, changes face color, moves vertices, rotates mesh"""

    def construct(self):
        self.set_camera_orientation(70 * m.DEGREES, 30 * m.DEGREES)
        # pyramid with only triangles
        manim_triangle_pyramid = ManimMesh(mesh=create_pyramid(triangles_only=True), display_vertices=True, name="Tri Mesh")
        manim_triangle_pyramid.shift(m.OUT)
        self.add(manim_triangle_pyramid)
        # pyramid with triangles and a quad bottom to test the renderer
        manim_quad_pyramid = ManimMesh(mesh=create_pyramid(triangles_only=False), display_vertices=True, name="Quad Mesh")
        manim_quad_pyramid.shift(m.IN, m.IN)
        self.add(manim_quad_pyramid)
        self.play(
            m.Rotate(manim_triangle_pyramid, angle=2 * m.PI, about_point=m.ORIGIN, run_time=2.0),
            m.Rotate(manim_quad_pyramid, angle=2 * m.PI, about_point=m.ORIGIN, run_time=2.0),
        )
        # ATTENTION: default animation m.Rotate does not work (yet) properly with our implementation
        # currently you have to call .mesh.apply_rotation with the same parameters after the m.Rotate is played
        manim_triangle_pyramid.mesh.apply_rotation(angle=2 * m.PI, about_point=m.ORIGIN)
        manim_quad_pyramid.mesh.apply_rotation(angle=2 * m.PI, about_point=m.ORIGIN)
        # color faces
        self.play(
            manim_triangle_pyramid.get_face(0).animate.set_fill(m.RED, 1),
            manim_quad_pyramid.get_face(4).animate.set_fill(m.GREEN, 1),
        )
        # test re-rendering of meshes
        manim_triangle_pyramid.move_vertex_to(vertex_idx=0, scene=self, pos=np.array([2, 2, 1])),
        manim_quad_pyramid.move_vertex_to(vertex_idx=1, scene=self, pos=np.array([2, -1.5, -1.5])),
        self.play(
            m.Rotate(manim_triangle_pyramid, angle=2 * m.PI, about_point=m.ORIGIN, run_time=2.0),
            m.Rotate(manim_quad_pyramid, angle=2 * m.PI, about_point=m.ORIGIN, run_time=2.0),
        )
        # ATTENTION, see above
        manim_triangle_pyramid.mesh.apply_rotation(angle=2 * m.PI, about_point=m.ORIGIN)
        manim_quad_pyramid.mesh.apply_rotation(angle=2 * m.PI, about_point=m.ORIGIN)


# run in manim-meshes
# preview: manim -p --renderer=opengl tests/test_scene.py TriangleScene
# use --write_to_movie instead of -p to render to file
class TriangleScene(m.ThreeDScene):
    """simple 2D mesh scene, visualizes delaunay criterion"""

    # pylint: disable=too-many-statements
    def construct(self):
        text = m.Text('Delaunay Example').scale(0.5).to_corner(m.UL)
        text.set_color(m.WHITE)
        text.fix_in_frame()
        self.add(text)
        mesh = create_coplanar_triangles()
        mesh_2d = TriangleManim2DMesh(mesh=mesh)
        self.add(mesh_2d)
        triangle = mesh_2d.get_face(0)
        self.play(triangle.animate.set_fill(m.YELLOW_D, None))  # mark triangle
        circle = get_circum_circle(mesh_2d, 0)  # circumcircle around triangle
        self.play(m.Create(circle))
        indices = get_point_indices_violating_delaunay(mesh_2d, 0)  # vertex indices
        points = mesh_2d.get_dots(indices)
        assert len(points) == 1  # should be one in this example
        self.play(m.FadeIn(points[0]))
        # add updater which colors the circle and point green if it is not violating the delaunay criteria
        # w.r.t. triangle (face index 0)
        points[0].add_updater(lambda mo: (mo.set_color(m.GREEN), circle.set_color(m.GREEN)) \
            if not is_point_violating_delaunay(mesh_2d, indices[0], 0) else None)
        # use mesh_2d.move_vertex_to and mesh_2d.shift_vertex instead of e.g. self.play(points[0].animate.move_to)
        # -> otherwise the faces will not be updated
        mesh_2d.shift_vertex(self, indices[0], 0.35 * m.DL[:2])
        points[0].remove_updater(points[0].non_time_updaters[-1])  # remove last updater
        self.play(m.FadeOut(points[0]), m.Uncreate(circle))
        self.play(triangle.animate.set_fill(mesh_2d.faces_color, None))  # unmark triangle
        # check delaunay for each triangle (except first ~> already checked above)
        for f in range(1, len(mesh_2d.mesh.faces)):
            circ = get_circum_circle(mesh_2d, f)  # circumcircle around triangle
            self.play(m.Create(circ, run_time=0.4))
            points = mesh_2d.get_dots(get_point_indices_violating_delaunay(mesh_2d, f))
            if len(points) == 0:  # no violating points, mark circle green
                self.play(circ.animate(run_time=0.4).set_color(m.GREEN))
            self.play(m.Uncreate(circ, run_time=0.4))
        # demonstrate edge flip
        self.wait(0.5)
        triangle_a = mesh_2d.get_face(2)
        triangle_b = mesh_2d.get_face(3)
        self.play(triangle_a.animate.set_fill(m.YELLOW_D, None), triangle_b.animate.set_fill(m.YELLOW_D, None))
        mesh_2d.edge_flip(self, 2, 3)
        circle_a = get_circum_circle(mesh_2d, 2)  # circumcircle around triangle
        circle_b = get_circum_circle(mesh_2d, 3)  # circumcircle around triangle
        self.play(m.Create(circle_a), m.Create(circle_b), self.camera.animate.shift(m.DOWN))
        points_a = mesh_2d.get_dots(get_point_indices_violating_delaunay(mesh_2d, 2))
        points_b = mesh_2d.get_dots(get_point_indices_violating_delaunay(mesh_2d, 3))
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
        mesh_2d.edge_flip(self, 2, 3)
        # get new circles
        circle_a = get_circum_circle(mesh_2d, 2)  # circumcircle around triangle
        circle_b = get_circum_circle(mesh_2d, 3)  # circumcircle around triangle
        self.play(m.Create(circle_a), m.Create(circle_b))
        # mark circle green, then remove them
        self.play(circle_a.animate.set_color(m.GREEN), circle_b.animate.set_color(m.GREEN))
        self.play(m.Uncreate(circle_a), m.Uncreate(circle_b))
        # unmark triangles
        self.play(triangle_a.animate.set_fill(mesh_2d.faces_color, None),
                  triangle_b.animate.set_fill(mesh_2d.faces_color, None))


# run in manim-meshes
# preview: manim -p --renderer=opengl tests/test_scene.py DivideAndConquerScene
# use --write_to_movie instead of -p to render to file
class DivideAndConquerScene(m.ThreeDScene):
    """simple 2D mesh scene, visualizes delaunay divide & conquer algorithm"""

    # pylint: disable=too-many-statements
    def construct(self):
        text = m.Text('Divide and Conquer').scale(0.5).to_corner(m.UL)
        text.set_color(m.WHITE)
        text.fix_in_frame()
        self.add(text)
        mesh = create_coplanar_points()
        # make sure TriangleManim2DMesh only consists of vertices / no faces and
        # that display_edges=True, display_vertices=True, else the algorithm is not visualized properly
        mesh_2d = TriangleManim2DMesh(mesh=mesh, display_vertices=True, display_edges=True,
                                      edges_color=m.BLACK, faces_color=m.BLUE_D)
        self.add(mesh_2d)
        self.wait(0.5)
        dac = DivideAndConquer(self, mesh_2d)
        # runs the complete algorithm, you could also write a modified version of this method to add additional objects
        # like explanatory text (e.g. create subclass of DivideAndConquer and overwrite divide_and_conquer_recursive) or
        # use the other methods like dac.split_points, dac.triangulate_leq_3 and dac.merge_sets to implement the
        # individual steps of the algorithm yourself
        dac.divide_and_conquer_recursive()
        self.wait(3)


# run in manim-meshes
# preview: manim -p --renderer=opengl tests/test_scene.py VoronoiDelaunayScene
# use --write_to_movie instead of -p to render to file
class VoronoiDelaunayScene(m.ThreeDScene):
    """simple 2D mesh scene, visualizes duality of voronoi diagram and delaunay triangulation"""

    # pylint: disable=too-many-statements
    def construct(self):
        text = m.Text('Voronoi & Delaunay').scale(0.5).to_corner(m.DL)
        text.set_color(m.WHITE)
        text.fix_in_frame()
        self.add(text)
        mesh = create_coplanar_points()
        # make sure TriangleManim2DMesh only consists of vertices / no faces and
        # that display_edges=True, display_vertices=True, else the algorithm is not visualized properly
        mesh_2d = TriangleManim2DMesh(mesh=mesh, display_vertices=True, display_edges=True,
                                      edges_color=m.BLACK, faces_color=m.BLUE_D)
        self.add(mesh_2d)
        self.wait(0.5)
        vd = VoronoiDelaunay(self, mesh_2d)
        voronoi_vertices, voronoi_lines = vd.create_voronoi()  # manim Dot and Line objects
        self.play(m.FadeIn(voronoi_vertices, voronoi_lines))
        self.wait(0.3)
        # create delaunay triangulation from voronoi diagram
        for idx, manim_vert in enumerate(voronoi_vertices):
            # voronoi vertices are centers of circum-circles of the delaunay triangles, show circle,
            # then create triangle
            circle = vd.get_circum_circle(idx)
            self.play(m.Create(circle))
            self.wait(0.3)
            vd.create_triangle(idx)
            self.wait(0.3)
            self.play(m.Uncreate(circle))
        self.wait(3)


# manim --renderer=opengl --write_to_movie tests/test_scene.py SnapToGridScene
class SnapToGridScene(m.ThreeDScene):
    """
    A basic example of a 2D grid with some points that are close to but not on the grid.
    Explaining the overall process of grid-snapping
    """

    def construct(self):
        self.camera.scale(0.3)  # camera settings so three points are better visible
        # header text
        header = m.Text('Snap To Grid').scale(0.5).to_corner(0.6 * m.LEFT + 0.4 * m.UL)
        header.set_color(m.WHITE)
        header.fix_in_frame()
        self.add(header)
        # descriptive text
        description = m.Text(
            "- Regular 1x1 grid \n- high threshold of 0.3 \n- move the vertices to the grid"
        ).scale(0.3).to_corner(m.UL)
        description.set_color(m.WHITE)
        description.fix_in_frame()
        self.add(description)
        # create mesh for grid
        grid_mesh = Manim2DMesh(
            mesh=create_grid([(-2, 2, 5), (-2, 2, 5)]),
            display_vertices=False, display_edges=True, display_faces=False,
            faces_opacity=0.0,
            edges_color=m.WHITE,
            edges_width=0.1,
        )
        self.add(grid_mesh)
        # generate "random" points and define their color for later
        p1 = [0.2, 0.1]
        p2 = [0.5, 0.65]
        p3 = [-1.2, 0.4]
        other_vertices = np.array([[1, 1], [-1, 1.1], [-1.5, -1.4], [1.6, 1.7], [1.1, 1.5], [-0.15, -0.75],
                                   [-0.15, 0.05], [-0.15, -1.15], [0.15, -1.15], [1.5, -1.5], [-0.95, 0.4]])
        vertices_color = [2, 0, 1, 2, 2, 0, 1, 1, 2, 2, 2, 2, 0]
        # create point mesh
        vertex_mesh = Manim2DMesh(
            mesh=Mesh(vertices=np.array([p1, p2, p3]), faces=None, parts=None),
            display_vertices=True, display_faces=False, display_edges=False,
            clear_vertices=True,
            verts_color=m.YELLOW_E,
            verts_size=0.04,
        )
        self.add(vertex_mesh)
        self.wait(0.5)
        # show Braces - manim needs 3D vertices
        # define horizontal braces
        brace1h = m.BraceBetweenPoints([0.0, p1[1], 0.0], p1 + [0], color=m.PURE_GREEN)
        brace2h = m.BraceBetweenPoints(p2 + [0], [0.0, p2[1], 0.0], color=m.RED)
        brace3h = m.BraceBetweenPoints([-1.0, p3[1], 0.0], p3 + [0], color=m.PURE_GREEN)
        # define vertical braces
        brace1v = m.BraceBetweenPoints(p1 + [0], [p1[0], 0.0, 0.0], color=m.PURE_GREEN)
        brace2v = m.BraceBetweenPoints(p2 + [0], [p2[0], 1.0, 0.0], color=m.RED)
        brace3v = m.BraceBetweenPoints(p3 + [0], [p3[0], 0.0, 0.0], color=m.RED)
        # show distances as braces in colors
        self.play(m.FadeIn(brace1h), m.FadeIn(brace2h), m.FadeIn(brace3h))
        self.wait(0.5)
        self.play(m.FadeOut(brace1h), m.FadeOut(brace2h), m.FadeOut(brace3h))
        self.wait(0.5)
        self.play(m.FadeIn(brace1v), m.FadeIn(brace2v), m.FadeIn(brace3v))
        self.wait(0.5)
        # color points
        self.play(
            m.FadeOut(brace2v),
            m.FadeIn(brace1h),
            m.FadeIn(brace3h),
            vertex_mesh.get_vertex(0).animate.set_color(m.PURE_GREEN),  # p1
            vertex_mesh.get_vertex(1).animate.set_color(m.RED),  # p2
            vertex_mesh.get_vertex(2).animate.set_color(m.GREEN_E),  # p3
        )
        self.wait(0.5)
        self.play(m.FadeOut(brace1h), m.FadeOut(brace1v), m.FadeOut(brace3h), m.FadeOut(brace3v))
        # zoom out to be able to show all the vertices
        self.play(self.camera.animate.scale(2.0))  # previously 0.3
        # add additional points to mesh. then show and color them
        vertex_mesh.mesh.add_vertices(other_vertices)
        colors = {0: m.RED, 1: m.GREEN_E, 2: m.PURE_GREEN}
        # fade out current ones, fade in all vertices, color everything
        self.play(
            m.FadeOut(vertex_mesh.vertices),
            m.FadeIn(vertex_mesh.setup_vertices()),
            *[vertex_mesh.get_vertex(i).animate.set_color(colors[other_color])
              for i, other_color in enumerate(vertices_color)]
        )
        self.wait(0.5)
        # start move to grid
        vertex_mesh.move_to_grid(scene=self, grid_sizes=(1, 1), threshold=(0.3, 0.3), shift_vertices_runtime=3.0)
        # turn all points yellow once more
        self.play(
            *[vertex_mesh.get_vertex(i).animate.set_color(m.YELLOW_E)
              for i in range(len(vertices_color))]
        )
        self.wait(1.0)
        # ---- Fade out everything so far to create blank slate ----
        self.play(*[m.FadeOut(mob) for mob in self.mobjects])
        # explanatory text
        edges_text = m.Text(
            "Moving points, whats the purpose?\n\n"
            "- connect close-by vertices created due to e.g. errors while measuring\n"
            "- this results in less holes and in practice creates more useful\n"
            "  points and faces resulting in a more regular mesh",
            should_center=True, font_size=24
        ).to_corner(m.ORIGIN)
        edges_text.set_color(m.WHITE)
        edges_text.fix_in_frame()
        self.play(m.FadeIn(edges_text))
        self.wait(6.0)  # should be enough time to read?
        self.play(m.FadeOut(edges_text))
        # redefine mesh with edges - no parts
        edges_mesh = Manim2DMesh(
            mesh=Mesh(
                vertices=np.vstack((np.array([p1, p2, p3]), other_vertices)),
                faces=[[0, 11, 12], [5, 10, 13], [2, 9, 8], [2, 4, 9], [1, 4, 9], [1, 3, 4], [3, 4, 7], [3, 6, 7]],
                parts=None,
            ),
            display_vertices=True, display_faces=True, display_edges=True,
            faces_opacity=0.8,
            clear_vertices=True,
            verts_color=m.YELLOW_E,
            verts_size=0.04,
        )
        # create statistics plus table to present them
        converged = deepcopy(edges_mesh.mesh)
        converged.snap_to_grid(grid_sizes=(1, 1), threshold=(0.3, 0.3), update_verts=True)
        statistics = m.IntegerTable(
            [[len(edges_mesh.mesh.split_mesh_into_objects()), len(converged.split_mesh_into_objects())],
             [len(edges_mesh.mesh.vertices), len(converged.vertices)]],
            col_labels=[m.Text("before", fill_color=m.WHITE), m.Text("after", fill_color=m.WHITE)],
            row_labels=[m.Text("nof Meshes", fill_color=m.WHITE), m.Text("nof Nodes", fill_color=m.WHITE)],
        ).align_on_border(0.2 * m.UL).scale(0.2)
        # show points and statistics
        self.play(m.FadeIn(edges_mesh), m.FadeIn(statistics))
        self.wait(1.5)
        # edge mesh move to grid with merging of points
        edges_mesh.move_to_grid(scene=self, grid_sizes=(1, 1), threshold=(0.3, 0.3), shift_vertices_runtime=3)
        self.wait(3.0)
