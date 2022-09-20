"""
functions to create delaunay meshes by divide and conquer
"""
# python imports
# third-party imports
# local imports


def split_points(dots, color_a, color_b):
    """ Splits set of all manim Dots in dots into two equally sized sets with different colors.
        Returns resulting sets and coloring animations"""

    # TODO sort by x-coordinate
    half_index = len(dots)//2
    vertices_a = dots[:half_index]
    vertices_b = dots[half_index:]
    anims = []
    for vertex in vertices_a:
        anims.append(vertex.animate.set_color(color_a))
    for vertex in vertices_b:
        anims.append(vertex.animate.set_color(color_b))
    return vertices_a, vertices_b, anims
