#version 330

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

#include ../include/camera_uniform_declarations.glsl

in vec3 xyz_coords[3];
in vec4 v_color[3];

#include ../include/position_point_into_frame.glsl

out vec3 g_coords;
out vec3 g_normal;
out vec4 g_color;

#include ../include/get_gl_Position.glsl
#include ../include/get_unit_normal.glsl

void main(){
    for(int i = 0; i < 3; i++){
            g_color = v_color[i];
            g_coords = xyz_coords[i];
            g_normal = get_unit_normal(xyz_coords);
            gl_Position = get_gl_Position(g_coords);
            EmitVertex();
    }
    EndPrimitive();
}
