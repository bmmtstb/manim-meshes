#version 330

#include ../include/camera_uniform_declarations.glsl

in vec3 point;
in vec4 color;

out vec3 xyz_coords;
out vec4 v_color;

#include ../include/position_point_into_frame.glsl
#include ../include/get_gl_Position.glsl

void main(){
    xyz_coords = position_point_into_frame(point);
    v_color = color;
    gl_Position = get_gl_Position(xyz_coords);
}
