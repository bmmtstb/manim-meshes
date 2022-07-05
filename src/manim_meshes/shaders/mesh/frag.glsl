#version 330

uniform vec3 light_source_position;
uniform float gloss;
uniform float shadow;

in vec3 g_coords;
in vec3 g_normal;
in vec4 g_color;

out vec4 frag_color;

#include ../include/finalize_color.glsl

void main() {
    frag_color = finalize_color(
        g_color,
        g_coords,
        normalize(g_normal),
        light_source_position,
        gloss,
        shadow
    );
}
