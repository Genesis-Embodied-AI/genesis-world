#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

#ifdef TRANSPARENT_SHADOW
#ifdef TEXCOORD_0_LOC
in vec2 v_uv_0[];
out vec2 uv_0;
#endif
#ifdef COLOR_0_LOC
in vec4 v_color_multiplier[];
out vec4 color_multiplier;
#endif
#endif

uniform mat4 light_matrix[6];
out vec3 frag_pos;

void main() {
    for (int i = 0; i < 6; i++) {
        gl_Layer = i;
        for (int j = 0 ; j < 3; j++) {
            frag_pos = gl_in[j].gl_Position.xyz;
#ifdef TRANSPARENT_SHADOW
#ifdef TEXCOORD_0_LOC
            uv_0 = v_uv_0[j];
#endif
#ifdef COLOR_0_LOC
            color_multiplier = v_color_multiplier[j];
#endif
#endif
            gl_Position = light_matrix[i] * gl_in[j].gl_Position;
            EmitVertex();
        }
        EndPrimitive();
    }
}
