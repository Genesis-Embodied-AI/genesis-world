#version 330 core

// See the invariance of the position in mesh.vert
invariant gl_Position;

layout(location = 0) in vec3 position;
layout(location = INST_M_LOC) in mat4 inst_m;
layout(location = INST_ENV_OFFSET_LOC) in vec3 inst_env_offset;

#ifdef TRANSPARENT_SHADOW
#ifdef TEXCOORD_0_LOC
layout(location = TEXCOORD_0_LOC) in vec2 texcoord_0;
out vec2 uv_0;
#endif
#ifdef COLOR_0_LOC
layout(location = COLOR_0_LOC) in vec4 color_0;
out vec4 color_multiplier;
#endif
#endif

#ifdef SHADOW
// The view-projection matrix of the light (see 'set_light' in jit_render.py)
uniform mat4 light_matrix;
#else
uniform mat4 P;
uniform mat4 V;
#endif
uniform mat4 M;
uniform float env_offset_scale;

void main()
{
    vec4 world_position = M * inst_m * vec4(position, 1.0);
    world_position.xyz += env_offset_scale * inst_env_offset;
#ifdef SHADOW
    gl_Position = light_matrix * world_position;
#else
    gl_Position = P * V * world_position;
#endif
#ifdef TRANSPARENT_SHADOW
#ifdef TEXCOORD_0_LOC
    uv_0 = texcoord_0;
#endif
#ifdef COLOR_0_LOC
    color_multiplier = color_0;
#endif
#endif
}
