#version 330 core
layout(location = 0) in vec3 position;
layout(location = INST_M_LOC) in mat4 inst_m;
layout(location = INST_ENV_OFFSET_LOC) in vec3 inst_env_offset;

// The geometry shader passes the inputs of the fragment shader on to each face of the cube map (see mesh_depth.frag)
#ifdef TRANSPARENT_SHADOW
#ifdef TEXCOORD_0_LOC
layout(location = TEXCOORD_0_LOC) in vec2 texcoord_0;
out vec2 v_uv_0;
#endif
#ifdef COLOR_0_LOC
layout(location = COLOR_0_LOC) in vec4 color_0;
out vec4 v_color_multiplier;
#endif
#endif

uniform mat4 M;
uniform float env_offset_scale;

void main()
{
    gl_Position = M * inst_m * vec4(position, 1.0);
    gl_Position.xyz += env_offset_scale * inst_env_offset;
#ifdef TRANSPARENT_SHADOW
#ifdef TEXCOORD_0_LOC
    v_uv_0 = texcoord_0;
#endif
#ifdef COLOR_0_LOC
    v_color_multiplier = color_0;
#endif
#endif
}
