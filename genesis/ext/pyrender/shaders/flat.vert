#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
#ifdef NORMAL_LOC
layout(location = NORMAL_LOC) in vec3 normal;
#endif
#ifdef TANGENT_LOC
layout(location = TANGENT_LOC) in vec4 tangent;
#endif
#ifdef TEXCOORD_0_LOC
layout(location = TEXCOORD_0_LOC) in vec2 texcoord_0;
#endif
#ifdef TEXCOORD_1_LOC
layout(location = TEXCOORD_1_LOC) in vec2 texcoord_1;
#endif
#ifdef COLOR_0_LOC
layout(location = COLOR_0_LOC) in vec4 color_0;
#endif
#ifdef JOINTS_0_LOC
layout(location = JOINTS_0_LOC) in vec4 joints_0;
#endif
#ifdef WEIGHTS_0_LOC
layout(location = WEIGHTS_0_LOC) in vec4 weights_0;
#endif
layout(location = INST_M_LOC) in mat4 inst_m;
layout(location = INST_ENV_OFFSET_LOC) in vec3 inst_env_offset;

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
uniform float env_offset_scale;

// Outputs
out vec3 frag_position;
#ifdef NORMAL_LOC
out vec3 frag_normal;
#endif
#ifdef HAS_NORMAL_TEX
#ifdef TANGENT_LOC
#ifdef NORMAL_LOC
out mat3 tbn;
#endif
#endif
#endif
#ifdef TEXCOORD_0_LOC
out vec2 uv_0;
#endif
#ifdef TEXCOORD_1_LOC
out vec2 uv_1;
#endif
#ifdef COLOR_0_LOC
out vec4 color_multiplier;
#endif


void main()
{
    vec4 world_position = M * inst_m * vec4(position, 1);
    world_position.xyz += env_offset_scale * inst_env_offset;
    gl_Position = P * V * world_position;
    frag_position = world_position.xyz;

    mat4 N = transpose(inverse(M * inst_m));

#ifdef NORMAL_LOC
    frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
#endif

#ifdef HAS_NORMAL_TEX
#ifdef TANGENT_LOC
#ifdef NORMAL_LOC
    vec3 normal_w = normalize(vec3(N * vec4(normal, 0.0)));
    vec3 tangent_w = normalize(vec3(N * vec4(tangent.xyz, 0.0)));
    vec3 bitangent_w = cross(normal_w, tangent_w) * tangent.w;
    tbn = mat3(tangent_w, bitangent_w, normal_w);
#endif
#endif
#endif
#ifdef TEXCOORD_0_LOC
    uv_0 = texcoord_0;
#endif
#ifdef TEXCOORD_1_LOC
    uv_1 = texcoord_1;
#endif
#ifdef COLOR_0_LOC
    color_multiplier = color_0;
#endif
}
