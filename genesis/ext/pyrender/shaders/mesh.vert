#version 330 core

// The color, depth, segmentation and normal passes use different programs. They compute the position with the same
// expression and declare it invariant, or a driver may round it differently in each, so that their images would
// disagree on the pixels along edges.
invariant gl_Position;

// Vertex Attributes
layout(location = 0) in vec3 position;
#ifdef NORMAL_LOC
layout(location = NORMAL_LOC) in vec3 normal;
#endif
#ifdef TEXCOORD_0_LOC
layout(location = TEXCOORD_0_LOC) in vec2 texcoord_0;
#endif
#ifdef COLOR_0_LOC
layout(location = COLOR_0_LOC) in vec4 color_0;
#endif
layout(location = INST_M_LOC) in mat4 inst_m;
layout(location = INST_ENV_OFFSET_LOC) in vec3 inst_env_offset;
#ifdef INST_TINT_LOC
// The tint of the instance (see Primitive.inst_tints)
layout(location = INST_TINT_LOC) in vec4 inst_tint;
#endif

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// 1 in a pass drawing the environments side by side, which moves each instance by its environment offset, 0 otherwise
uniform float env_offset_scale;

// Outputs
#ifdef DOUBLE_SIDED
    out vec3 v_frag_position;
    #ifdef NORMAL_LOC
    out vec3 v_frag_normal;
    #endif
    #ifdef TEXCOORD_0_LOC
    out vec2 v_uv_0;
    #endif
    #ifdef COLOR_0_LOC
    out vec4 v_color_multiplier;
    #endif
    #ifdef INST_TINT_LOC
    flat out vec4 v_tint;
    #endif
#else
    out vec3 frag_position;
    #ifdef NORMAL_LOC
    out vec3 frag_normal;
    #endif
    #ifdef TEXCOORD_0_LOC
    out vec2 uv_0;
    #endif
    #ifdef COLOR_0_LOC
    out vec4 color_multiplier;
    #endif
    #ifdef INST_TINT_LOC
    flat out vec4 tint;
    #endif
#endif


void main()
{
    vec4 world_position = M * inst_m * vec4(position, 1);
    world_position.xyz += env_offset_scale * inst_env_offset;
    gl_Position = P * V * world_position;

#ifdef DOUBLE_SIDED
    v_frag_position = world_position.xyz;

    mat4 N = transpose(inverse(M * inst_m));

    #ifdef NORMAL_LOC
        v_frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
    #endif

    #ifdef TEXCOORD_0_LOC
        v_uv_0 = texcoord_0;
    #endif
    #ifdef COLOR_0_LOC
        v_color_multiplier = color_0;
    #endif
    #ifdef INST_TINT_LOC
        v_tint = inst_tint;
    #endif
#else
    frag_position = world_position.xyz;

    mat4 N = transpose(inverse(M * inst_m));

    #ifdef NORMAL_LOC
        frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
    #endif

    #ifdef TEXCOORD_0_LOC
        uv_0 = texcoord_0;
    #endif
    #ifdef COLOR_0_LOC
        color_multiplier = color_0;
    #endif
    #ifdef INST_TINT_LOC
        tint = inst_tint;
    #endif
#endif
}
