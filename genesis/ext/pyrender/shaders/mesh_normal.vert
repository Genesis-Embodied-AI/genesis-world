#version 330 core

// See the invariance of the position in mesh.vert
invariant gl_Position;

// Vertex Attributes
layout(location = 0) in vec3 position;
#ifdef NORMAL_LOC
layout(location = NORMAL_LOC) in vec3 normal;
#endif
layout(location = INST_M_LOC) in mat4 inst_m;
layout(location = INST_ENV_OFFSET_LOC) in vec3 inst_env_offset;

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
uniform float env_offset_scale;

// Outputs
#ifdef DOUBLE_SIDED
    out vec3 v_frag_position;
    out vec3 v_frag_normal;
#else
    out vec3 frag_position;
    out vec3 frag_normal;
#endif

void main()
{
    vec4 world_position = M * inst_m * vec4(position, 1);
    world_position.xyz += env_offset_scale * inst_env_offset;
    gl_Position = P * V * world_position;
#ifdef NORMAL_LOC
    vec3 world_normal = normalize(vec3(transpose(inverse(M * inst_m)) * vec4(normal, 0.0)));
#else
    // A primitive without normals faces the camera, as it is shaded (see 'get_normal' in mesh.frag)
    vec3 world_normal = normalize(inverse(V)[3].xyz - world_position.xyz);
#endif

#ifdef DOUBLE_SIDED
    v_frag_position = world_position.xyz;
    v_frag_normal = world_normal;
#else
    frag_position = world_position.xyz;
    frag_normal = world_normal;
#endif
}