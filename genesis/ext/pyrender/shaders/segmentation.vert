#version 330 core

// See the invariance of the position in mesh.vert
invariant gl_Position;

// Vertex Attributes
layout(location = 0) in vec3 position;
layout(location = INST_M_LOC) in mat4 inst_m;
layout(location = INST_ENV_OFFSET_LOC) in vec3 inst_env_offset;

// Uniforms
uniform mat4 P;
uniform mat4 V;
uniform mat4 M;
uniform float env_offset_scale;

void main()
{
    vec4 world_position = M * inst_m * vec4(position, 1.0);
    world_position.xyz += env_offset_scale * inst_env_offset;
    gl_Position = P * V * world_position;
}

