#version 330 core
layout(location = 0) in vec3 position;
layout(location = INST_M_LOC) in mat4 inst_m;
layout(location = INST_ENV_OFFSET_LOC) in vec3 inst_env_offset;

uniform mat4 M;
uniform float env_offset_scale;

void main()
{
    gl_Position = M * inst_m * vec4(position, 1.0);
    gl_Position.xyz += env_offset_scale * inst_env_offset;
}
