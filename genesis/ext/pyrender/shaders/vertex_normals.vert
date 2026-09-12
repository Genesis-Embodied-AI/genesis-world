#version 330 core

// Inputs
layout(location = 0) in vec3 position;
layout(location = NORMAL_LOC) in vec3 normal;
layout(location = INST_M_LOC) in mat4 inst_m;
layout(location = INST_ENV_OFFSET_LOC) in vec3 inst_env_offset;

// Output data
out VS_OUT {
    vec3 position;
    vec3 normal;
    mat4 mvp;
    vec4 env_offset_clip;
} vs_out;

// Uniform data
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
uniform float env_offset_scale;

// Render loop
void main() {
    vs_out.mvp = P * V * M * inst_m;
    vs_out.env_offset_clip = P * V * vec4(env_offset_scale * inst_env_offset, 0.0);
    vs_out.position = position;
    vs_out.normal = normal;

    gl_Position = vec4(position, 1.0);
}
