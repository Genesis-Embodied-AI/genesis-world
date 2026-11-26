#version 330 core

// A triangle covering the viewport, drawn without vertex buffer
void main()
{
    vec2 corner = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(2.0 * corner - 1.0, 0.0, 1.0);
}
