#version 330 core

// See the invariance of the position in mesh.vert
invariant gl_Position;

layout (triangles) in;
layout (triangle_strip, max_vertices=6) out;

void emit_vertex(int i, bool reversed) {
    gl_Position = gl_in[i].gl_Position;
    EmitVertex();
}

void main() {
    emit_vertex(0, false);
    emit_vertex(1, false);
    emit_vertex(2, false);
    EndPrimitive();

    emit_vertex(0, true);
    emit_vertex(2, true);
    emit_vertex(1, true);
    EndPrimitive();
}
