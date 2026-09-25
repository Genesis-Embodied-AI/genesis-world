#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices=6) out;

in vec3 v_frag_position[];
in vec3 v_frag_normal[];
out vec3 frag_position;
out vec3 frag_normal;

void emit_vertex(int i, bool reversed) {
    gl_Position = gl_in[i].gl_Position;
    frag_position = v_frag_position[i];
    frag_normal = reversed ? -v_frag_normal[i] : v_frag_normal[i];
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
