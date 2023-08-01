#version 450
// https://stackoverflow.com/a/59739538/782170

layout(location = 0) out vec2 texcoord; // texcoords are in the normalized [0,1] range for the viewport-filling quad part of the triangle
void main() {
        vec2 vertices[3]=vec2[3](vec2(-1,-1), vec2(3,-1), vec2(-1, 3));
        gl_Position = vec4(vertices[gl_VertexIndex],0,1);
        texcoord = 0.5 * gl_Position.xy + vec2(0.5);
}