#version 330 core

// The image of a fisheye camera, resampled from the cube map of what the camera sees all around it (see FisheyeCamera
// in camera.py). The faces of the cube map are rendered in the frame of the camera.

const float PI = 3.14159265358979;

uniform samplerCube color_cube;
uniform samplerCube depth_cube;
// The focal lengths and the principal point (fx, fy, cx, cy), in pixels from the top-left corner of the image
uniform vec4 intrinsics;
// The distortion coefficients (k1, k2, k3, k4) of the Kannala-Brandt model
uniform vec4 distortion;
// The clipping distances of the camera, the far plane of the faces of the cube map, and the height of the image
uniform vec4 depth_range;
// The width and height of each face of the cube map, in texels
uniform int face_size;

out vec4 frag_color;

void main()
{
    // The pixel from the top-left corner of the image, the rows going down as OpenCV counts them
    vec2 pixel = vec2(gl_FragCoord.x, depth_range.w - gl_FragCoord.y);
    vec2 xy = (pixel - intrinsics.zw) / intrinsics.xy;
    float theta_d = length(xy);

    // The angle of the ray to the line of sight, the root of theta * (1 + k1 theta^2 + ... + k4 theta^8) = theta_d,
    // which a few Newton iterations reach from theta_d for the small coefficients of a calibrated lens
    float theta = theta_d;
    for (int i = 0; i < 10; i++) {
        float theta2 = theta * theta;
        float residual = theta * (1.0 + theta2 * (distortion.x + theta2 * (distortion.y + theta2 * (distortion.z
            + theta2 * distortion.w)))) - theta_d;
        float slope = 1.0 + theta2 * (3.0 * distortion.x + theta2 * (5.0 * distortion.y + theta2 * (7.0 * distortion.z
            + theta2 * 9.0 * distortion.w)));
        theta -= residual / slope;
    }

    // The ray in the frame of the camera, which looks along -z with y up, while OpenCV looks along z with y down
    vec2 radial = theta_d > 0.0 ? xy / theta_d : vec2(0.0);
    vec3 dir = vec3(radial.x * sin(theta), -radial.y * sin(theta), -cos(theta));

    // The cube maps are sampled before any fragment is discarded, which keeps the level of detail of the filtered
    // color well defined
    vec4 color = texture(color_cube, dir);
    float depth = texture(depth_cube, dir).r;
    if (theta > PI || depth >= 1.0) {
        discard;
    }

    // The depth along the axis of the face, then along the ray through the centre of the texel read, which holds the
    // depth of the surface along that ray. Converting it along the ray of the pixel instead would err by as much as
    // the slant of the ray across the texel. A pixel farther than the far clipping distance shows the background, as
    // if nothing lay there.
    float near = depth_range.x;
    float far = depth_range.y;
    float face_far = depth_range.z;
    float face_depth = 2.0 * near * face_far / (face_far + near - (2.0 * depth - 1.0) * (face_far - near));
    vec3 dir_abs = abs(dir);
    float major = max(dir_abs.x, max(dir_abs.y, dir_abs.z));
    vec2 minor = dir_abs.x == major ? dir.yz : (dir_abs.y == major ? dir.xz : dir.xy);
    vec2 texel = (floor((0.5 * minor / major + 0.5) * float(face_size)) + 0.5) / float(face_size) * 2.0 - 1.0;
    float dist = face_depth * sqrt(1.0 + dot(texel, texel));
    if (dist > far) {
        discard;
    }

    frag_color = color;
    // The depth that a perspective projection with the same clipping planes stores for a point at that distance, which
    // the reading of the depth image converts back into the distance
    gl_FragDepth = 0.5 * ((far + near) / (far - near) - 2.0 * far * near / ((far - near) * dist)) + 0.5;
}
