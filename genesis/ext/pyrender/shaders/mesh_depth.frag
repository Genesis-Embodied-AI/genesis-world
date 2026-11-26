#version 330 core

// A point light stores in its cube map the distance of each fragment to it, scaled to the depth range
#ifdef POINT_SHADOW
in vec3 frag_pos;
uniform vec3 light_pos;
#endif

#ifdef TRANSPARENT_SHADOW
#ifdef TEXCOORD_0_LOC
in vec2 uv_0;
#endif
#ifdef COLOR_0_LOC
in vec4 color_multiplier;
#endif

uniform vec4 base_color_factor;
uniform float transmission_factor;
#ifdef HAS_BASE_COLOR_TEX
#ifdef TEXCOORD_0_LOC
#ifdef BASE_COLOR_TEX_ARRAY
uniform sampler2DArray base_color_texture;
uniform float base_color_layer;
#else
uniform sampler2D base_color_texture;
#endif
#endif
#endif

// 4x4 ordered dithering thresholds
const float BAYER[16] = float[](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
#endif

out vec4 frag_color;

void main()
{
#ifdef TRANSPARENT_SHADOW
    vec4 base_color = base_color_factor;
#ifdef HAS_BASE_COLOR_TEX
#ifdef TEXCOORD_0_LOC
#ifdef BASE_COLOR_TEX_ARRAY
    vec4 base_color_texel = texture(base_color_texture, vec3(uv_0, base_color_layer));
#ifdef BASE_COLOR_TEX_LINEAR
    // A texture array holding linear values is encoded to sRGB, the encoding of every other base color (see
    // linear_to_srgb in mesh.frag)
    vec3 texel_srgb = 1.055 * pow(base_color_texel.rgb, vec3(1.0 / 2.4)) - 0.055;
    base_color_texel.rgb = mix(12.92 * base_color_texel.rgb, texel_srgb, step(0.0031308, base_color_texel.rgb));
#endif
    base_color *= base_color_texel;
#else
    base_color *= texture(base_color_texture, uv_0);
#endif
#endif
#endif
#ifdef COLOR_0_LOC
    base_color *= color_multiplier;
#endif
    // A surface blocks the share of the light that it neither lets through its transparent texels nor transmits, the
    // light it transmits being tinted by its linear color (see srgb_to_linear in mesh.frag). The shadow map keeps that
    // share of its fragments in an ordered pattern, which the filtering of the shadow lookups averages into a partial
    // shadow.
    vec3 tint_linear = pow((base_color.rgb + 0.055) / 1.055, vec3(2.4));
    tint_linear = mix(base_color.rgb / 12.92, tint_linear, step(0.04045, base_color.rgb));
    float opacity = base_color.a * (1.0 - transmission_factor * dot(tint_linear, vec3(1.0 / 3.0)));
    ivec2 pixel = ivec2(gl_FragCoord.xy) % 4;
    if (opacity <= (BAYER[4 * pixel.y + pixel.x] + 0.5) / 16.0) {
        discard;
    }
#endif
#ifdef POINT_SHADOW
    gl_FragDepth = length(frag_pos - light_pos) / 25.0;
#endif
    frag_color = vec4(1.0);
}
