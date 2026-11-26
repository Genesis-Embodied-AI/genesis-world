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
#ifdef TRANSMITTANCE
uniform float ior;
#endif
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
    // A surface blocks the share of the light that it neither lets through its transparent texels nor transmits. The
    // shadow map holds the texels blocking all of it, sharp at its full resolution, and the transmittance map what the
    // others let through, which multiplies over the surfaces a ray of the light crosses (see '_transmittance_pass' in
    // renderer.py).
    float opacity = base_color.a * (1.0 - transmission_factor);
    bool is_opaque = opacity > 0.999;
#ifdef TRANSMITTANCE
    if (is_opaque) {
        discard;
    }
    // The light past the covered share of the surface is tinted by its linear color (see srgb_to_linear in mesh.frag)
    // as much as its alpha covers, and the light its material transmits by its color alone, less what its interface
    // reflects head-on. The alpha keeps the depth of the nearest surface.
    vec3 tint_linear = pow((base_color.rgb + 0.055) / 1.055, vec3(2.4));
    tint_linear = mix(base_color.rgb / 12.92, tint_linear, step(0.04045, base_color.rgb));
    float ior_ratio = (ior - 1.0) / (ior + 1.0);
    vec3 passed = (1.0 - base_color.a) * mix(vec3(1.0), tint_linear, base_color.a);
    vec3 transmitted = base_color.a * transmission_factor * (1.0 - ior_ratio * ior_ratio) * tint_linear;
#ifdef POINT_SHADOW
    float depth = length(frag_pos - light_pos) / 25.0;
#else
    float depth = gl_FragCoord.z;
#endif
    frag_color = vec4(passed + transmitted, depth);
    return;
#else
    if (!is_opaque) {
        discard;
    }
#endif
#endif
#ifdef POINT_SHADOW
    gl_FragDepth = length(frag_pos - light_pos) / 25.0;
#endif
    frag_color = vec4(1.0);
}
