#version 330 core
///////////////////////////////////////////////////////////////////////////////
// Structs
///////////////////////////////////////////////////////////////////////////////

struct SpotLight {
    vec3 color;
    float intensity;
    float range;
    vec3 position;
    vec3 direction;
    float light_angle_scale;
    float light_angle_offset;

    #ifdef SPOT_LIGHT_SHADOWS
    sampler2DShadow shadow_map;
    mat4 light_matrix;
    #endif
};

struct DirectionalLight {
    vec3 color;
    float intensity;
    vec3 direction;

    #ifdef DIRECTIONAL_LIGHT_SHADOWS
    sampler2DShadow shadow_map;
    mat4 light_matrix;
    #endif
};

struct PointLight {
    vec3 color;
    float intensity;
    float range;
    vec3 position;

    #ifdef POINT_LIGHT_SHADOWS
    samplerCube shadow_map;
    #endif
};

struct Material {
    vec3 emissive_factor;

#ifdef USE_METALLIC_MATERIAL
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
    float transmission_factor;
    float ior;
#endif

#ifdef USE_GLOSSY_MATERIAL
    vec4 diffuse_factor;
    vec3 specular_factor;
    float glossiness_factor;
#endif

#ifdef HAS_NORMAL_TEX
    sampler2D normal_texture;
#endif
#ifdef HAS_OCCLUSION_TEX
    sampler2D occlusion_texture;
#endif
#ifdef HAS_EMISSIVE_TEX
#ifdef EMISSIVE_TEX_ARRAY
    sampler2DArray emissive_texture;
    float emissive_layer;
#else
    sampler2D emissive_texture;
#endif
#endif
#ifdef HAS_BASE_COLOR_TEX
#ifdef BASE_COLOR_TEX_ARRAY
    sampler2DArray base_color_texture;
    float base_color_layer;
#else
    sampler2D base_color_texture;
#endif
#endif
#ifdef HAS_METALLIC_ROUGHNESS_TEX
    sampler2D metallic_roughness_texture;
#endif
#ifdef HAS_DIFFUSE_TEX
    sampler2D diffuse_texture;
#endif
#ifdef HAS_SPECULAR_GLOSSINESS_TEX
    sampler2D specular_glossiness;
#endif
};

struct PBRInfo {
    float nl;
    float nv;
    float nh;
    float lh;
    float vh;
    float roughness;
    float metallic;
    vec3 f0;
    vec3 c_diff;
};

///////////////////////////////////////////////////////////////////////////////
// Uniforms
///////////////////////////////////////////////////////////////////////////////
uniform Material material;
uniform PointLight point_lights[MAX_POINT_LIGHTS];
uniform int n_point_lights;
uniform DirectionalLight directional_lights[MAX_DIRECTIONAL_LIGHTS];
uniform int n_directional_lights;
uniform SpotLight spot_lights[MAX_SPOT_LIGHTS];
uniform int n_spot_lights;
uniform vec3 cam_pos;
uniform vec3 ambient_light;
#ifdef REFLECTIVE_FLOOR
// The mirror image of the scene through the floor (see '_floor_pass' in renderer.py)
uniform sampler2D floor_tex;
uniform vec2 screen_size;
#endif

uniform mat4 reflection_mat;

#ifdef USE_IBL
uniform samplerCube diffuse_env;
uniform samplerCube specular_env;
#endif

///////////////////////////////////////////////////////////////////////////////
// Inputs
///////////////////////////////////////////////////////////////////////////////

in vec3 frag_position;
#ifdef NORMAL_LOC
in vec3 frag_normal;
#endif
#ifdef HAS_NORMAL_TEX
#ifdef TANGENT_LOC
#ifdef NORMAL_LOC
in mat3 tbn;
#endif
#endif
#endif
#ifdef TEXCOORD_0_LOC
in vec2 uv_0;
#endif
#ifdef TEXCOORD_1_LOC
in vec2 uv_1;
#endif
#ifdef COLOR_0_LOC
in vec4 color_multiplier;
#endif
#ifdef INST_TINT_LOC
flat in vec4 tint;
// Whether the pass draws a debug view, the only views showing the tint (see 'Primitive.inst_tints')
uniform bool debug_view;
#endif

///////////////////////////////////////////////////////////////////////////////
// OUTPUTS
///////////////////////////////////////////////////////////////////////////////

layout(location = 0, index = 0) out vec4 frag_color;
// The share of the background that shows through the fragment. Dual-source blending multiplies the background by it
// for a transmissive material.
layout(location = 0, index = 1) out vec4 frag_transmittance;

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////
const float PI = 3.141592653589793;
const float min_roughness = 0.04;

///////////////////////////////////////////////////////////////////////////////
// Utility Functions
///////////////////////////////////////////////////////////////////////////////
vec4 srgb_to_linear(vec4 srgb)
{
#ifndef SRGB_CORRECTED
    // Fast Approximation
    //vec3 linOut = pow(srgbIn.xyz,vec3(2.2));
    //
    vec3 b_less = step(vec3(0.04045),srgb.xyz);
    vec3 lin_out = mix( srgb.xyz/vec3(12.92), pow((srgb.xyz+vec3(0.055))/vec3(1.055),vec3(2.4)), b_less );
    return vec4(lin_out, srgb.w);
#else
    return srgb;
#endif
}

// The sRGB encoding of IEC 61966-2-1, inverse of srgb_to_linear
vec4 linear_to_srgb(vec4 linear)
{
#ifndef SRGB_CORRECTED
    vec3 b_less = step(vec3(0.0031308), linear.xyz);
    vec3 srgb_out = mix(linear.xyz * vec3(12.92), vec3(1.055) * pow(linear.xyz, vec3(1.0 / 2.4)) - vec3(0.055), b_less);
    return vec4(srgb_out, linear.w);
#else
    return linear;
#endif
}

// Normal computation
vec3 get_normal()
{
#ifdef HAS_NORMAL_TEX

#ifndef HAS_TANGENTS
    vec3 pos_dx = dFdx(frag_position);
    vec3 pos_dy = dFdy(frag_position);
    vec3 tex_dx = dFdx(vec3(uv_0, 0.0));
    vec3 tex_dy = dFdy(vec3(uv_0, 0.0));
    vec3 t = (tex_dy.t * pos_dx - tex_dx.t * pos_dy) / (tex_dx.s * tex_dy.t - tex_dy.s * tex_dx.t);

#ifdef NORMAL_LOC
    vec3 ng = normalize(frag_normal);
#else
    vec3 ng = cross(pos_dx, pos_dy);
#endif

    t = normalize(t - ng * dot(ng, t));
    vec3 b = normalize(cross(ng, t));
    mat3 tbn_n = mat3(t, b, ng);

#else

    mat3 tbn_n = tbn;

#endif

    vec3 n = texture(material.normal_texture, uv_0).rgb;
    n = normalize(tbn_n * ((2.0 * n - 1.0) * vec3(1.0, 1.0, 1.0)));
    return n; // TODO NORMAL MAPPING

#else

#ifdef NORMAL_LOC
    return frag_normal;
#else
    return normalize(cam_pos - frag_position);
#endif

#endif
}

// Fresnel
vec3 specular_reflection(PBRInfo info)
{
     vec3 res = info.f0 + (1.0 - info.f0) * pow(clamp(1.0 - info.vh, 0.0, 1.0), 5.0);
     return res;
}

// Smith
float geometric_occlusion(PBRInfo info)
{
    float r = info.roughness + 1.0;
    float k = r * r  / 8.0;
    float g1 = info.nv / (info.nv * (1.0 - k) + k);
    float g2 = info.nl / (info.nl * (1.0 - k) + k);
    //float k = info.roughness * sqrt(2.0 / PI);
    //float g1 = info.lh / (info.lh * (1.0 - k) + k);
    //float g2 = info.nh / (info.nh * (1.0 - k) + k);
    return g1 * g2;
}

float microfacet_distribution(PBRInfo info)
{
    float a = info.roughness * info.roughness;
    float a2 = a * a;
    float nh2 = info.nh * info.nh;

    float denom = (nh2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

vec3 compute_brdf(vec3 n, vec3 v, vec3 l,
                  float roughness, float metalness,
                  vec3 f0, vec3 c_diff, vec3 albedo,
                  vec3 radiance)
{
        vec3 h = normalize(l+v);
        float nl = clamp(dot(n, l), 0.001, 1.0);
        float nv = clamp(abs(dot(n, v)), 0.001, 1.0);
        float nh = clamp(dot(n, h), 0.0, 1.0);
        float lh = clamp(dot(l, h), 0.0, 1.0);
        float vh = clamp(dot(v, h), 0.0, 1.0);

        PBRInfo info = PBRInfo(nl, nv, nh, lh, vh, roughness, metalness, f0, c_diff);

        // Compute PBR terms
        vec3 F = specular_reflection(info);
        float G = geometric_occlusion(info);
        float D = microfacet_distribution(info);

        // Compute BRDF
        vec3 diffuse_contrib = (1.0 - F) * c_diff / PI;
        vec3 spec_contrib = F * G * D / (4.0 * nl * nv + 0.001);

        vec3 color = nl * radiance * (diffuse_contrib + spec_contrib);
        return color;
}

// Percentage-closer filtering over a 3x3 texel neighborhood. Each tap is one fetch: the sampler compares the reference
// depth against the four texels around the tap and blends the outcomes bilinearly (see the depth textures in
// texture.py), the result a fetch of each texel followed by the comparison and the blend in the shader would give.
float PCF(sampler2DShadow shadow_map, vec3 coords){
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
    float result = 0.0;
    for(int x=-1; x<=1; x++){
        for(int y=-1; y<=1; y++){
            result += texture(shadow_map, coords + vec3(vec2(x, y) * texel_size, 0.0));
        }
    }
    return result / 9.0;
}

float shadow_calc(mat4 light_matrix, sampler2DShadow shadow_map, float nl)
{
    // Compute light texture UV coords
    vec4 proj_coords = vec4(light_matrix * vec4(frag_position.xyz, 1.0));
    vec3 light_coords = proj_coords.xyz / proj_coords.w;
    light_coords = light_coords * 0.5 + 0.5;
    float current_depth = light_coords.z;
    float bias = max(0.001 * (1.0 - nl), 0.0001) / proj_coords.w;
    float compare = (current_depth - bias);
    float shadow = PCF(shadow_map, vec3(light_coords.xy, compare));
    //if (light_coords.z > 1.0) {
        //shadow = 0.0;
    //}
    return shadow;
}

vec3 point_PCF_samples[20] = vec3[]
(
   vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1), 
   vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
   vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
   vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
   vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
);   
int point_PCF_n = 20;

float point_PCF(samplerCube shadow_map, vec3 frag_to_light, float bias)
{
    float point_PCF_radius = 40.0 / textureSize(shadow_map, 0)[0];
    float dist = length(frag_to_light);
    float shadow = 0.0;
    for (int i = 0; i < point_PCF_n; i++) {
        float dis_shadow = texture(shadow_map, frag_to_light + point_PCF_samples[i] * point_PCF_radius).r * 25.0;
        shadow += (dis_shadow < 24.5 && dist > dis_shadow + bias) ? 1.0 : 0.0;
    }
    return shadow / point_PCF_n;
}

///////////////////////////////////////////////////////////////////////////////
// MAIN
///////////////////////////////////////////////////////////////////////////////
void main()
{
    vec3 reflected_position = (reflection_mat * vec4(frag_position, 1.0)).xyz;
    if (length(reflected_position - frag_position) > 1e-4
        && length(reflected_position - cam_pos) < length(frag_position - cam_pos)) {
        discard;
    }

    vec4 color = vec4(vec3(0.0), 1.0);
///////////////////////////////////////////////////////////////////////////////
// Handle Metallic Materials
///////////////////////////////////////////////////////////////////////////////
#ifdef USE_METALLIC_MATERIAL

    // Compute metallic/roughness factors
    float roughness = material.roughness_factor;
    float metallic = material.metallic_factor;
#ifdef HAS_METALLIC_ROUGHNESS_TEX
    vec2 mr = texture(material.metallic_roughness_texture, uv_0).rg;
    roughness = roughness * mr.r;
    metallic = metallic * mr.g;
#endif
    roughness = clamp(roughness, min_roughness, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    // In convention, material roughness is perceputal roughness ^ 2
    float alpha_roughness = roughness * roughness;

    // Compute albedo
    vec4 base_color = material.base_color_factor;
#ifdef HAS_BASE_COLOR_TEX
#ifdef BASE_COLOR_TEX_ARRAY
    vec4 base_color_texel = texture(material.base_color_texture, vec3(uv_0, material.base_color_layer));
#ifdef BASE_COLOR_TEX_LINEAR
    // A texture array holding linear values is encoded to sRGB, the encoding of every other base color
    base_color_texel = linear_to_srgb(base_color_texel);
#endif
    base_color = base_color * base_color_texel;
#else
    base_color = base_color * texture(material.base_color_texture, uv_0);
#endif
#endif

#ifdef COLOR_0_LOC
    base_color *= color_multiplier;
#endif

#ifdef INST_TINT_LOC
    // The tint sets the hue and the material keeps its brightness. This way the geoms and the textures of one instance
    // stay distinct. A color out of gamut moves toward the grey of the same luminance, so its hue stays the same.
    if (debug_view) {
        const vec3 luma = vec3(0.2126, 0.7152, 0.0722);
        vec3 tinted = tint.rgb * (2.0 * dot(base_color.rgb, luma));
        float lum_tinted = dot(tinted, luma);
        float peak = max(max(tinted.r, tinted.g), tinted.b);
        if (peak > 1.0) {
            tinted = mix(vec3(lum_tinted), tinted, clamp((1.0 - lum_tinted) / max(peak - lum_tinted, 1e-6), 0.0, 1.0));
        }
        base_color.rgb = mix(base_color.rgb, tinted, tint.a);
    }
#endif

    base_color = srgb_to_linear(base_color);

    // Compute specular and diffuse colors
    vec3 dialectric_spec = vec3(min_roughness);
    vec3 c_diff = mix(vec3(0.0), base_color.rgb * (1 - min_roughness), 1.0 - metallic);
    vec3 f0 = mix(dialectric_spec, base_color.rgb, metallic);

    // A transmissive dielectric diffuses no light. It reflects the share of the light that its index of refraction
    // gives, and lets the rest through tinted by its base color.
    float transmission = material.transmission_factor;
    if (transmission > 0.0) {
        float ior_ratio = (material.ior - 1.0) / (material.ior + 1.0);
        f0 = mix(vec3(ior_ratio * ior_ratio), base_color.rgb, metallic);
        c_diff *= 1.0 - transmission;
    }

    // Compute normal. The far side of a transmissive surface shows through its near side, and is shaded as seen from
    // the viewer.
    vec3 n = normalize(get_normal());
    vec3 v = normalize(cam_pos - frag_position); // Vector towards camera
    if (transmission > 0.0 && dot(n, v) < 0.0) {
        n = -n;
    }

    // Loop over lights
    for (int i = 0; i < n_directional_lights; i++) {
        vec3 direction = directional_lights[i].direction;
        vec3 l = normalize(-1.0 * direction);   // Vector towards light

        // Compute attenuation and radiance
        float attenuation = directional_lights[i].intensity;
        vec3 radiance = attenuation * directional_lights[i].color;

        // Compute outbound color
        vec3 res = compute_brdf(n, v, l, roughness, metallic,
                                f0, c_diff, base_color.rgb, radiance);

        // Compute shadow
#ifdef DIRECTIONAL_LIGHT_SHADOWS
        float nl = clamp(dot(n,l), 0.0, 1.0);
        float shadow = shadow_calc(
            directional_lights[i].light_matrix,
            directional_lights[i].shadow_map,
            nl
        );
        res = res * (1.0 - shadow);
#endif
        color.xyz += res;
    }

    for (int i = 0; i < n_point_lights; i++) {
        vec3 position = point_lights[i].position;
        vec3 l = normalize(position - frag_position); // Vector towards light

        // Compute attenuation and radiance
        float dist = length(position - frag_position);
        float attenuation = point_lights[i].intensity / (dist * dist);
        vec3 radiance = attenuation * point_lights[i].color;

        // Compute outbound color
        vec3 res = compute_brdf(n, v, l, roughness, metallic,
                                f0, c_diff, base_color.rgb, radiance);

#ifdef POINT_LIGHT_SHADOWS
        float bias = max(0.05 * (1.0 - dot(n, l)), 0.005);  
        float shadow = point_PCF(point_lights[i].shadow_map, frag_position - position, bias);
        res *= (1.0 - shadow);
#endif

        color.xyz += res;
    }
    for (int i = 0; i < n_spot_lights; i++) {
        vec3 position = spot_lights[i].position;
        vec3 l = normalize(position - frag_position); // Vector towards light

        // Compute attenuation and radiance
        vec3 direction = spot_lights[i].direction;
        float las = spot_lights[i].light_angle_scale;
        float lao = spot_lights[i].light_angle_offset;
        float dist = length(position - frag_position);
        float cd = clamp(dot(direction, -l), 0.0, 1.0);
        float attenuation = clamp(cd * las + lao, 0.0, 1.0);
        attenuation = attenuation * attenuation * spot_lights[i].intensity;
        attenuation = attenuation / (dist * dist);
        vec3 radiance = attenuation * spot_lights[i].color;

        // Compute outbound color
        vec3 res = compute_brdf(n, v, l, roughness, metallic,
                                f0, c_diff, base_color.rgb, radiance);
#ifdef SPOT_LIGHT_SHADOWS
        float nl = clamp(dot(n,l), 0.0, 1.0);
        float shadow = shadow_calc(
            spot_lights[i].light_matrix,
            spot_lights[i].shadow_map,
            nl
        );
        res = res * (1.0 - shadow);
#endif
        color.xyz += res;
    }
    color.xyz += (1.0 - transmission) * base_color.xyz * ambient_light;

    // The environment that a surface mirrors is a sky above the horizon, as bright as a surface of albedo 0.5 facing up
    // under the directional lights, and the ambient light below it
    vec3 sky = ambient_light;
    for (int i = 0; i < n_directional_lights; i++) {
        float elevation = max(-directional_lights[i].direction.z, 0.0);
        sky += 0.5 * directional_lights[i].intensity * directional_lights[i].color * elevation / PI;
    }
    vec3 env = mix(ambient_light, sky, smoothstep(-0.1, 0.1, reflect(-v, n).z));
    float nv_env = clamp(abs(dot(n, v)), 0.0, 1.0);

    // A metal diffuses no light, so away from the highlights of the lights it only shows the environment it mirrors. A
    // rough metal already spreads the light of the lamps over its whole surface, so the mirrored environment fades with
    // the roughness. The analytical fit of the environment BRDF integrates the specular lobe for the roughness and the
    // view angle.
    vec4 env_r = roughness * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
    float env_a = min(env_r.x * env_r.x, exp2(-9.28 * nv_env)) * env_r.x + env_r.y;
    vec2 env_ab = vec2(-1.04, 1.04) * env_a + env_r.zw;
    float smoothness = 1.0 - roughness;
    // A reflective floor mirrors the scene, rendered into 'floor_tex' through the floor, with the share of the light
    // that its surface reflects. A rough floor blurs the mirror image, sampled at a coarser level of detail. Away from
    // the mirrored geometry, which the alpha of the mirror image covers, the floor shows the environment as any
    // surface does. The program of the floor alone holds the lookup: a texture fetch in a program changes how some
    // software renderers compute its depth, which must match that of the other programs.
    float mirror_coverage = 0.0;
    vec3 mirror = vec3(0.0);
#ifdef REFLECTIVE_FLOOR
    vec4 mirror_texel = textureLod(floor_tex, gl_FragCoord.xy / screen_size, 4.0 * roughness * roughness);
    mirror_coverage = mirror_texel.a;
    mirror = srgb_to_linear(vec4(mirror_texel.rgb / max(mirror_coverage, 1e-4), 1.0)).rgb;
#endif
    vec3 env_reflected = (1.0 - mirror_coverage) * metallic * smoothness * smoothness * env + mirror_coverage * mirror;
    color.xyz += (f0 * env_ab.x + env_ab.y) * env_reflected;

    // A transmissive surface reflects more of the environment at grazing angles. The reflected share caps at the
    // smoothness of the surface, a rough surface scattering its reflection away.
    vec3 transmittance = vec3(0.0);
    if (transmission > 0.0) {
        vec3 fresnel = f0 + (max(vec3(smoothness), f0) - f0) * pow(1.0 - nv_env, 5.0);
        color.xyz += transmission * fresnel * env;
        transmittance = transmission * (1.0 - fresnel) * base_color.rgb;
    }

    // Calculate lighting from environment
#ifdef USE_IBL
    // TODO
#endif

    // Apply occlusion
#ifdef HAS_OCCLUSION_TEX
    float ao = texture(material.occlusion_texture, uv_0).r;
    color.xyz *= ao;
#endif

    // Apply emissive map. 'emissive' already folds in the factor, so add it directly rather than scaling by the
    // factor a second time.
    vec3 emissive = material.emissive_factor;
#ifdef HAS_EMISSIVE_TEX
#ifdef EMISSIVE_TEX_ARRAY
    // An emissive texture array holds linear values
    emissive *= texture(material.emissive_texture, vec3(uv_0, material.emissive_layer)).rgb;
#else
    emissive *= srgb_to_linear(texture(material.emissive_texture, uv_0)).rgb;
#endif
#endif
    color.xyz += emissive;

    frag_color = clamp(vec4(linear_to_srgb(color).rgb, color.a * base_color.a), 0.0, 1.0);
    // The blending runs on sRGB values, where scaling the background by the encoded transmittance approximates the
    // encoding of the attenuated background. The background shows unchanged through the share of the fragment that its
    // alpha leaves uncovered.
    float coverage = base_color.a;
    frag_transmittance = vec4(1.0 - coverage + coverage * linear_to_srgb(vec4(transmittance, 1.0)).rgb, 1.0);
    if (transmission > 0.0) {
        frag_color.rgb *= coverage;
    }


#else
    // TODO GLOSSY MATERIAL BRDF
#endif

///////////////////////////////////////////////////////////////////////////////
// Handle Glossy Materials
///////////////////////////////////////////////////////////////////////////////

}
