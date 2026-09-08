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
    sampler2D emissive_texture;
#endif
#ifdef HAS_BASE_COLOR_TEX
    sampler2D base_color_texture;
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
uniform sampler2D floor_tex;
uniform int floor_flag;
uniform vec2 screen_size;

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

///////////////////////////////////////////////////////////////////////////////
// OUTPUTS
///////////////////////////////////////////////////////////////////////////////

out vec4 frag_color;

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
    if(length(reflected_position - frag_position) > 1e-4 && length(reflected_position-cam_pos)<length(frag_position-cam_pos)) discard;
    
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
    base_color = base_color * texture(material.base_color_texture, uv_0);
#endif

#ifdef COLOR_0_LOC
    base_color *= color_multiplier;
#endif

    base_color = srgb_to_linear(base_color);

    // Compute specular and diffuse colors
    vec3 dialectric_spec = vec3(min_roughness);
    vec3 c_diff = mix(vec3(0.0), base_color.rgb * (1 - min_roughness), 1.0 - metallic);
    vec3 f0 = mix(dialectric_spec, base_color.rgb, metallic);

    // Compute normal
    vec3 n = normalize(get_normal());

    // Loop over lights
    for (int i = 0; i < n_directional_lights; i++) {
        vec3 direction = directional_lights[i].direction;
        vec3 v = normalize(cam_pos - frag_position); // Vector towards camera
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
        vec3 v = normalize(cam_pos - frag_position); // Vector towards camera
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
        vec3 v = normalize(cam_pos - frag_position); // Vector towards camera
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
    color.xyz += base_color.xyz * ambient_light;

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
    emissive *= srgb_to_linear(texture(material.emissive_texture, uv_0)).rgb;
#endif
    color.xyz += emissive;

    vec3 floor_color = floor_flag != 0 ? texture(floor_tex, gl_FragCoord.xy/screen_size).rgb : vec3(0.0);

    if (floor_flag != 0 && length(floor_color) > 1e-4) {
        //color.xyz *= 0;
        vec3 floor_radiance = pow(floor_color, vec3(2.2));
        vec3 v = normalize(cam_pos - frag_position); // Vector towards camera
        vec3 l = n * dot(n, v) * 2 - v;

        vec3 h = normalize(l+v);
        float vh = clamp(dot(v, h), 0.0, 1.0);
        f0 = vec3(0.03);
        vec3 reflectance = (f0 + (1.0 - f0) * pow(clamp(1.0 - vh, 0.0, 1.0), 5.0)) * 1.0;
        color.xyz = color.xyz * (1-reflectance) + reflectance * floor_radiance;
        // color.xyz += compute_brdf(n, v, l, roughness, metallic, f0, c_diff, vec3(0.0), 1*floor_radiance);
    }
    // if (floor_flag != 0) color.xyz = floor_color.xyz;

    frag_color = clamp(vec4(pow(color.xyz, vec3(1.0/2.2)), color.a * base_color.a), 0.0, 1.0);


#else
    // TODO GLOSSY MATERIAL BRDF
#endif

///////////////////////////////////////////////////////////////////////////////
// Handle Glossy Materials
///////////////////////////////////////////////////////////////////////////////

}
