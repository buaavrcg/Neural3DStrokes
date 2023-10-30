#pragma once
#include <cstdint>
#include <cmath>
#include "common.h"
#include "helper_math.h"
#include "noise.h"
#include "sh.h"

enum ColorType
{
    CONSTANT_RGB = 0,
    GRADIENT_RGB = 1,
    NOISE_BRUSH_RGB = 2,
    CONSTANT_SH2 = 3,
    CONSTANT_SH3 = 4,
    NB_COLORS,
};

/////////////////////////////////////////////////////////////////////
// Color Fields
/////////////////////////////////////////////////////////////////////

template <ColorType color_type>
struct ColorField
{
    static constexpr int color_dim = 3;
    static constexpr bool use_unit_pos = false;
    static constexpr bool use_viewdir = false;
    // Stores color value into color_out for the given pos and shape params.
    __device__ static void get_color(float *color_out, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke);
    // Stores grad_params for given pos and color params.
    // Note: use atomic operation on grad_params
    __device__ static void grad_color(float *grad_params, const float *grad_color, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke);
};

template <>
struct ColorField<CONSTANT_RGB>
{
    static constexpr int color_dim = 3;
    static constexpr bool use_unit_pos = false;
    static constexpr bool use_viewdir = false;
    __device__ static void get_color(float *color_out, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke)
    {
        *(float3 *)color_out = *(float3 *)params;
    }
    __device__ static void grad_color(float *grad_params, const float *grad_color, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke)
    {
        atomicAdd3(grad_params, *(float3 *)grad_color);
    }
};

template <>
struct ColorField<GRADIENT_RGB>
{
    static constexpr int color_dim = 3;
    static constexpr bool use_unit_pos = true;
    static constexpr bool use_viewdir = false;
    __device__ static void get_color(float *color_out, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke)
    {
        float3 pos0 = ((float3 *)params)[0];
        float3 pos1 = ((float3 *)params)[1];
        float3 c0 = ((float3 *)params)[2];
        float3 c1 = ((float3 *)params)[3];
        float3 v0 = pos - pos0;
        float3 v1 = pos1 - pos0;
        float d0 = dot(v0, v1);
        float d1 = dot(v1, v1);
        float t_raw = d0 / d1;
        float t = sigmoid(t_raw * 5.0f - 2.5f);
        float3 c = c0 * (1.0f - t) + c1 * t;
        *(float3 *)color_out = c;
    }
    __device__ static void grad_color(float *grad_params, const float *grad_color, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke)
    {
        float3 pos0 = ((float3 *)params)[0];
        float3 pos1 = ((float3 *)params)[1];
        float3 c0 = ((float3 *)params)[2];
        float3 c1 = ((float3 *)params)[3];
        float3 v0 = pos - pos0;
        float3 v1 = pos1 - pos0;
        float d0 = dot(v0, v1);
        float d1 = dot(v1, v1);
        float t_raw = d0 / d1;
        float t = sigmoid(t_raw * 5.0f - 2.5f);

        float3 dL_dColor = *(float3 *)grad_color;
        float dL_dt = dot(dL_dColor, c1 - c0);
        float dL_dt_raw = 5.0f * t * (1.0f - t) * dL_dt;
        float dL_dd0 = dL_dt_raw / d1;
        float dL_dd1 = -dL_dt_raw * d0 / (d1 * d1);
        float3 dL_dv1 = dL_dd0 * v0 + dL_dd1 * 2.0f * v1;
        float3 dL_dv0 = dL_dd0 * v1;
        float3 dL_pos1 = dL_dv1;
        float3 dL_pos0 = -dL_dv0 - dL_dv1;

        atomicAdd3(grad_params + 0, dL_pos0);
        atomicAdd3(grad_params + 3, dL_pos1);
        atomicAdd3(grad_params + 6, dL_dColor * make_float3(1.0f - t));
        atomicAdd3(grad_params + 9, dL_dColor * make_float3(t));
    }
};

template <>
struct ColorField<NOISE_BRUSH_RGB>
{
    static constexpr int color_dim = 3;
    static constexpr bool use_unit_pos = true;
    static constexpr bool use_viewdir = false;
    __device__ static float tint(float3 pos, const uint32_t idx_stroke)
    {
        float3 dir = hash31((float)idx_stroke);
        // dir = dir * dir * dir; // pow(dir, 3.0f)
        // float3 dir_exp = make_float3(expf(dir.x), expf(dir.y), expf(dir.z));
        // float3 dir_softmax = dir_exp / (dir_exp.x + dir_exp.y + dir_exp.z);
        float p = dot(pos, dir) * 12.0f;
        float t = 1.0f + 0.4f * fbm11<4>(p);
        // pos *= 16.0f * dir_softmax;
        // float t = 1.0f + 0.3f * fbm13<4>(pos);
        return clamp(t, 0.f, 1.f);
    }
    __device__ static void get_color(float *color_out, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke)
    {
        float3 color = *(float3 *)params;
        *(float3 *)color_out = color * tint(pos, idx_stroke);
    }
    __device__ static void grad_color(float *grad_params, const float *grad_color, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke)
    {
        float3 dL_dColor = *(float3 *)grad_color;
        atomicAdd3(grad_params + 0, dL_dColor * tint(pos, idx_stroke));
    }
};

template <>
struct ColorField<CONSTANT_SH2>
{
    static constexpr int color_dim = 3;
    static constexpr bool use_unit_pos = false;
    static constexpr bool use_viewdir = true;
    __device__ static void get_color(float *color_out, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke)
    {
        float weights[4];
        sh_enc(2, viewdir, weights);

        float3 rgb = make_float3(0.0f);
#pragma unroll
        for (int i = 0; i < 4; i++)
            rgb += ((float3 *)params)[i] * weights[i];

        *(float3 *)color_out = make_float3(sigmoid(rgb.x), sigmoid(rgb.y), sigmoid(rgb.z));
    }
    __device__ static void grad_color(float *grad_params, const float *grad_color, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke)
    {
        float weights[4];
        sh_enc(2, viewdir, weights);

        float3 rgb = make_float3(0.0f);
#pragma unroll
        for (int i = 0; i < 4; i++)
            rgb += ((float3 *)params)[i] * weights[i];
        rgb = make_float3(sigmoid(rgb.x), sigmoid(rgb.y), sigmoid(rgb.z));
        float3 grad_rgb = *(float3 *)grad_color * rgb * (1.0f - rgb);

#pragma unroll
        for (int i = 0; i < 4; i++)
            atomicAdd3(grad_params + i * 3, grad_rgb * weights[i]);
    }
};

template <>
struct ColorField<CONSTANT_SH3>
{
    static constexpr int color_dim = 3;
    static constexpr bool use_unit_pos = false;
    static constexpr bool use_viewdir = true;
    __device__ static void get_color(float *color_out, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke)
    {
        float weights[9];
        sh_enc(3, viewdir, weights);

        float3 rgb = make_float3(0.0f);
#pragma unroll
        for (int i = 0; i < 9; i++)
            rgb += ((float3 *)params)[i] * weights[i];

        *(float3 *)color_out = make_float3(sigmoid(rgb.x), sigmoid(rgb.y), sigmoid(rgb.z));
    }
    __device__ static void grad_color(float *grad_params, const float *grad_color, float3 pos, float3 viewdir, const float *params, const uint32_t idx_stroke)
    {
        float weights[9];
        sh_enc(3, viewdir, weights);

        float3 rgb = make_float3(0.0f);
#pragma unroll
        for (int i = 0; i < 9; i++)
            rgb += ((float3 *)params)[i] * weights[i];
        rgb = make_float3(sigmoid(rgb.x), sigmoid(rgb.y), sigmoid(rgb.z));
        float3 grad_rgb = *(float3 *)grad_color * rgb * (1.0f - rgb);

#pragma unroll
        for (int i = 0; i < 9; i++)
            atomicAdd3(grad_params + i * 3, grad_rgb * weights[i]);
    }
};
