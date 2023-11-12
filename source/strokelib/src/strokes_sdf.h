#pragma once
#include <cstdint>
#include <cmath>
#include "common.h"
#include "helper_math.h"
#include "noise.h"

enum BaseSDFType
{
    UNIT_SPHERE = 0,
    UNIT_CUBE = 1,
    UNIT_ROUND_CUBE = 2,
    UNIT_CAPPED_TORUS = 3,
    UNIT_CAPSULE = 4,
    UNIT_LINE = 5,
    UNIT_TRIPRISM = 6,
    UNIT_OCTAHEDRON = 7,
    UNIT_TETRAHEDRON = 8,
    QUADRATIC_BEZIER = 9,
    CUBIC_BEZIER = 10,
    CATMULL_ROM = 11,
    NB_BASE_SDFS,
};

template <bool Inverse = true>
__device__ inline float3 rotate_point(float3 point, float3 angle)
{
    float3 s = sin(angle);
    float3 c = cos(angle);

    float r00 = c.y * c.z;
    float r01 = s.x * s.y * c.z - c.x * s.z;
    float r02 = s.x * s.z + c.x * s.y * c.z;
    float r10 = c.y * s.z;
    float r11 = c.x * c.z + s.x * s.y * s.z;
    float r12 = c.x * s.y * s.z - s.x * c.z;
    float r20 = -s.y;
    float r21 = s.x * c.y;
    float r22 = c.x * c.y;

    float px, py, pz;
    if constexpr (Inverse)
    {
        px = r00 * point.x + r10 * point.y + r20 * point.z;
        py = r01 * point.x + r11 * point.y + r21 * point.z;
        pz = r02 * point.x + r12 * point.y + r22 * point.z;
    }
    else
    {
        px = r00 * point.x + r01 * point.y + r02 * point.z;
        py = r10 * point.x + r11 * point.y + r12 * point.z;
        pz = r20 * point.x + r21 * point.y + r22 * point.z;
    }

    return make_float3(px, py, pz);
}

// Returns dL/dPoint from dL/dRotatedPoint
template <bool Inverse = true>
__device__ inline float3 grad_point_rotate_point(float3 dL_dRotatedPoint, float3 point, float3 angle)
{
    float3 s = sin(angle);
    float3 c = cos(angle);

    float r00 = c.y * c.z;
    float r01 = s.x * s.y * c.z - c.x * s.z;
    float r02 = s.x * s.z + c.x * s.y * c.z;
    float r10 = c.y * s.z;
    float r11 = c.x * c.z + s.x * s.y * s.z;
    float r12 = c.x * s.y * s.z - s.x * c.z;
    float r20 = -s.y;
    float r21 = s.x * c.y;
    float r22 = c.x * c.y;

    float3 dL_dPoint = make_float3(0.0f, 0.0f, 0.0f);
    if constexpr (Inverse)
    {
        dL_dPoint.x = r00 * dL_dRotatedPoint.x + r01 * dL_dRotatedPoint.y + r02 * dL_dRotatedPoint.z;
        dL_dPoint.y = r10 * dL_dRotatedPoint.x + r11 * dL_dRotatedPoint.y + r12 * dL_dRotatedPoint.z;
        dL_dPoint.z = r20 * dL_dRotatedPoint.x + r21 * dL_dRotatedPoint.y + r22 * dL_dRotatedPoint.z;
    }
    else
    {
        dL_dPoint.x = r00 * dL_dRotatedPoint.x + r10 * dL_dRotatedPoint.y + r20 * dL_dRotatedPoint.z;
        dL_dPoint.y = r01 * dL_dRotatedPoint.x + r11 * dL_dRotatedPoint.y + r21 * dL_dRotatedPoint.z;
        dL_dPoint.z = r02 * dL_dRotatedPoint.x + r12 * dL_dRotatedPoint.y + r22 * dL_dRotatedPoint.z;
    }

    return dL_dPoint;
}

// Returns dL/dAngle from dL/dRotatedPoint
template <bool Inverse = true>
__device__ inline float3 grad_angle_rotate_point(float3 dL_dRotatedPoint, float3 point, float3 angle)
{
    float3 s = sin(angle);
    float3 c = cos(angle);

    float dr00_dax = 0.0f;
    float dr00_day = -s.y * c.z;
    float dr00_daz = -c.y * s.z;

    float dr01_dax = c.x * s.y * c.z + s.x * s.z;
    float dr01_day = s.x * c.y * c.z;
    float dr01_daz = -s.x * s.y * s.z - c.x * c.z;

    float dr02_dax = c.x * s.z - s.x * s.y * c.z;
    float dr02_day = c.x * c.y * c.z;
    float dr02_daz = s.x * c.z - c.x * s.y * s.z;

    float dr10_dax = 0.0f;
    float dr10_day = -s.y * s.z;
    float dr10_daz = c.y * c.z;

    float dr11_dax = -s.x * c.z + c.x * s.y * s.z;
    float dr11_day = s.x * c.y * s.z;
    float dr11_daz = -c.x * s.z + s.x * s.y * c.z;

    float dr12_dax = -s.x * s.y * s.z - c.x * c.z;
    float dr12_day = c.x * c.y * s.z;
    float dr12_daz = c.x * s.y * c.z + s.x * s.z;

    float dr20_dax = 0.0f;
    float dr20_day = -c.y;
    float dr20_daz = 0.0f;

    float dr21_dax = c.x * c.y;
    float dr21_day = -s.x * s.y;
    float dr21_daz = 0.0f;

    float dr22_dax = -s.x * c.y;
    float dr22_day = -c.x * s.y;
    float dr22_daz = 0.0f;

    float3 dL_dAngle = make_float3(0.0f, 0.0f, 0.0f);
    if constexpr (Inverse)
    {
        dL_dAngle += dL_dRotatedPoint.x * point.x * make_float3(dr00_dax, dr00_day, dr00_daz);
        dL_dAngle += dL_dRotatedPoint.x * point.y * make_float3(dr10_dax, dr10_day, dr10_daz);
        dL_dAngle += dL_dRotatedPoint.x * point.z * make_float3(dr20_dax, dr20_day, dr20_daz);
        dL_dAngle += dL_dRotatedPoint.y * point.x * make_float3(dr01_dax, dr01_day, dr01_daz);
        dL_dAngle += dL_dRotatedPoint.y * point.y * make_float3(dr11_dax, dr11_day, dr11_daz);
        dL_dAngle += dL_dRotatedPoint.y * point.z * make_float3(dr21_dax, dr21_day, dr21_daz);
        dL_dAngle += dL_dRotatedPoint.z * point.x * make_float3(dr02_dax, dr02_day, dr02_daz);
        dL_dAngle += dL_dRotatedPoint.z * point.y * make_float3(dr12_dax, dr12_day, dr12_daz);
        dL_dAngle += dL_dRotatedPoint.z * point.z * make_float3(dr22_dax, dr22_day, dr22_daz);
    }
    else
    {
        dL_dAngle += dL_dRotatedPoint.x * point.x * make_float3(dr00_dax, dr00_day, dr00_daz);
        dL_dAngle += dL_dRotatedPoint.x * point.y * make_float3(dr01_dax, dr01_day, dr01_daz);
        dL_dAngle += dL_dRotatedPoint.x * point.z * make_float3(dr02_dax, dr02_day, dr02_daz);
        dL_dAngle += dL_dRotatedPoint.y * point.x * make_float3(dr10_dax, dr10_day, dr10_daz);
        dL_dAngle += dL_dRotatedPoint.y * point.y * make_float3(dr11_dax, dr11_day, dr11_daz);
        dL_dAngle += dL_dRotatedPoint.y * point.z * make_float3(dr12_dax, dr12_day, dr12_daz);
        dL_dAngle += dL_dRotatedPoint.z * point.x * make_float3(dr20_dax, dr20_day, dr20_daz);
        dL_dAngle += dL_dRotatedPoint.z * point.y * make_float3(dr21_dax, dr21_day, dr21_daz);
        dL_dAngle += dL_dRotatedPoint.z * point.z * make_float3(dr22_dax, dr22_day, dr22_daz);
    }

    return dL_dAngle;
}

template <bool enable_translation,
          bool enable_rotation,
          bool enable_singlescale,
          bool enable_multiscale>
__device__ inline float3 inverse_transform(float3 point, const float *&sp_reverse)
{
    if constexpr (enable_translation)
    {
        float3 translation = *(const float3 *)(sp_reverse -= 3);
        point -= translation;
    }
    if constexpr (enable_rotation)
    {
        float3 eular_angle = *(const float3 *)(sp_reverse -= 3);
        point = rotate_point(point, eular_angle);
    }
    if constexpr (enable_singlescale)
    {
        float scale = *(sp_reverse -= 1);
        point /= scale;
    }
    else if constexpr (enable_multiscale)
    {
        float3 scale = *(const float3 *)(sp_reverse -= 3);
        point /= scale;
    }
    return point;
}

template <bool enable_rotation,
          bool enable_multiscale>
__device__ inline float3 inverse_transform_direction(float3 dir, const float *&sp_reverse)
{
    if constexpr (enable_rotation)
    {
        float3 eular_angle = *(const float3 *)(sp_reverse -= 3);
        dir = rotate_point(dir, eular_angle);
    }
    if constexpr (enable_multiscale)
    {
        float3 scale = *(const float3 *)(sp_reverse -= 3);
        dir /= scale;
        dir = normalize(dir);
    }
    return dir;
}

template <int num_base_params>
__device__ inline float3 permute_multiscale_axis(float3 d, const float *shape_params)
{
    float3 scale = *(const float3 *)(shape_params + num_base_params);
    return (scale.x > scale.y ?
                scale.y > scale.z ? make_float3(d.x, d.y, d.z)
                                  : scale.x > scale.z ? make_float3(d.x, d.z, d.y)
                                                      : make_float3(d.z, d.x, d.y)
              : scale.z > scale.y ? make_float3(d.z, d.y, d.x)
                                  : scale.x > scale.z ? make_float3(d.y, d.x, d.z)
                                                      : make_float3(d.y, d.z, d.x));
}

/////////////////////////////////////////////////////////////////////
// Base Signed Distance Fields - Primitives
/////////////////////////////////////////////////////////////////////

template <BaseSDFType sdf_type>
struct BaseSDF
{
    // Returns SDF value for the given pos and shape params.
    __device__ static float sdf(float3 pos, const float *params);
    // Returns dSDF/dPos and stores grad_params for given pos and shape params.
    // Note: use atomic operation on grad_params, and
    //       if grad_params is nullptr, then do not compute gradients w.r.t. params.
    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params);
    // Returns the surface texture coordinate (u,v) for the given pos and shape params.
    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params);
};

template <>
struct BaseSDF<UNIT_SPHERE>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 p_sq = pos * pos;
        return sqrt(p_sq.x + p_sq.y + p_sq.z + 1e-8f) - 1.f;
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float3 p_sq = pos * pos;
        float inv_norm = rsqrt(p_sq.x + p_sq.y + p_sq.z + 1e-8f);
        return pos * inv_norm;
    }

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        float3 d = normalize(pos);
        if constexpr (enable_multiscale)
            d = permute_multiscale_axis<0>(d, params);
        float u = asinf(clamp(d.x, -1.0f, 1.0f)) / PI + 0.5f;
        float v = atan2f(d.z, d.y) / (2.0f * PI) + 0.5f;
        return make_float2(u, v);
    }
};

template <>
struct BaseSDF<UNIT_CUBE>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 q = fabs(pos) - 1.f;
        return length(fmaxf(q, make_float3(0.0f))) + min(max(q.x, max(q.y, q.z)), 0.0f);
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float3 q = fabs(pos) - 1.f;

        float grad_x, grad_y, grad_z;
        if (max(q.x, max(q.y, q.z)) <= 0.0f) {  // inside the cube
            grad_x = q.x >= q.y && q.x >= q.z ? (pos.x < 0.0f ? -1.0f : 1.0f) : 0.0f;
            grad_y = q.x < q.y && q.y >= q.z ? (pos.y < 0.0f ? -1.0f : 1.0f) : 0.0f;
            grad_z = q.x < q.z && q.y < q.z ? (pos.z < 0.0f ? -1.0f : 1.0f) : 0.0f;
        } else {
            float3 grad_length = 2.0f * fmaxf(q, make_float3(0.0f));
            float3 grad_q = make_float3(q.x > 0.0f ? grad_length.x : 0.0f,
                                        q.y > 0.0f ? grad_length.y : 0.0f,
                                        q.z > 0.0f ? grad_length.z : 0.0f);
            grad_x = pos.x < 0.0f ? -grad_q.x : grad_q.x;
            grad_y = pos.y < 0.0f ? -grad_q.y : grad_q.y;
            grad_z = pos.z < 0.0f ? -grad_q.z : grad_q.z;
        }

        return make_float3(grad_x, grad_y, grad_z);
    }

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        float3 uvt = pos * 0.5f + 0.5f;
        if constexpr (enable_multiscale)
            uvt = permute_multiscale_axis<0>(uvt, params);
        float u = clamp(uvt.x, 0.0f, 1.0f);
        float v = clamp(uvt.y, 0.0f, 1.0f);
        return make_float2(u, v);
    }
};

template <>
struct BaseSDF<UNIT_ROUND_CUBE>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 p_abs = fabs(pos);
        float3 p_dis = p_abs - make_float3(1.0f, 1.0f, 1.0f);
        float3 p_dis_positive = fmaxf(p_dis, make_float3(0.0f, 0.0f, 0.0f));
        float3 p_dis_square = p_dis_positive * p_dis_positive;
        float p_dis_norm = sqrt(p_dis_square.x + p_dis_square.y + p_dis_square.z + 1e-8f);
        float p_dis_min = min(max(p_dis.x, max(p_dis.y, p_dis.z)), 0.0f);
        return p_dis_norm + p_dis_min - params[0];
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        if (grad_params)
            atomicAdd(grad_params + 0, -grad_SDF * 1.0f);

        float3 p_abs = fabs(pos);
        float3 p_dis = p_abs - make_float3(1.0f, 1.0f, 1.0f);
        float3 p_dis_positive = fmaxf(p_dis, make_float3(0.0f, 0.0f, 0.0f));
        float3 p_dis_square = p_dis_positive * p_dis_positive;
        float p_dis_norm = sqrt(p_dis_square.x + p_dis_square.y + p_dis_square.z + 1e-8f);
        float p_dis_norm_reciprocal = rsqrt(p_dis_square.x + p_dis_square.y + p_dis_square.z + 1e-8);
        float3 grad_p_dis_positive = p_dis_norm > 0.0f ? p_dis_positive * p_dis_norm_reciprocal : make_float3(0.0f, 0.0f, 0.0f);
        float grad_p_x_sym = (pos.x < 0.0f ? -1.0f : 1.0f);
        float grad_p_y_sym = (pos.y < 0.0f ? -1.0f : 1.0f);
        float grad_p_z_sym = (pos.z < 0.0f ? -1.0f : 1.0f);
        float3 grad_p_dis_sym = make_float3(grad_p_x_sym, grad_p_y_sym, grad_p_z_sym);
        float3 grad_p_dis = grad_p_dis_positive * grad_p_dis_sym;

        float p_dis_min = min(max(p_dis.x, max(p_dis.y, p_dis.z)), 0.0f);
        float grad_p_dis_min = p_dis_min < 0.0f ? 1.0f : 0.0f;
        float grad_p_dis_min_x = grad_p_dis_min * (p_dis.x >= p_dis.y && p_dis.x >= p_dis.z ? 1.0f : 0.0f) * grad_p_dis_sym.x;
        float grad_p_dis_min_y = grad_p_dis_min * (p_dis.x < p_dis.y && p_dis.y >= p_dis.z ? 1.0f : 0.0f) * grad_p_dis_sym.y;
        float grad_p_dis_min_z = grad_p_dis_min * (p_dis.x < p_dis.z && p_dis.y < p_dis.z ? 1.0f : 0.0f) * grad_p_dis_sym.z;

        float grad_x = grad_p_dis.x + grad_p_dis_min_x;
        float grad_y = grad_p_dis.y + grad_p_dis_min_y;
        float grad_z = grad_p_dis.z + grad_p_dis_min_z;

        return make_float3(grad_x, grad_y, grad_z);
    }

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        float3 uvt = pos * 0.5f + 0.5f;
        if constexpr (enable_multiscale)
            uvt = permute_multiscale_axis<0>(uvt, params);
        float u = clamp(uvt.x, 0.0f, 1.0f);
        float v = clamp(uvt.y, 0.0f, 1.0f);
        return make_float2(u, v);
    }
};

template <>
struct BaseSDF<UNIT_CAPPED_TORUS>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 p_abs = make_float3(fabs(pos.x), pos.y, pos.z);
        float2 p_xy = make_float2(p_abs.x, p_abs.y);
        float2 sc = make_float2(sinf(params[0]), cosf(params[0]));
        float k = cross(make_float3(p_xy, 0.0f), make_float3(sc, 0.0f)).z > 0 ? dot(p_xy, sc) : sqrt(p_xy.x * p_xy.x + p_xy.y * p_xy.y + 1e-8f);
        return sqrtf(dot(p_abs, p_abs) + 1.f - 2.f * k + 1e-8f) - params[1];
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float3 p_abs = make_float3(fabs(pos.x), pos.y, pos.z);
        float2 p_xy = make_float2(p_abs.x, p_abs.y);
        float2 sc = make_float2(sinf(params[0]), cosf(params[0]));
        float2 sc_f = make_float2(cosf(params[0]), sinf(params[0]));
        float k = cross(make_float3(p_xy, 0.0f), make_float3(sc, 0.0f)).z > 0 ? dot(p_xy, sc) : length(p_xy);
        float inv_norm = rsqrt(dot(p_abs, p_abs) + 1.f - 2.f * k + 1e-8f);

        float3 grad_p_dot = pos;
        float grad_x = 0.0f;
        float grad_y = 0.0f;
        float grad_z = 0.0f;

        if (k > 0.f)
        {
            float grad_p_k_x = -(pos.x < 0.0f ? -1.0f : 1.0f) * sc.x;
            float grad_p_k_y = -sc.y;
            float grad_p_k_z = 0.0f;
            float3 grad_p_k = make_float3(grad_p_k_x, grad_p_k_y, grad_p_k_z);
            if (grad_params)
                atomicAdd(grad_params + 0, -grad_SDF * dot(p_xy, sc_f));
            grad_x = (grad_p_dot.x + grad_p_k.x) * inv_norm;
            grad_y = (grad_p_dot.y + grad_p_k.y) * inv_norm;
            grad_z = (grad_p_dot.z + grad_p_k.z) * inv_norm;
        }
        else
        {
            float grad_p_k_x = pos.x;
            float grad_p_k_y = pos.y;
            float grad_p_k_z = 0.0f;
            float3 grad_p_k = make_float3(grad_p_k_x, grad_p_k_y, grad_p_k_z);
            grad_x = (grad_p_dot.x + grad_p_k.x) * inv_norm;
            grad_y = (grad_p_dot.y + grad_p_k.y) * inv_norm;
            grad_z = (grad_p_dot.z + grad_p_k.z) * inv_norm;
        }

        if (grad_params)
            atomicAdd(grad_params + 1, -grad_SDF);
        return make_float3(grad_x, grad_y, grad_z);
    }

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        return make_float2(0.0f, 0.0f);
    }
};

template <>
struct BaseSDF<UNIT_CAPSULE>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float h = params[0];
        pos.y -= clamp(pos.y, -h, h);
        float3 p_sq = pos * pos;
        return sqrt(p_sq.x + p_sq.y + p_sq.z + 1e-8) - 1.f;
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float h = params[0];
        float pos_y = pos.y;
        pos.y -= clamp(pos.y, -h, h);
        float3 p_sq = pos * pos;
        float3 grad_pos = pos * rsqrt(p_sq.x + p_sq.y + p_sq.z + 1e-8);
        if (-h <= pos_y && pos_y <= h)
            grad_pos.y = 0.0f;
        else if (grad_params)
            atomicAdd(grad_params + 0, grad_SDF * grad_pos.y * (pos_y > 0.f ? -1.f : 1.f));
        return grad_pos;
    }

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        float h = params[0];
        float u = clamp((pos.y / (h + 1.0f)) * 0.5f + 0.5f, 0.0f, 1.0f);
        if constexpr (enable_multiscale) {
            float3 scale = *(const float3 *)(params + 1);
            if (scale.z > scale.x)  // swap x and z if scale_z > scale_x
                pos = make_float3(pos.z, pos.y, pos.x);
        }
        float v = atan2f(pos.z, pos.x) / (2.0f * PI) + 0.5f;
        return make_float2(u, v);
    }
};

template <>
struct BaseSDF<UNIT_LINE>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float h = params[0];
        float r_diff = params[1];
        float t = clamp((pos.y + h) / (2.0f * h), 0.0f, 1.0f);
        float r = lerp(1.f - r_diff, 1.f + r_diff, t);
        pos.y -= clamp(pos.y, -h, h);
        float3 p_sq = pos * pos;
        return sqrt(p_sq.x + p_sq.y + p_sq.z + 1e-8) - r;
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float h = params[0];
        float r_diff = params[1];
        float pos_y = pos.y;
        float t = clamp((pos.y + h) / (2.0f * h), 0.0f, 1.0f);
        float r = lerp(1.f - r_diff, 1.f + r_diff, t);
        pos.y -= clamp(pos.y, -h, h);
        float3 p_sq = pos * pos;
        float3 grad_pos = pos * rsqrt(p_sq.x + p_sq.y + p_sq.z + 1e-8);
        float grad_param0 = 0.0f;
        if (-h <= pos_y && pos_y <= h)
        {
            grad_pos.y = -r_diff / h;
            grad_param0 = pos_y * r_diff / (h * h);
        }
        else
        {
            grad_param0 = grad_pos.y * (pos_y > 0.f ? -1.f : 1.f);
        }
        if (grad_params) {
            atomicAdd(grad_params + 0, grad_SDF * grad_param0);
            atomicAdd(grad_params + 1, grad_SDF * (1.f - 2.f * t));
        }
        return grad_pos;
    }

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        float h = params[0];
        float r_diff = params[1];
        float u = clamp((pos.y + h + 1.0f - r_diff) / (2.0f * h + 2.0f), 0.0f, 1.0f);
        if constexpr (enable_multiscale) {
            float3 scale = *(const float3 *)(params + 1);
            if (scale.z > scale.x)  // swap x and z if scale_z > scale_x
                pos = make_float3(pos.z, pos.y, pos.x);
        }
        float v = atan2f(pos.z, pos.x) / (2.0f * PI) + 0.5f;
        return make_float2(u, v);
    }
};

template <>
struct BaseSDF<UNIT_TRIPRISM>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 q = fabs(pos);
        return fmaxf(q.y - params[0], fmaxf(q.x * 0.866025f + pos.z * 0.5f, -pos.z) - 0.5f);
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float3 q = fabs(pos);
        float3 grad_q = make_float3(pos.x < 0.0f ? -1.0f : 1.0f, pos.y < 0.0f ? -1.0f : 1.0f, pos.z < 0.0f ? -1.0f : 1.0f);
        float grad_x = 0.0f;
        float grad_y = 0.0f;
        float grad_z = 0.0f;
        if (q.x * 0.866025f + pos.z * 0.5f > -pos.z)
        {
            if (q.y - params[0] > q.x * 0.866025f + pos.z * 0.5f - 0.5f)
            {
                grad_y = grad_q.y;
                if (grad_params)
                    atomicAdd(grad_params + 0, -grad_SDF);
            }
            else
            {
                grad_x = 0.866025f * grad_q.x;
                grad_z = 0.5f;
            }
        }
        else
        {
            if (q.y - params[0] > -pos.z - 0.5f)
            {
                grad_y = grad_q.y;
                if (grad_params)
                    atomicAdd(grad_params + 0, -grad_SDF);
            }
            else
            {
                grad_z = -1.0f;
            }
        }
        return make_float3(grad_x, grad_y, grad_z);
    }

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        return make_float2(0.0f, 0.0f);
    }
};

template <>
struct BaseSDF<UNIT_OCTAHEDRON>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 p_abs = fabs(pos);
        return (p_abs.x + p_abs.y + p_abs.z - 1.0f) * 0.57735027f;
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float3 p_abs = fabs(pos);
        float grad_x = pos.x < 0.0f ? -0.57735027f : 0.57735027f;
        float grad_y = pos.y < 0.0f ? -0.57735027f : 0.57735027f;
        float grad_z = pos.z < 0.0f ? -0.57735027f : 0.57735027f;
        return make_float3(grad_x, grad_y, grad_z);
    }

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        return make_float2(0.0f, 0.0f);
    }
};


template <>
struct BaseSDF<UNIT_TETRAHEDRON>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        return (max(fabsf(pos.x + pos.y) - pos.z, fabsf(pos.x - pos.y) + pos.z) - 1.0f) * 0.57735027f;
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float a = pos.x + pos.y;
        float c = pos.x - pos.y;

        float grad_x, grad_y, grad_z;
        if (fabsf(a) - pos.z > fabsf(c) + pos.z) {
            grad_x = a < 0.0f ? -0.57735027f : 0.57735027f;
            grad_y = a < 0.0f ? -0.57735027f : 0.57735027f;
            grad_z = -0.57735027f;
        } else {
            grad_x = c < 0.0f ? -0.57735027f : 0.57735027f;
            grad_y = c < 0.0f ? 0.57735027f : -0.57735027f;
            grad_z = 0.57735027f;
        }

        return make_float3(grad_x, grad_y, grad_z);
    }

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        return make_float2(0.0f, 0.0f);
    }
};


/////////////////////////////////////////////////////////////////////
// Base Signed Distance Fields - Curves
/////////////////////////////////////////////////////////////////////

template <>
struct BaseSDF<QUADRATIC_BEZIER>
{
    __device__ static float sdf_quad_bezier(float3 pos, float3 A, float3 B, float3 C, float r1, float r2)
    {
        float2 res = make_float2(0.0f, 0.0f);

        float3 a = B - A;
        float3 b = A - 2.0f * B + C;
        float3 c = a * 2.0f;
        float3 d = A - pos;

        float kk = 1.0f / dot(b, b);
        float kx = kk * dot(a, b);
        float ky = kk * (2.0f * dot(a, a) + dot(d, b)) / 3.0f;
        float kz = kk * dot(d, a);

        float p = ky - kx * kx;
        float p3 = p * p * p;
        float q = kx * (2.0f * kx * kx - 3.0f * ky) + kz;
        float h = q * q + 4.0f * p3;

        if (h >= 0.0f)
        {
            h = sqrtf(h + 1e-8f);
            float2 x = make_float2((h - q) * 0.5f, -(h + q) * 0.5f);
            float2 uv = make_float2(
                copysignf(powf(fabs(x.x), 1.0f / 3.0f), x.x),
                copysignf(powf(fabs(x.y), 1.0f / 3.0f), x.y));
            float t = clamp((uv.x + uv.y) - kx, 0.0, 1.0);

            res = make_float2(dot(d + (c + b * t) * t, d + (c + b * t) * t), t);
        }
        else
        {
            float z = sqrtf(-p + 1e-8f);
            float v = acosf(q / (p * z * 2.0f + 1e-8f)) / 3.0f;
            float m = cosf(v);
            float n = sinf(v) * 1.732050808f;
            float3 t = clamp(make_float3(m + n, -n - m, n - m) * z - kx, 0.0f, 1.0f);

            float3 dis3 = d + (c + b * t.x) * t.x;
            float dis = dot(dis3, dis3);
            res = make_float2(dis, t.x);

            dis3 = d + (c + b * t.y) * t.y;
            dis = dot(dis3, dis3);
            res = dis < res.x ? make_float2(dis, t.y) : res;
        }

        res.x = sqrtf(res.x + 1e-8f);
        float r = lerp(r1, r2, res.y);
        return res.x - r;
    }

    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 A = make_float3(params[0], params[1], params[2]);
        float3 B = make_float3(params[3], params[4], params[5]);
        float3 C = make_float3(params[6], params[7], params[8]);
        return sdf_quad_bezier(pos, A, B, C, params[9], params[10]);
    }

#if 1  // manual gradient
    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float3 A = make_float3(params[0], params[1], params[2]);
        float3 B = make_float3(params[3], params[4], params[5]);
        float3 C = make_float3(params[6], params[7], params[8]);
        float2 res = make_float2(0.0f, 0.0f);

        float3 a = B - A;
        float3 b = A - 2.0f * B + C;
        float3 c = a * 2.0f;
        float3 d = A - pos;

        float kk = 1.0f / dot(b, b);
        float kx = kk * dot(a, b);
        float ky = kk * (2.0f * dot(a, a) + dot(d, b)) / 3.0f;
        float kz = kk * dot(d, a);

        float p = ky - kx * kx;
        float p3 = p * p * p;
        float q = kx * (2.0f * kx * kx - 3.0f * ky) + kz;
        float s = q * q + 4.0f * p3;

        float b2 = dot(b, b);
        float grad_kk_b2 = -1.0f / (b2 * b2);

        float3 grad_a_p_0 = make_float3(-1.0f, 0.0f, 0.0f);
        float3 grad_a_p_1 = make_float3(0.0f, -1.0f, 0.0f);
        float3 grad_a_p_2 = make_float3(0.0f, 0.0f, -1.0f);
        float3 grad_a_p_3 = make_float3(1.0f, 0.0f, 0.0f);
        float3 grad_a_p_4 = make_float3(0.0f, 1.0f, 0.0f);
        float3 grad_a_p_5 = make_float3(0.0f, 0.0f, 1.0f);
        float3 grad_a_p_6 = make_float3(0.0f, 0.0f, 0.0f);
        float3 grad_a_p_7 = make_float3(0.0f, 0.0f, 0.0f);
        float3 grad_a_p_8 = make_float3(0.0f, 0.0f, 0.0f);
        mat3 grad_a_xyz = mat3();

        float3 grad_b_p_0 = make_float3(1.0f, 0.0f, 0.0f);
        float3 grad_b_p_1 = make_float3(0.0f, 1.0f, 0.0f);
        float3 grad_b_p_2 = make_float3(0.0f, 0.0f, 1.0f);
        float3 grad_b_p_3 = make_float3(-2.0f, 0.0f, 0.0f);
        float3 grad_b_p_4 = make_float3(0.0f, -2.0f, 0.0f);
        float3 grad_b_p_5 = make_float3(0.0f, 0.0f, -2.0f);
        float3 grad_b_p_6 = make_float3(1.0f, 0.0f, 0.0f);
        float3 grad_b_p_7 = make_float3(0.0f, 1.0f, 0.0f);
        float3 grad_b_p_8 = make_float3(0.0f, 0.0f, 1.0f);
        mat3 grad_b_xyz = mat3();

        float3 grad_c_p_0 = make_float3(-2.0f, 0.0f, 0.0f);
        float3 grad_c_p_1 = make_float3(0.0f, -2.0f, 0.0f);
        float3 grad_c_p_2 = make_float3(0.0f, 0.0f, -2.0f);
        float3 grad_c_p_3 = make_float3(2.0f, 0.0f, 0.0f);
        float3 grad_c_p_4 = make_float3(0.0f, 2.0f, 0.0f);
        float3 grad_c_p_5 = make_float3(0.0f, 0.0f, 2.0f);
        float3 grad_c_p_6 = make_float3(0.0f, 0.0f, 0.0f);
        float3 grad_c_p_7 = make_float3(0.0f, 0.0f, 0.0f);
        float3 grad_c_p_8 = make_float3(0.0f, 0.0f, 0.0f);
        mat3 grad_c_xyz = mat3();

        float3 grad_d_p_0 = make_float3(1.0f, 0.0f, 0.0f);
        float3 grad_d_p_1 = make_float3(0.0f, 1.0f, 0.0f);
        float3 grad_d_p_2 = make_float3(0.0f, 0.0f, 1.0f);
        float3 grad_d_p_3 = make_float3(0.0f, 0.0f, 0.0f);
        float3 grad_d_p_4 = make_float3(0.0f, 0.0f, 0.0f);
        float3 grad_d_p_5 = make_float3(0.0f, 0.0f, 0.0f);
        float3 grad_d_p_6 = make_float3(0.0f, 0.0f, 0.0f);
        float3 grad_d_p_7 = make_float3(0.0f, 0.0f, 0.0f);
        float3 grad_d_p_8 = make_float3(0.0f, 0.0f, 0.0f);
        mat3 grad_d_xyz = mat3::identity() * (-1.0f);

        float grad_b2_p_0 = dot(2.0f * b, grad_b_p_0);
        float grad_b2_p_1 = dot(2.0f * b, grad_b_p_1);
        float grad_b2_p_2 = dot(2.0f * b, grad_b_p_2);
        float grad_b2_p_3 = dot(2.0f * b, grad_b_p_3);
        float grad_b2_p_4 = dot(2.0f * b, grad_b_p_4);
        float grad_b2_p_5 = dot(2.0f * b, grad_b_p_5);
        float grad_b2_p_6 = dot(2.0f * b, grad_b_p_6);
        float grad_b2_p_7 = dot(2.0f * b, grad_b_p_7);
        float grad_b2_p_8 = dot(2.0f * b, grad_b_p_8);
        float3 grad_b2_xyz = make_float3(0.0f, 0.0f, 0.0f);

        float grad_kk_p_0 = grad_kk_b2 * grad_b2_p_0;
        float grad_kk_p_1 = grad_kk_b2 * grad_b2_p_1;
        float grad_kk_p_2 = grad_kk_b2 * grad_b2_p_2;
        float grad_kk_p_3 = grad_kk_b2 * grad_b2_p_3;
        float grad_kk_p_4 = grad_kk_b2 * grad_b2_p_4;
        float grad_kk_p_5 = grad_kk_b2 * grad_b2_p_5;
        float grad_kk_p_6 = grad_kk_b2 * grad_b2_p_6;
        float grad_kk_p_7 = grad_kk_b2 * grad_b2_p_7;
        float grad_kk_p_8 = grad_kk_b2 * grad_b2_p_8;
        float3 grad_kk_xyz = make_float3(0.0f, 0.0f, 0.0f);

        float ab = dot(a, b);
        float grad_kx_ab = kk;
        float grad_kx_kk = ab;

        float grad_ab_p_0 = dot(b, grad_a_p_0) + dot(a, grad_b_p_0);
        float grad_ab_p_1 = dot(b, grad_a_p_1) + dot(a, grad_b_p_1);
        float grad_ab_p_2 = dot(b, grad_a_p_2) + dot(a, grad_b_p_2);
        float grad_ab_p_3 = dot(b, grad_a_p_3) + dot(a, grad_b_p_3);
        float grad_ab_p_4 = dot(b, grad_a_p_4) + dot(a, grad_b_p_4);
        float grad_ab_p_5 = dot(b, grad_a_p_5) + dot(a, grad_b_p_5);
        float grad_ab_p_6 = dot(b, grad_a_p_6) + dot(a, grad_b_p_6);
        float grad_ab_p_7 = dot(b, grad_a_p_7) + dot(a, grad_b_p_7);
        float grad_ab_p_8 = dot(b, grad_a_p_8) + dot(a, grad_b_p_8);
        float3 grad_ab_xyz = make_float3(0.0f, 0.0f, 0.0f);

        float grad_kx_p_0 = grad_kx_ab * grad_ab_p_0 + grad_kx_kk * grad_kk_p_0;
        float grad_kx_p_1 = grad_kx_ab * grad_ab_p_1 + grad_kx_kk * grad_kk_p_1;
        float grad_kx_p_2 = grad_kx_ab * grad_ab_p_2 + grad_kx_kk * grad_kk_p_2;
        float grad_kx_p_3 = grad_kx_ab * grad_ab_p_3 + grad_kx_kk * grad_kk_p_3;
        float grad_kx_p_4 = grad_kx_ab * grad_ab_p_4 + grad_kx_kk * grad_kk_p_4;
        float grad_kx_p_5 = grad_kx_ab * grad_ab_p_5 + grad_kx_kk * grad_kk_p_5;
        float grad_kx_p_6 = grad_kx_ab * grad_ab_p_6 + grad_kx_kk * grad_kk_p_6;
        float grad_kx_p_7 = grad_kx_ab * grad_ab_p_7 + grad_kx_kk * grad_kk_p_7;
        float grad_kx_p_8 = grad_kx_ab * grad_ab_p_8 + grad_kx_kk * grad_kk_p_8;
        float3 grad_kx_xyz = make_float3(0.0f, 0.0f, 0.0f);

        float aa = dot(a, a);
        float db = dot(d, b);
        float grad_ky_aa = 2.0f / 3.0f * kk;
        float grad_ky_db = 1.0f / 3.0f * kk;
        float grad_ky_kk = 1.0f / 3.0f * (2 * aa + db);

        float grad_aa_p_0 = dot(2.0f * a, grad_a_p_0);
        float grad_aa_p_1 = dot(2.0f * a, grad_a_p_1);
        float grad_aa_p_2 = dot(2.0f * a, grad_a_p_2);
        float grad_aa_p_3 = dot(2.0f * a, grad_a_p_3);
        float grad_aa_p_4 = dot(2.0f * a, grad_a_p_4);
        float grad_aa_p_5 = dot(2.0f * a, grad_a_p_5);
        float grad_aa_p_6 = dot(2.0f * a, grad_a_p_6);
        float grad_aa_p_7 = dot(2.0f * a, grad_a_p_7);
        float grad_aa_p_8 = dot(2.0f * a, grad_a_p_8);
        float3 grad_aa_xyz = make_float3(0.0f, 0.0f, 0.0f);

        float grad_db_p_0 = dot(b, grad_d_p_0) + dot(d, grad_b_p_0);
        float grad_db_p_1 = dot(b, grad_d_p_1) + dot(d, grad_b_p_1);
        float grad_db_p_2 = dot(b, grad_d_p_2) + dot(d, grad_b_p_2);
        float grad_db_p_3 = dot(b, grad_d_p_3) + dot(d, grad_b_p_3);
        float grad_db_p_4 = dot(b, grad_d_p_4) + dot(d, grad_b_p_4);
        float grad_db_p_5 = dot(b, grad_d_p_5) + dot(d, grad_b_p_5);
        float grad_db_p_6 = dot(b, grad_d_p_6) + dot(d, grad_b_p_6);
        float grad_db_p_7 = dot(b, grad_d_p_7) + dot(d, grad_b_p_7);
        float grad_db_p_8 = dot(b, grad_d_p_8) + dot(d, grad_b_p_8);
        float3 grad_db_xyz = b * grad_d_xyz + d * grad_b_xyz;

        float grad_ky_p_0 = grad_ky_aa * grad_aa_p_0 + grad_ky_db * grad_db_p_0 + grad_ky_kk * grad_kk_p_0;
        float grad_ky_p_1 = grad_ky_aa * grad_aa_p_1 + grad_ky_db * grad_db_p_1 + grad_ky_kk * grad_kk_p_1;
        float grad_ky_p_2 = grad_ky_aa * grad_aa_p_2 + grad_ky_db * grad_db_p_2 + grad_ky_kk * grad_kk_p_2;
        float grad_ky_p_3 = grad_ky_aa * grad_aa_p_3 + grad_ky_db * grad_db_p_3 + grad_ky_kk * grad_kk_p_3;
        float grad_ky_p_4 = grad_ky_aa * grad_aa_p_4 + grad_ky_db * grad_db_p_4 + grad_ky_kk * grad_kk_p_4;
        float grad_ky_p_5 = grad_ky_aa * grad_aa_p_5 + grad_ky_db * grad_db_p_5 + grad_ky_kk * grad_kk_p_5;
        float grad_ky_p_6 = grad_ky_aa * grad_aa_p_6 + grad_ky_db * grad_db_p_6 + grad_ky_kk * grad_kk_p_6;
        float grad_ky_p_7 = grad_ky_aa * grad_aa_p_7 + grad_ky_db * grad_db_p_7 + grad_ky_kk * grad_kk_p_7;
        float grad_ky_p_8 = grad_ky_aa * grad_aa_p_8 + grad_ky_db * grad_db_p_8 + grad_ky_kk * grad_kk_p_8;
        float3 grad_ky_xyz = grad_ky_aa * grad_aa_xyz + grad_ky_db * grad_db_xyz + grad_ky_kk * grad_kk_xyz;

        float da = dot(d, a);
        float grad_kz_da = kk;
        float grad_kz_kk = da;

        float grad_da_p_0 = dot(d, grad_a_p_0) + dot(a, grad_d_p_0);
        float grad_da_p_1 = dot(d, grad_a_p_1) + dot(a, grad_d_p_1);
        float grad_da_p_2 = dot(d, grad_a_p_2) + dot(a, grad_d_p_2);
        float grad_da_p_3 = dot(d, grad_a_p_3) + dot(a, grad_d_p_3);
        float grad_da_p_4 = dot(d, grad_a_p_4) + dot(a, grad_d_p_4);
        float grad_da_p_5 = dot(d, grad_a_p_5) + dot(a, grad_d_p_5);
        float grad_da_p_6 = dot(d, grad_a_p_6) + dot(a, grad_d_p_6);
        float grad_da_p_7 = dot(d, grad_a_p_7) + dot(a, grad_d_p_7);
        float grad_da_p_8 = dot(d, grad_a_p_8) + dot(a, grad_d_p_8);
        float3 grad_da_xyz = d * grad_a_xyz + a * grad_d_xyz;

        float grad_kz_p_0 = grad_kz_da * grad_da_p_0 + grad_kz_kk * grad_kk_p_0;
        float grad_kz_p_1 = grad_kz_da * grad_da_p_1 + grad_kz_kk * grad_kk_p_1;
        float grad_kz_p_2 = grad_kz_da * grad_da_p_2 + grad_kz_kk * grad_kk_p_2;
        float grad_kz_p_3 = grad_kz_da * grad_da_p_3 + grad_kz_kk * grad_kk_p_3;
        float grad_kz_p_4 = grad_kz_da * grad_da_p_4 + grad_kz_kk * grad_kk_p_4;
        float grad_kz_p_5 = grad_kz_da * grad_da_p_5 + grad_kz_kk * grad_kk_p_5;
        float grad_kz_p_6 = grad_kz_da * grad_da_p_6 + grad_kz_kk * grad_kk_p_6;
        float grad_kz_p_7 = grad_kz_da * grad_da_p_7 + grad_kz_kk * grad_kk_p_7;
        float grad_kz_p_8 = grad_kz_da * grad_da_p_8 + grad_kz_kk * grad_kk_p_8;
        float3 grad_kz_xyz = grad_kz_da * grad_da_xyz + grad_kz_kk * grad_kk_xyz;

        float grad_p_p_0 = grad_ky_p_0 - 2.0f * kx * grad_kx_p_0;
        float grad_p_p_1 = grad_ky_p_1 - 2.0f * kx * grad_kx_p_1;
        float grad_p_p_2 = grad_ky_p_2 - 2.0f * kx * grad_kx_p_2;
        float grad_p_p_3 = grad_ky_p_3 - 2.0f * kx * grad_kx_p_3;
        float grad_p_p_4 = grad_ky_p_4 - 2.0f * kx * grad_kx_p_4;
        float grad_p_p_5 = grad_ky_p_5 - 2.0f * kx * grad_kx_p_5;
        float grad_p_p_6 = grad_ky_p_6 - 2.0f * kx * grad_kx_p_6;
        float grad_p_p_7 = grad_ky_p_7 - 2.0f * kx * grad_kx_p_7;
        float grad_p_p_8 = grad_ky_p_8 - 2.0f * kx * grad_kx_p_8;
        float3 grad_p_xyz = grad_ky_xyz - 2.0f * kx * grad_kx_xyz;

        float grad_q_p_0 = 6.0f * kx * kx * grad_kx_p_0 - 3.0f * ky * grad_kx_p_0 - 3.0f * kx * grad_ky_p_0 + grad_kz_p_0;
        float grad_q_p_1 = 6.0f * kx * kx * grad_kx_p_1 - 3.0f * ky * grad_kx_p_1 - 3.0f * kx * grad_ky_p_1 + grad_kz_p_1;
        float grad_q_p_2 = 6.0f * kx * kx * grad_kx_p_2 - 3.0f * ky * grad_kx_p_2 - 3.0f * kx * grad_ky_p_2 + grad_kz_p_2;
        float grad_q_p_3 = 6.0f * kx * kx * grad_kx_p_3 - 3.0f * ky * grad_kx_p_3 - 3.0f * kx * grad_ky_p_3 + grad_kz_p_3;
        float grad_q_p_4 = 6.0f * kx * kx * grad_kx_p_4 - 3.0f * ky * grad_kx_p_4 - 3.0f * kx * grad_ky_p_4 + grad_kz_p_4;
        float grad_q_p_5 = 6.0f * kx * kx * grad_kx_p_5 - 3.0f * ky * grad_kx_p_5 - 3.0f * kx * grad_ky_p_5 + grad_kz_p_5;
        float grad_q_p_6 = 6.0f * kx * kx * grad_kx_p_6 - 3.0f * ky * grad_kx_p_6 - 3.0f * kx * grad_ky_p_6 + grad_kz_p_6;
        float grad_q_p_7 = 6.0f * kx * kx * grad_kx_p_7 - 3.0f * ky * grad_kx_p_7 - 3.0f * kx * grad_ky_p_7 + grad_kz_p_7;
        float grad_q_p_8 = 6.0f * kx * kx * grad_kx_p_8 - 3.0f * ky * grad_kx_p_8 - 3.0f * kx * grad_ky_p_8 + grad_kz_p_8;
        float3 grad_q_xyz = 6.0f * kx * kx * grad_kx_xyz - 3.0f * ky * grad_kx_xyz - 3.0f * kx * grad_ky_xyz + grad_kz_xyz;

        if (s >= 0.0f)
        {
            float h = sqrtf(s + 1e-8f);
            float2 x = make_float2((h - q) * 0.5f, -(h + q) * 0.5f);
            float2 uv = make_float2(
                copysignf(powf(fabs(x.x), 1.0f / 3.0f), x.x),
                copysignf(powf(fabs(x.y), 1.0f / 3.0f), x.y));
            float t = clamp((uv.x + uv.y) - kx, 0.0, 1.0);

            res = make_float2(dot(d + (c + b * t) * t, d + (c + b * t) * t), t);
            if (t == 0.0f || t == 1.0f)
            {
                if (grad_params) {
                    atomicAdd(grad_params + 0, grad_SDF * grad_ans_params_category_3(t, d, grad_d_p_0, c, b, grad_c_p_0, grad_b_p_0, res.x));
                    atomicAdd(grad_params + 1, grad_SDF * grad_ans_params_category_3(t, d, grad_d_p_1, c, b, grad_c_p_1, grad_b_p_1, res.x));
                    atomicAdd(grad_params + 2, grad_SDF * grad_ans_params_category_3(t, d, grad_d_p_2, c, b, grad_c_p_2, grad_b_p_2, res.x));
                    atomicAdd(grad_params + 3, grad_SDF * grad_ans_params_category_3(t, d, grad_d_p_3, c, b, grad_c_p_3, grad_b_p_3, res.x));
                    atomicAdd(grad_params + 4, grad_SDF * grad_ans_params_category_3(t, d, grad_d_p_4, c, b, grad_c_p_4, grad_b_p_4, res.x));
                    atomicAdd(grad_params + 5, grad_SDF * grad_ans_params_category_3(t, d, grad_d_p_5, c, b, grad_c_p_5, grad_b_p_5, res.x));
                    atomicAdd(grad_params + 6, grad_SDF * grad_ans_params_category_3(t, d, grad_d_p_6, c, b, grad_c_p_6, grad_b_p_6, res.x));
                    atomicAdd(grad_params + 7, grad_SDF * grad_ans_params_category_3(t, d, grad_d_p_7, c, b, grad_c_p_7, grad_b_p_7, res.x));
                    atomicAdd(grad_params + 8, grad_SDF * grad_ans_params_category_3(t, d, grad_d_p_8, c, b, grad_c_p_8, grad_b_p_8, res.x));
                    atomicAdd(grad_params + 9, -grad_SDF * (-res.y + 1.0f));
                    atomicAdd(grad_params + 10, -grad_SDF * (res.y));
                }

                return grad_ans_xyz_category_3(d, res.x);
            }
            else
            {
                if (grad_params) {
                    atomicAdd(grad_params + 0, grad_SDF * grad_ans_params_category_1(s, b, c, d, t, x, q, p, grad_kx_p_0, grad_q_p_0, grad_p_p_0, grad_d_p_0, grad_c_p_0, grad_b_p_0, params[9], params[10], res.x));
                    atomicAdd(grad_params + 1, grad_SDF * grad_ans_params_category_1(s, b, c, d, t, x, q, p, grad_kx_p_1, grad_q_p_1, grad_p_p_1, grad_d_p_1, grad_c_p_1, grad_b_p_1, params[9], params[10], res.x));
                    atomicAdd(grad_params + 2, grad_SDF * grad_ans_params_category_1(s, b, c, d, t, x, q, p, grad_kx_p_2, grad_q_p_2, grad_p_p_2, grad_d_p_2, grad_c_p_2, grad_b_p_2, params[9], params[10], res.x));
                    atomicAdd(grad_params + 3, grad_SDF * grad_ans_params_category_1(s, b, c, d, t, x, q, p, grad_kx_p_3, grad_q_p_3, grad_p_p_3, grad_d_p_3, grad_c_p_3, grad_b_p_3, params[9], params[10], res.x));
                    atomicAdd(grad_params + 4, grad_SDF * grad_ans_params_category_1(s, b, c, d, t, x, q, p, grad_kx_p_4, grad_q_p_4, grad_p_p_4, grad_d_p_4, grad_c_p_4, grad_b_p_4, params[9], params[10], res.x));
                    atomicAdd(grad_params + 5, grad_SDF * grad_ans_params_category_1(s, b, c, d, t, x, q, p, grad_kx_p_5, grad_q_p_5, grad_p_p_5, grad_d_p_5, grad_c_p_5, grad_b_p_5, params[9], params[10], res.x));
                    atomicAdd(grad_params + 6, grad_SDF * grad_ans_params_category_1(s, b, c, d, t, x, q, p, grad_kx_p_6, grad_q_p_6, grad_p_p_6, grad_d_p_6, grad_c_p_6, grad_b_p_6, params[9], params[10], res.x));
                    atomicAdd(grad_params + 7, grad_SDF * grad_ans_params_category_1(s, b, c, d, t, x, q, p, grad_kx_p_7, grad_q_p_7, grad_p_p_7, grad_d_p_7, grad_c_p_7, grad_b_p_7, params[9], params[10], res.x));
                    atomicAdd(grad_params + 8, grad_SDF * grad_ans_params_category_1(s, b, c, d, t, x, q, p, grad_kx_p_8, grad_q_p_8, grad_p_p_8, grad_d_p_8, grad_c_p_8, grad_b_p_8, params[9], params[10], res.x));
                    atomicAdd(grad_params + 9, -grad_SDF * (-res.y + 1.0f));
                    atomicAdd(grad_params + 10, -grad_SDF * (res.y));
                }

                return grad_ans_xyz_category_1(b, c, d, t, x, s, p, q, grad_q_xyz, grad_p_xyz, grad_d_xyz, params[9], params[10], res.x);
            }
        }
        else
        {
            float z = sqrtf(-p + 1e-8f);
            float v = acosf(q / (p * z * 2.0f + 1e-8f)) / 3.0f;
            float m = cosf(v);
            float n = sinf(v) * 1.732050808f;
            float3 t = clamp(make_float3(m + n, -m - n, n - m) * z - kx, 0.0f, 1.0f);

            float dis = dot(d + (c + b * t.x) * t.x, d + (c + b * t.x) * t.x);
            res = make_float2(dis, t.x);

            dis = dot(d + (c + b * t.y) * t.y, d + (c + b * t.y) * t.y);

            if (dis < res.x)
            {
                res = make_float2(dis, t.y);
                if (t.y == 0.0f || t.y == 1.0f)
                {
                    if (grad_params) {
                        atomicAdd(grad_params + 0, grad_SDF * grad_ans_params_category_3(t.y, d, grad_d_p_0, c, b, grad_c_p_0, grad_b_p_0, res.x));
                        atomicAdd(grad_params + 1, grad_SDF * grad_ans_params_category_3(t.y, d, grad_d_p_1, c, b, grad_c_p_1, grad_b_p_1, res.x));
                        atomicAdd(grad_params + 2, grad_SDF * grad_ans_params_category_3(t.y, d, grad_d_p_2, c, b, grad_c_p_2, grad_b_p_2, res.x));
                        atomicAdd(grad_params + 3, grad_SDF * grad_ans_params_category_3(t.y, d, grad_d_p_3, c, b, grad_c_p_3, grad_b_p_3, res.x));
                        atomicAdd(grad_params + 4, grad_SDF * grad_ans_params_category_3(t.y, d, grad_d_p_4, c, b, grad_c_p_4, grad_b_p_4, res.x));
                        atomicAdd(grad_params + 5, grad_SDF * grad_ans_params_category_3(t.y, d, grad_d_p_5, c, b, grad_c_p_5, grad_b_p_5, res.x));
                        atomicAdd(grad_params + 6, grad_SDF * grad_ans_params_category_3(t.y, d, grad_d_p_6, c, b, grad_c_p_6, grad_b_p_6, res.x));
                        atomicAdd(grad_params + 7, grad_SDF * grad_ans_params_category_3(t.y, d, grad_d_p_7, c, b, grad_c_p_7, grad_b_p_7, res.x));
                        atomicAdd(grad_params + 8, grad_SDF * grad_ans_params_category_3(t.y, d, grad_d_p_8, c, b, grad_c_p_8, grad_b_p_8, res.x));
                        atomicAdd(grad_params + 9, -grad_SDF * (-res.y + 1.0f));
                        atomicAdd(grad_params + 10, -grad_SDF * (res.y));
                    }

                    return grad_ans_xyz_category_3(d, res.x);
                }
                else
                {
                    if (grad_params) {
                        atomicAdd(grad_params + 0, grad_SDF * grad_ans_params_category_2(b, c, d, t.y, z, m, n, p, v, q, grad_q_p_0, grad_p_p_0, grad_d_p_0, grad_kx_p_0, grad_c_p_0, grad_b_p_0, params[9], params[10], res.x));
                        atomicAdd(grad_params + 1, grad_SDF * grad_ans_params_category_2(b, c, d, t.y, z, m, n, p, v, q, grad_q_p_1, grad_p_p_1, grad_d_p_1, grad_kx_p_1, grad_c_p_1, grad_b_p_1, params[9], params[10], res.x));
                        atomicAdd(grad_params + 2, grad_SDF * grad_ans_params_category_2(b, c, d, t.y, z, m, n, p, v, q, grad_q_p_2, grad_p_p_2, grad_d_p_2, grad_kx_p_2, grad_c_p_2, grad_b_p_2, params[9], params[10], res.x));
                        atomicAdd(grad_params + 3, grad_SDF * grad_ans_params_category_2(b, c, d, t.y, z, m, n, p, v, q, grad_q_p_3, grad_p_p_3, grad_d_p_3, grad_kx_p_3, grad_c_p_3, grad_b_p_3, params[9], params[10], res.x));
                        atomicAdd(grad_params + 4, grad_SDF * grad_ans_params_category_2(b, c, d, t.y, z, m, n, p, v, q, grad_q_p_4, grad_p_p_4, grad_d_p_4, grad_kx_p_4, grad_c_p_4, grad_b_p_4, params[9], params[10], res.x));
                        atomicAdd(grad_params + 5, grad_SDF * grad_ans_params_category_2(b, c, d, t.y, z, m, n, p, v, q, grad_q_p_5, grad_p_p_5, grad_d_p_5, grad_kx_p_5, grad_c_p_5, grad_b_p_5, params[9], params[10], res.x));
                        atomicAdd(grad_params + 6, grad_SDF * grad_ans_params_category_2(b, c, d, t.y, z, m, n, p, v, q, grad_q_p_6, grad_p_p_6, grad_d_p_6, grad_kx_p_6, grad_c_p_6, grad_b_p_6, params[9], params[10], res.x));
                        atomicAdd(grad_params + 7, grad_SDF * grad_ans_params_category_2(b, c, d, t.y, z, m, n, p, v, q, grad_q_p_7, grad_p_p_7, grad_d_p_7, grad_kx_p_7, grad_c_p_7, grad_b_p_7, params[9], params[10], res.x));
                        atomicAdd(grad_params + 8, grad_SDF * grad_ans_params_category_2(b, c, d, t.y, z, m, n, p, v, q, grad_q_p_8, grad_p_p_8, grad_d_p_8, grad_kx_p_8, grad_c_p_8, grad_b_p_8, params[9], params[10], res.x));
                        atomicAdd(grad_params + 9, -grad_SDF * (-res.y + 1.0f));
                        atomicAdd(grad_params + 10, -grad_SDF * (res.y));
                    }

                    return grad_ans_xyz_category_2(b, c, d, t.y, z, m, n, p, v, q, grad_q_xyz, grad_p_xyz, grad_d_xyz, params[9], params[10], res.x);
                }
            }
            else
            {
                if (t.x == 0.0f || t.x == 1.0f)
                {
                    if (grad_params) {
                        atomicAdd(grad_params + 0, grad_SDF * grad_ans_params_category_3(t.x, d, grad_d_p_0, c, b, grad_c_p_0, grad_b_p_0, res.x));
                        atomicAdd(grad_params + 1, grad_SDF * grad_ans_params_category_3(t.x, d, grad_d_p_1, c, b, grad_c_p_1, grad_b_p_1, res.x));
                        atomicAdd(grad_params + 2, grad_SDF * grad_ans_params_category_3(t.x, d, grad_d_p_2, c, b, grad_c_p_2, grad_b_p_2, res.x));
                        atomicAdd(grad_params + 3, grad_SDF * grad_ans_params_category_3(t.x, d, grad_d_p_3, c, b, grad_c_p_3, grad_b_p_3, res.x));
                        atomicAdd(grad_params + 4, grad_SDF * grad_ans_params_category_3(t.x, d, grad_d_p_4, c, b, grad_c_p_4, grad_b_p_4, res.x));
                        atomicAdd(grad_params + 5, grad_SDF * grad_ans_params_category_3(t.x, d, grad_d_p_5, c, b, grad_c_p_5, grad_b_p_5, res.x));
                        atomicAdd(grad_params + 6, grad_SDF * grad_ans_params_category_3(t.x, d, grad_d_p_6, c, b, grad_c_p_6, grad_b_p_6, res.x));
                        atomicAdd(grad_params + 7, grad_SDF * grad_ans_params_category_3(t.x, d, grad_d_p_7, c, b, grad_c_p_7, grad_b_p_7, res.x));
                        atomicAdd(grad_params + 8, grad_SDF * grad_ans_params_category_3(t.x, d, grad_d_p_8, c, b, grad_c_p_8, grad_b_p_8, res.x));
                        atomicAdd(grad_params + 9, -grad_SDF * (-res.y + 1.0f));
                        atomicAdd(grad_params + 10, -grad_SDF * (res.y));
                    }

                    return grad_ans_xyz_category_3(d, res.x);
                }
                else
                {
                    if (grad_params) {
                        atomicAdd(grad_params + 0, grad_SDF * grad_ans_params_category_2(b, c, d, t.x, z, m, n, p, v, q, grad_q_p_0, grad_p_p_0, grad_d_p_0, grad_kx_p_0, grad_c_p_0, grad_b_p_0, params[9], params[10], res.x));
                        atomicAdd(grad_params + 1, grad_SDF * grad_ans_params_category_2(b, c, d, t.x, z, m, n, p, v, q, grad_q_p_1, grad_p_p_1, grad_d_p_1, grad_kx_p_1, grad_c_p_1, grad_b_p_1, params[9], params[10], res.x));
                        atomicAdd(grad_params + 2, grad_SDF * grad_ans_params_category_2(b, c, d, t.x, z, m, n, p, v, q, grad_q_p_2, grad_p_p_2, grad_d_p_2, grad_kx_p_2, grad_c_p_2, grad_b_p_2, params[9], params[10], res.x));
                        atomicAdd(grad_params + 3, grad_SDF * grad_ans_params_category_2(b, c, d, t.x, z, m, n, p, v, q, grad_q_p_3, grad_p_p_3, grad_d_p_3, grad_kx_p_3, grad_c_p_3, grad_b_p_3, params[9], params[10], res.x));
                        atomicAdd(grad_params + 4, grad_SDF * grad_ans_params_category_2(b, c, d, t.x, z, m, n, p, v, q, grad_q_p_4, grad_p_p_4, grad_d_p_4, grad_kx_p_4, grad_c_p_4, grad_b_p_4, params[9], params[10], res.x));
                        atomicAdd(grad_params + 5, grad_SDF * grad_ans_params_category_2(b, c, d, t.x, z, m, n, p, v, q, grad_q_p_5, grad_p_p_5, grad_d_p_5, grad_kx_p_5, grad_c_p_5, grad_b_p_5, params[9], params[10], res.x));
                        atomicAdd(grad_params + 6, grad_SDF * grad_ans_params_category_2(b, c, d, t.x, z, m, n, p, v, q, grad_q_p_6, grad_p_p_6, grad_d_p_6, grad_kx_p_6, grad_c_p_6, grad_b_p_6, params[9], params[10], res.x));
                        atomicAdd(grad_params + 7, grad_SDF * grad_ans_params_category_2(b, c, d, t.x, z, m, n, p, v, q, grad_q_p_7, grad_p_p_7, grad_d_p_7, grad_kx_p_7, grad_c_p_7, grad_b_p_7, params[9], params[10], res.x));
                        atomicAdd(grad_params + 8, grad_SDF * grad_ans_params_category_2(b, c, d, t.x, z, m, n, p, v, q, grad_q_p_8, grad_p_p_8, grad_d_p_8, grad_kx_p_8, grad_c_p_8, grad_b_p_8, params[9], params[10], res.x));
                        atomicAdd(grad_params + 9, -grad_SDF * (res.y + 1.0f));
                        atomicAdd(grad_params + 10, -grad_SDF * (res.y));
                    }

                    return grad_ans_xyz_category_2(b, c, d, t.x, z, m, n, p, v, q, grad_q_xyz, grad_p_xyz, grad_d_xyz, params[9], params[10], res.x);
                }
            }
        }
    }

    __device__ static float3 grad_ans_xyz_category_1(float3 b, float3 c, float3 d, float t, float2 x, float s, float p, float q, float3 grad_q_xyz, float3 grad_p_xyz, mat3 grad_d_xyz, float r1, float r2, float resx)
    {
        float3 m = d + (c + b * t) * t;
        float3 grad_ansx_m = 2.0f * m;
        float grad_m_d = 1.0f;
        float3 grad_m_t = c + 2.0f * b * t;

        float grad_uvx_h = (powf(x.x * x.x, -1.0f / 3.0f), x.x) / 6.0f;
        float grad_uvy_h = -1.0f * (powf(x.y * x.y, -1.0f / 3.0f), x.y) / 6.0f;
        float grad_h_s = 1.0f / 2.0f * rsqrtf(s + 1e-8f);
        float grad_uvx_s = grad_uvx_h * grad_h_s;
        float grad_uvy_s = grad_uvy_h * grad_h_s;

        float grad_s_q = 2.0f * q;
        float grad_s_p = 12.0f * p * p;
        float grad_uvx_q = -1.0f * grad_uvx_h;
        float grad_uvy_q = -1.0f * grad_uvy_h;

        float3 grad_t_xyz = (grad_uvx_q * grad_q_xyz + grad_uvx_s * (grad_s_q * grad_q_xyz + grad_s_p * grad_p_xyz)) + (grad_uvy_q * grad_q_xyz + grad_uvy_s * (grad_s_q * grad_q_xyz + grad_s_p * grad_p_xyz));
        float3 grad_ansx_xyz = grad_ansx_m * (grad_d_xyz * grad_m_d + outerproduct(grad_t_xyz, grad_m_t));
        grad_ansx_xyz = grad_ansx_xyz * (1.0f / 2.0f * rsqrtf(resx + 1e-8));

        return grad_ansx_xyz - (r2 - r1) * grad_t_xyz;
    }

    __device__ static float3 grad_ans_xyz_category_2(float3 b, float3 c, float3 d, float t, float z, float m, float n, float p, float v, float q, float3 grad_q_xyz, float3 grad_p_xyz, mat3 grad_d_xyz, float r1, float r2, float resx)
    {
        float3 h = d + (c + b * t) * t;
        float3 grad_ansx_h = 2.0f * h;
        float grad_h_d = 1.0f;
        float3 grad_h_t = c + 2.0f * b * t;
        float grad_t_n = -1.0f * z;
        float grad_t_m = -1.0f * z;
        float grad_t_z = -m - n;
        float grad_z_p = -1.0f / 2.0f * rsqrtf(-p + 1e-8f);
        float grad_n_v = cosf(v) * 1.732050808f;
        float grad_m_v = -sinf(v);
        float grad_v_q = 1.0f / (p * z * 2.0f + 1e-8f) * (-1.0f) * rsqrtf(1.0f - q * q / (p * p * z * z * 4.0f) + 1e-8f) / 3.0f;
        float grad_v_p = 1.0f / p * q / (p * z * 2.0f + 1e-8f) * rsqrtf(1.0f - q * q / (p * p * z * z * 4.0f) + 1e-8f) / 3.0f;
        float grad_v_z = 1.0f / z * q / (p * z * 2.0f + 1e-8f) * rsqrtf(1.0f - q * q / (p * p * z * z * 4.0f) + 1e-8f) / 3.0f;

        float3 grad_v_xyz = grad_q_xyz * grad_v_q + grad_p_xyz * grad_v_p + grad_p_xyz * (grad_v_z * grad_z_p);

        float3 grad_t_xyz = (grad_v_xyz * (grad_t_n * grad_n_v) + grad_v_xyz * (grad_t_m * grad_m_v)) + grad_p_xyz * (grad_t_z * grad_z_p);
        float3 grad_ansx_xyz = grad_ansx_h * (grad_d_xyz * grad_h_d + outerproduct(grad_t_xyz, grad_h_t));
        grad_ansx_xyz = grad_ansx_xyz * (1.0f / 2.0f * rsqrtf(resx + 1e-8));

        return grad_ansx_xyz - (r2 - r1) * grad_t_xyz;
    }

    __device__ static float3 grad_ans_xyz_category_3(float3 d, float resx)
    {

        return d * (-1.0f * rsqrtf(resx + 1e-8f));
    }

    __device__ static float grad_ans_params_category_1(float s, float3 b, float3 c, float3 d, float t, float2 x, float q, float p, float grad_kx_p, float grad_q_p, float grad_p_p, float3 grad_d_p, float3 grad_c_p, float3 grad_b_p, float r1, float r2, float resx)
    {
        float3 m = d + (c + b * t) * t;
        float3 grad_ansx_m = 2.0f * m;
        float grad_m_d = 1.0f;
        float3 grad_m_t = c + 2.0f * b * t;

        float grad_uvx_h = (powf(x.x * x.x + 1e-8, -1.0f / 3.0f), x.x) / 3.0f;
        float grad_uvy_h = -1.0f * (powf(x.y * x.y + 1e-8, -1.0f / 3.0f), x.y) / 3.0f;
        float grad_h_s = 1.0f / 2.0f * rsqrtf(s + 1e-8f);
        float grad_uvx_s = grad_uvx_h * grad_h_s;
        float grad_uvy_s = grad_uvy_h * grad_h_s;

        float grad_s_q = 2.0f * q;
        float grad_s_p = 12.0f * p * p;
        float grad_uvx_q = -1.0f * grad_uvx_h;
        float grad_uvy_q = -1.0f * grad_uvy_h;
        float grad_m_c = t;
        float grad_m_b = t * t;

        float grad_t_p = (grad_uvx_q * grad_q_p + grad_uvx_s * (grad_s_q * grad_q_p + grad_s_p * grad_p_p)) + (grad_uvy_q * grad_q_p + grad_uvy_s * (grad_s_q * grad_q_p + grad_s_p * grad_p_p)) - grad_kx_p;

        float3 grad_m_p = grad_d_p * grad_m_d + grad_c_p * grad_m_c + grad_m_t * grad_t_p + grad_b_p * grad_m_b;

        float grad_ansx_p = dot(grad_ansx_m, grad_m_p);
        grad_ansx_p = grad_ansx_p * (1.0f / 2.0f * rsqrtf(resx + 1e-8));

        return grad_ansx_p - (r2 - r1) * grad_t_p;
    }

    __device__ static float grad_ans_params_category_2(float3 b, float3 c, float3 d, float t, float z, float m, float n, float p, float v, float q, float grad_q_p, float grad_p_p, float3 grad_d_p, float grad_kx_p, float3 grad_c_p, float3 grad_b_p, float r1, float r2, float resx)
    {
        float3 h = d + (c + b * t) * t;
        float3 grad_ansx_h = 2.0f * h;
        float grad_t_n = -1.0f * z;
        float grad_t_m = -1.0f * z;
        float grad_t_z = -m - n;
        float grad_z_p = -1.0f / 2.0f * rsqrtf(-p + 1e-8f);
        float grad_n_v = cosf(v) * 1.732050808f;
        float grad_m_v = -sinf(v);
        float grad_v_q = 1.0f / (p * z * 2.0f + 1e-8f) * (-1.0f) * rsqrtf(1.0f - q * q / (p * p * z * z * 4.0f) + 1e-8f) / 3.0f;
        float grad_v_p = 1.0f / p * q / (p * z * 2.0f + 1e-8f) * rsqrtf(1.0f - q * q / (p * p * z * z * 4.0f) + 1e-8f) / 3.0f;
        float grad_v_z = 1.0f / z * q / (p * z * 2.0f + 1e-8f) * rsqrtf(1.0f - q * q / (p * p * z * z * 4.0f) + 1e-8f) / 3.0f;

        float grad_v_p_i = grad_v_q * grad_q_p + grad_v_p * grad_p_p + grad_v_z * grad_z_p * grad_p_p;

        float grad_t_p = (grad_t_n * grad_n_v * grad_v_p_i + grad_t_m * grad_m_v * grad_v_p_i) + grad_t_z * grad_z_p * grad_p_p - grad_kx_p;

        float3 grad_h_p = grad_d_p + c * grad_t_p + grad_c_p * t + b * 2.0f * t * grad_t_p + grad_b_p * t * t;

        float grad_ansx_p = dot(grad_ansx_h, grad_h_p);
        grad_ansx_p = grad_ansx_p * (1.0f / 2.0f * rsqrtf(resx + 1e-8));

        return grad_ansx_p - (r2 - r1) * grad_t_p;
    }

    __device__ static float grad_ans_params_category_3(float t, float3 d, float3 grad_d_p, float3 c, float3 b, float3 grad_c_p, float3 grad_b_p, float resx)
    {

        return (t == 0.0f ? dot(2.0f * d, grad_d_p) : dot(2.0f * (d + c + b), grad_d_p + grad_c_p + grad_b_p)) * (1.0f / 2.0f * rsqrtf(resx + 1e-8f));
    }

#else // finite difference gradient
    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float delta = 1e-6f;
        if (grad_params) {
            atomicAdd(grad_params + 0, grad_SDF * grad_param_i<0>(pos, params, delta));
            atomicAdd(grad_params + 1, grad_SDF * grad_param_i<1>(pos, params, delta));
            atomicAdd(grad_params + 2, grad_SDF * grad_param_i<2>(pos, params, delta));
            atomicAdd(grad_params + 3, grad_SDF * grad_param_i<3>(pos, params, delta));
            atomicAdd(grad_params + 4, grad_SDF * grad_param_i<4>(pos, params, delta));
            atomicAdd(grad_params + 5, grad_SDF * grad_param_i<5>(pos, params, delta));
            atomicAdd(grad_params + 6, grad_SDF * grad_param_i<6>(pos, params, delta));
            atomicAdd(grad_params + 7, grad_SDF * grad_param_i<7>(pos, params, delta));
            atomicAdd(grad_params + 8, grad_SDF * grad_param_i<8>(pos, params, delta));
            atomicAdd(grad_params + 9, grad_SDF * grad_param_i<9>(pos, params, delta));
            atomicAdd(grad_params + 10, grad_SDF * grad_param_i<10>(pos, params, delta));
        }
        float grad_x = (sdf(pos + make_float3(delta, 0.0f, 0.0f), params) - sdf(pos - make_float3(delta, 0.0f, 0.0f), params)) / (2.0f * delta);
        float grad_y = (sdf(pos + make_float3(0.0f, delta, 0.0f), params) - sdf(pos - make_float3(0.0f, delta, 0.0f), params)) / (2.0f * delta);
        float grad_z = (sdf(pos + make_float3(0.0f, 0.0f, delta), params) - sdf(pos - make_float3(0.0f, 0.0f, delta), params)) / (2.0f * delta);
        return make_float3(grad_x, grad_y, grad_z);
    }

    template <int i, int j>
    __device__ static inline float get_param(const float *params, float delta)
    {
        return params[j] + (j == i ? delta : 0.0f);
    }

    template <int i>
    __device__ static float get_sdf_delta(float3 pos, const float *params, float delta)
    {
        float3 A = make_float3(get_param<i, 0>(params, delta), get_param<i, 1>(params, delta), get_param<i, 2>(params, delta));
        float3 B = make_float3(get_param<i, 3>(params, delta), get_param<i, 4>(params, delta), get_param<i, 5>(params, delta));
        float3 C = make_float3(get_param<i, 6>(params, delta), get_param<i, 7>(params, delta), get_param<i, 8>(params, delta));
        return sdf_quad_bezier(pos, A, B, C, get_param<i, 9>(params, delta), get_param<i, 10>(params, delta));
    }

    template <int i>
    __device__ static float grad_param_i(float3 pos, const float *params, float delta)
    {
        float sdf_positive = get_sdf_delta<i>(pos, params, +delta);
        float sdf_negative = get_sdf_delta<i>(pos, params, -delta);
        float grad_params_i = (sdf_positive - sdf_negative) / (2.0f * delta);
        return grad_params_i;
    }
#endif

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        return make_float2(0.0f, 0.0f);
    }
};

__device__ inline float2 point_segment_distance(float3 point, float3 A, float3 B)
{
    float3 AB = B - A;
    float t = dot(point - A, AB) / dot(AB, AB);
    t = clamp(t, 0.0f, 1.0f);
    float3 P = A + t * AB;
    return make_float2(length(point - P), t);
}

template <>
struct BaseSDF<CUBIC_BEZIER>
{
    static constexpr int num_samples = 10;

    __device__ static inline float3 point_cubic_bezier(float t, float3 A, float3 B, float3 C, float3 D)
    {
        float u = 1.0f - t;
        float w0 = u * u * u;
        float w1 = 3.0f * u * u * t;
        float w2 = 3.0f * u * t * t;
        float w3 = t * t * t;
        return w0 * A + w1 * B + w2 * C + w3 * D;
    }

    __device__ static float2 sdf_cubic_bezier(float3 pos, float3 A, float3 B, float3 C, float3 D, float r1, float r2)
    {
        constexpr float step = 1.0f / num_samples;
        float dist_min = 1e9f;
        float t_min = 0.0f;
        float t_start = 0.0f;
        float3 segment_start = point_cubic_bezier(t_start, A, B, C, D);
        for (int i = 1; i <= num_samples; i++) {
            float t_end = i * step;
            float3 segment_end = point_cubic_bezier(t_end, A, B, C, D);
            float2 dist_and_t = point_segment_distance(pos, segment_start, segment_end);
            if (dist_and_t.x < dist_min) {
                dist_min = dist_and_t.x;
                t_min = t_start * (1.0f - dist_and_t.y) + t_end * dist_and_t.y;
            }
            t_start = t_end;
            segment_start = segment_end;
        }

        float3 P = point_cubic_bezier(t_min, A, B, C, D);
        float dist = length(P - pos);
        float r = lerp(r1, r2, t_min);
        return make_float2(dist - r, t_min);
    }

    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 A = make_float3(params[0], params[1], params[2]);
        float3 B = make_float3(params[3], params[4], params[5]);
        float3 C = make_float3(params[6], params[7], params[8]);
        float3 D = make_float3(params[9], params[10], params[11]);
        return sdf_cubic_bezier(pos, A, B, C, D, params[12], params[13]).x;
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params) {
        float3 A = make_float3(params[0], params[1], params[2]);
        float3 B = make_float3(params[3], params[4], params[5]);
        float3 C = make_float3(params[6], params[7], params[8]);
        float3 D = make_float3(params[9], params[10], params[11]);
        float2 sdf_and_t = sdf_cubic_bezier(pos, A, B, C, D, params[12], params[13]);
        float t = sdf_and_t.y;
        float u = 1.0f - t;

        float3 P = point_cubic_bezier(t, A, B, C, D);
        float3 v = P - pos;
        float3 grad_v = grad_SDF * 2.0f * v;

        if (grad_params) {
            float3 grad_P = grad_v;
            float3 grad_A = grad_P * (u * u * u);
            float3 grad_B = grad_P * (3.0f * u * u * t);
            float3 grad_C = grad_P * (3.0f * u * t * t);
            float3 grad_D = grad_P * (t * t * t);
            float grad_r1 = -grad_SDF * u;
            float grad_r2 = -grad_SDF * t;

            atomicAdd3(grad_params + 0, grad_A);
            atomicAdd3(grad_params + 3, grad_B);
            atomicAdd3(grad_params + 6, grad_C);
            atomicAdd3(grad_params + 9, grad_D);
            atomicAdd(grad_params + 12, grad_r1);
            atomicAdd(grad_params + 13, grad_r2);
        }

        return -grad_v;
    }

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        return make_float2(0.0f, 0.0f);
    }
};

template <>
struct BaseSDF<CATMULL_ROM>
{
    static constexpr int num_samples = 10;

    __device__ static inline float3 point_catmull_rom(float t, float3 A, float3 B, float3 C, float3 D)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        return t3 * A + t2 * B + t * C + D;
    }

    __device__ static float2 sdf_catmull_rom(float3 pos, float3 p0, float3 p1, float3 p2, float3 p3, float r1, float r2)
    {
        float t01 = sqrt(length(p0 - p1) + 1e-8f);
        float t12 = sqrt(length(p1 - p2) + 1e-8f);
        float t23 = sqrt(length(p2 - p3) + 1e-8f);

        float3 m1 = p2 - p1 + t12 * ((p1 - p0) / t01 - (p2 - p0) / (t01 + t12));
        float3 m2 = p2 - p1 + t12 * ((p3 - p2) / t23 - (p3 - p1) / (t12 + t23));
        float3 A = 2.0f * (p1 - p2) + m1 + m2;
        float3 B = 3.0f * (p2 - p1) - 2.0f * m1 - m2;
        float3 C = m1;
        float3 D = p1;

        constexpr float step = 1.0f / num_samples;
        float dist_min = 1e9f;
        float t_min = 0.0f;
        float t_start = 0.0f;
        float3 segment_start = point_catmull_rom(t_start, A, B, C, D);
        for (int i = 1; i <= num_samples; i++) {
            float t_end = i * step;
            float3 segment_end = point_catmull_rom(t_end, A, B, C, D);
            float2 dist_and_t = point_segment_distance(pos, segment_start, segment_end);
            if (dist_and_t.x < dist_min) {
                dist_min = dist_and_t.x;
                t_min = t_start * (1.0f - dist_and_t.y) + t_end * dist_and_t.y;
            }
            t_start = t_end;
            segment_start = segment_end;
        }

        float3 P = point_catmull_rom(t_min, A, B, C, D);
        float dist = length(P - pos);
        float r = lerp(r1, r2, t_min);
        return make_float2(dist - r, t_min);
    }

    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 p0 = make_float3(params[0], params[1], params[2]);
        float3 p1 = make_float3(params[3], params[4], params[5]);
        float3 p2 = make_float3(params[6], params[7], params[8]);
        float3 p3 = make_float3(params[9], params[10], params[11]);
        return sdf_catmull_rom(pos, p0, p1, p2, p3, params[12], params[13]).x;
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float3 p0 = make_float3(params[0], params[1], params[2]);
        float3 p1 = make_float3(params[3], params[4], params[5]);
        float3 p2 = make_float3(params[6], params[7], params[8]);
        float3 p3 = make_float3(params[9], params[10], params[11]);
        float t = sdf_catmull_rom(pos, p0, p1, p2, p3, params[12], params[13]).y;

        float3 p0p1 = p0 - p1;
        float3 p1p2 = p1 - p2;
        float3 p2p3 = p2 - p3;
        float t01 = sqrt(length(p0p1) + 1e-8f);
        float t12 = sqrt(length(p1p2) + 1e-8f);
        float t23 = sqrt(length(p2p3) + 1e-8f);

        float3 m1 = p2 - p1 + t12 * ((p1 - p0) / t01 - (p2 - p0) / (t01 + t12));
        float3 m2 = p2 - p1 + t12 * ((p3 - p2) / t23 - (p3 - p1) / (t12 + t23));
        float3 A = 2.0f * (p1 - p2) + m1 + m2;
        float3 B = 3.0f * (p2 - p1) - 2.0f * m1 - m2;
        float3 C = m1;
        float3 D = p1;

        float3 P = point_catmull_rom(t, A, B, C, D);
        float3 v = P - pos;
        float3 grad_v = grad_SDF * 2.0f * v;

        if (grad_params) {
            float3 grad_P = grad_v;
            float3 grad_A = grad_P * (t * t * t);
            float3 grad_B = grad_P * (t * t);
            float3 grad_C = grad_P * t;
            float3 grad_D = grad_P;
            float3 grad_m1 = grad_A - grad_B * 2.0f + grad_C;
            float3 grad_m2 = grad_A - grad_B;

            float dm1_dp0 = t12 / (t01 + t12) - t12 / t01;
            float dm1_dp1 = -1.0f + t12 / t01;
            float dm1_dp2 = 1.0f - t12 / (t01 + t12);
            float dm2_dp1 = -1.0f + t12 / (t12 + t23);
            float dm2_dp2 = 1.0f - t12 / t23;
            float dm2_dp3 = t12 / t23 - t12 / (t12 + t23);
            float3 dm1_dt01 = t12 * ((p2 - p0) / ((t01 + t12) * (t01 + t12)) - (p1 - p0) / (t01 * t01));
            float3 dm1_dt12 = (p1 - p0) / t01 + t12 * (p2 - p0) / ((t01 + t12) * (t01 + t12)) - (p2 - p0) / (t01 + t12);
            float3 dm2_dt12 = (p2 - p1) / t23 + t12 * (p3 - p1) / ((t12 + t23) * (t12 + t23)) - (p3 - p1) / (t12 + t23);
            float3 dm2_dt23 = t12 * ((p2 - p3) / (t23 * t23) + (p3 - p1) / ((t12 + t23) * (t12 + t23)));
            float3 dt01_dp0p1 = p0p1 / (2.0f * pow(dot(p0p1, p0p1) + 1e-8f, 0.75f));
            float3 dt12_dp1p2 = p1p2 / (2.0f * pow(dot(p1p2, p1p2) + 1e-8f, 0.75f));
            float3 dt23_dp2p3 = p2p3 / (2.0f * pow(dot(p2p3, p2p3) + 1e-8f, 0.75f));

            float3 grad_p0 = grad_m1 * (dm1_dp0 + dm1_dt01 * dt01_dp0p1);
            float3 grad_p1 = grad_A * 2.0f + grad_B * -3.0f + grad_D + grad_m1 * (dm1_dp1 - dm1_dt01 * dt01_dp0p1 + dm1_dt12 * dt12_dp1p2) + grad_m2 * (dm2_dp1 + dm2_dt12 * dt12_dp1p2);
            float3 grad_p2 = grad_A * -2.0f + grad_B * 3.0f + grad_m1 * (dm1_dp2 - dm1_dt12 * dt12_dp1p2) + grad_m2 * (dm2_dp2 - dm2_dt12 * dt12_dp1p2 + dm2_dt23 * dt23_dp2p3);
            float3 grad_p3 = grad_m2 * (dm2_dp3 - dm2_dt23 * dt23_dp2p3);
            atomicAdd3(grad_params + 0, grad_p0);
            atomicAdd3(grad_params + 3, grad_p1);
            atomicAdd3(grad_params + 6, grad_p2);
            atomicAdd3(grad_params + 9, grad_p3);

            float grad_r1 = -grad_SDF * (1.0f - t);
            float grad_r2 = -grad_SDF * t;
            atomicAdd(grad_params + 12, grad_r1);
            atomicAdd(grad_params + 13, grad_r2);
        }

        return -grad_v;
    }

    template <bool enable_multiscale>
    __device__ static float2 texcoord(float3 pos, const float *params)
    {
        return make_float2(0.0f, 0.0f);
    }
};