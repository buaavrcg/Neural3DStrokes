#ifndef NOISE_H
#define NOISE_H

#include "helper_math.h"
#include <tuple>

////////////////////////////////////////////////////////////////////////////////
// Hash Functions
// Modified from https://www.shadertoy.com/view/4djSRW
////////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------------------
//  1 out, 1 in...
__device__ inline float hash11(float p)
{
    p = fracf(p * .1031f);
    p *= p + 33.33f;
    p *= p + p;
    return fracf(p);
}

//----------------------------------------------------------------------------------------
//  1 out, 3 in...
__device__ inline float hash13(float3 p3)
{
    p3 = fracf(p3 * .1031f);
    p3 += dot(p3, make_float3(p3.z, p3.y, p3.x) + 31.32f);
    return fracf((p3.x + p3.y) * p3.z);
}

//----------------------------------------------------------------------------------------
//  3 out, 1 in...
__device__ inline float3 hash31(float p)
{
    float3 p3 = fracf(make_float3(p) * make_float3(.1031f, .1030f, .0973f));
    p3 += dot(p3, make_float3(p3.y, p3.z, p3.x) + 33.33f);
    float3 q3 = make_float3(p3.x + p3.y, p3.x + p3.z, p3.y + p3.z);
    float3 s3 = make_float3(p3.z, p3.y, p3.x);
    return fracf(q3 * s3);
}

//----------------------------------------------------------------------------------------
///  3 out, 3 in...
__device__ inline float3 hash33(float3 p3)
{
    p3 = fracf(p3 * make_float3(.1031f, .1030f, .0973f));
    p3 += dot(p3, make_float3(p3.y, p3.z, p3.x) + 33.33f);
    float3 q3 = make_float3(p3.x + p3.y, p3.x + p3.x, p3.y + p3.x);
    float3 s3 = make_float3(p3.z, p3.y, p3.x);
    return fracf(q3 * s3);
}

////////////////////////////////////////////////////////////////////////////////
// Noise Functions
////////////////////////////////////////////////////////////////////////////////

/// 1D noise
/// @param x noise sample position
/// @return noise value in range [-1, 1]
__device__ inline float noise11(float x)
{
    float i = floorf(x);
    float f = fracf(x);
    f = f * f * (3.f - 2.f * f);
    float v = lerp(hash11(i), hash11(i + 1.f), f);
    return v * 2.0f - 1.0f;
}

/// 3D noise
/// @param x noise sample position
/// @return noise value in range [-1, 1]
__device__ inline float noise13(float3 x)
{
    float3 i = floorf(x);
    float3 f = fracf(x);
    f = f * f * (3.f - 2.f * f);
    float v = lerp(lerp(lerp(hash13(i + make_float3(0.f, 0.f, 0.f)),
                             hash13(i + make_float3(1.f, 0.f, 0.f)), f.x),
                        lerp(hash13(i + make_float3(0.f, 1.f, 0.f)),
                             hash13(i + make_float3(1.f, 1.f, 0.f)), f.x),
                        f.y),
                   lerp(lerp(hash13(i + make_float3(0.f, 0.f, 1.f)),
                             hash13(i + make_float3(1.f, 0.f, 1.f)), f.x),
                        lerp(hash13(i + make_float3(0.f, 1.f, 1.f)),
                             hash13(i + make_float3(1.f, 1.f, 1.f)), f.x),
                        f.y),
                   f.z);
    return v * 2.0f - 1.0f;
}

////////////////////////////////////////////////////////////////////////////////
// FBM Noise Functions
////////////////////////////////////////////////////////////////////////////////

/// 1D fbm noise
template <int NumOctaves>
__device__ float fbm11(float x, const float G = 0.5f)
{
    constexpr float s = 2.001f;
    float a = 1.0f;
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < NumOctaves; i++)
    {
        t += a * noise11(x);
        a *= G;
        x *= s;
    }
    return t;
}

/// 3D fbm noise
template <int NumOctaves>
__device__ float fbm13(float3 x, const float G = 0.5f)
{
    const mat3 m = mat3(+0.00f, +0.80f, +0.60f,
                        -0.80f, +0.36f, -0.48f,
                        -0.60f, -0.48f, +0.64f);
    constexpr float s = 2.001f;
    float a = 1.0f;
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < NumOctaves; i++)
    {
        t += a * noise13(x);
        a *= G;
        x = m * x * s;
    }
    return t;
}

#endif