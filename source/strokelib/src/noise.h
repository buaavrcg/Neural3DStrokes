#ifndef NOISE_H
#define NOISE_H

#include "helper_math.h"


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
	p3  = fracf(p3 * .1031f);
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


#endif