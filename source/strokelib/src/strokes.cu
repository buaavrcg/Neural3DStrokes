#include <cstdint>
#include <utility>
#include <array>
#include <math.h>
#include "helper_math.h"
#include "common.h"

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
    UNIT_BEZIER = 8,
    NB_BASE_SDFS,
};

enum ColorType
{
    CONSTANT_RGB = 0,
    GRADIENT_RGB = 1,
    NB_COLORS,
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

/////////////////////////////////////////////////////////////////////
// Base Signed Distance Fields
/////////////////////////////////////////////////////////////////////

template <BaseSDFType sdf_type>
struct BaseSDF
{
    // Returns SDF value for the given pos and shape params.
    __device__ static float sdf(float3 pos, const float *params);
    // Returns dSDF/dPos and stores grad_params for given pos and shape params.
    // Note: use atomic operation on grad_params
    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params);
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
};

template <>
struct BaseSDF<UNIT_CUBE>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 p_abs = fabs(pos);
        return max(max(p_abs.x, p_abs.y), p_abs.z) - 1.f;
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float3 p_abs = fabs(pos);
        float grad_x = p_abs.x >= p_abs.y && p_abs.x >= p_abs.z ? (pos.x < 0.0f ? -1.0f : 1.0f) : 0.0f;
        float grad_y = p_abs.x < p_abs.y && p_abs.y >= p_abs.z ? (pos.y < 0.0f ? -1.0f : 1.0f) : 0.0f;
        float grad_z = p_abs.x < p_abs.z && p_abs.y < p_abs.z ? (pos.z < 0.0f ? -1.0f : 1.0f) : 0.0f;
        return make_float3(grad_x, grad_y, grad_z);
    }
};

template <>
struct BaseSDF<UNIT_ROUND_CUBE>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 p_abs = fabs(pos);
        float3 p_dis= p_abs - make_float3(1.0f, 1.0f, 1.0f);
        float3 p_dis_positive=fmaxf(p_dis, make_float3(0.0f, 0.0f, 0.0f));
        float3 p_dis_square=p_dis_positive*p_dis_positive;
        float p_dis_norm = sqrt(p_dis_square.x+p_dis_square.y+p_dis_square.z+1e-8f);
        float p_dis_min = min(max(p_dis.x, max(p_dis.y, p_dis.z)), 0.0f);
        return p_dis_norm + p_dis_min-params[0];
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        atomicAdd(grad_params + 0, -grad_SDF*1.0f);

        float3 p_abs = fabs(pos);
        float3 p_dis = p_abs - make_float3(1.0f, 1.0f, 1.0f); 
        float3 p_dis_positive=fmaxf(p_dis, make_float3(0.0f, 0.0f, 0.0f));
        float3 p_dis_square=p_dis_positive*p_dis_positive;
        float p_dis_norm = sqrt(p_dis_square.x+p_dis_square.y+p_dis_square.z+1e-8f);
        float p_dis_norm_reciprocal = rsqrt(p_dis_square.x+p_dis_square.y+p_dis_square.z+1e-8) ;
        float3 grad_p_dis_positive = p_dis_norm > 0.0f ? p_dis_positive *p_dis_norm_reciprocal : make_float3(0.0f, 0.0f, 0.0f);
        float grad_p_x_sym = (pos.x < 0.0f ? -1.0f : 1.0f);
        float grad_p_y_sym = (pos.y < 0.0f ? -1.0f : 1.0f);
        float grad_p_z_sym = (pos.z < 0.0f ? -1.0f : 1.0f);
        float3 grad_p_dis_sym = make_float3(grad_p_x_sym, grad_p_y_sym, grad_p_z_sym);
        float3 grad_p_dis = grad_p_dis_positive*grad_p_dis_sym;

        float p_dis_min = min(max(p_dis.x, max(p_dis.y, p_dis.z)), 0.0f);
        float grad_p_dis_min = p_dis_min < 0.0f ? 1.0f : 0.0f;
        float grad_p_dis_min_x = grad_p_dis_min * (p_dis.x>=p_dis.y&&p_dis.x>=p_dis.z ? 1.0f : 0.0f)*grad_p_dis_sym.x;
        float grad_p_dis_min_y = grad_p_dis_min * (p_dis.x<p_dis.y&&p_dis.y>=p_dis.z ? 1.0f : 0.0f)*grad_p_dis_sym.y;
        float grad_p_dis_min_z = grad_p_dis_min * (p_dis.x<p_dis.z&&p_dis.y<p_dis.z ? 1.0f : 0.0f)*grad_p_dis_sym.z;

        float grad_x = grad_p_dis.x + grad_p_dis_min_x;
        float grad_y = grad_p_dis.y + grad_p_dis_min_y;
        float grad_z = grad_p_dis.z + grad_p_dis_min_z;

        return make_float3(grad_x, grad_y, grad_z);
    }
};

template <>
struct BaseSDF<UNIT_CAPPED_TORUS>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float3 p_abs = make_float3(fabs(pos.x), pos.y, pos.z);
        float2 p_xy= make_float2(p_abs.x, p_abs.y);
        float2 sc= make_float2(sinf(params[0]), cosf(params[0]));
        float k = cross(make_float3(p_xy, 0.0f), make_float3(sc, 0.0f)).z>0?dot(p_xy, sc):sqrt(p_xy.x*p_xy.x+p_xy.y*p_xy.y+1e-8f);
        return sqrtf(dot(p_abs, p_abs) + 1.f - 2.f * k+1e-8f) - params[1];
    }

    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float3 p_abs = make_float3(fabs(pos.x), pos.y, pos.z);
        float2 p_xy = make_float2(p_abs.x, p_abs.y);
        float2 sc = make_float2(sinf(params[0]), cosf(params[0]));
        float2 sc_f= make_float2(cosf(params[0]), sinf(params[0]));
        float k = cross(make_float3(p_xy, 0.0f), make_float3(sc, 0.0f)).z > 0 ? dot(p_xy, sc) : length(p_xy);
        float inv_norm = rsqrt(dot(p_abs, p_abs) + 1.f - 2.f * k+1e-8f);

        float3 grad_p_dot = pos;
        float grad_x=0.0f;
        float grad_y=0.0f;
        float grad_z=0.0f;

        if(k>0.f){
            float grad_p_k_x = -(pos.x < 0.0f ? -1.0f : 1.0f)*sc.x;
            float grad_p_k_y = -sc.y;
            float grad_p_k_z = 0.0f;
            float3 grad_p_k = make_float3(grad_p_k_x, grad_p_k_y, grad_p_k_z);
            atomicAdd(grad_params + 0, -grad_SDF*dot(p_xy, sc_f));
            grad_x=(grad_p_dot.x+grad_p_k.x)*inv_norm;
            grad_y=(grad_p_dot.y+grad_p_k.y)*inv_norm;
            grad_z=(grad_p_dot.z+grad_p_k.z)*inv_norm;

        }
        else {
            float grad_p_k_x = pos.x;
            float grad_p_k_y = pos.y;
            float grad_p_k_z = 0.0f;
            float3 grad_p_k = make_float3(grad_p_k_x, grad_p_k_y, grad_p_k_z);
            grad_x=(grad_p_dot.x+grad_p_k.x)*inv_norm;
            grad_y=(grad_p_dot.y+grad_p_k.y)*inv_norm;
            grad_z=(grad_p_dot.z+grad_p_k.z)*inv_norm;
        }

        atomicAdd(grad_params + 1, -grad_SDF);
        return make_float3(grad_x, grad_y, grad_z);


    }
};

template <>
struct BaseSDF<UNIT_CAPSULE>
{
    __device__ static float sdf(float3 pos, const float *params)
    {
        float p_y= pos.y-fminf(fmaxf(pos.y, 0.0f), params[0]);
        float3 p_xyz= make_float3(pos.x, p_y, pos.z);
        float3 p_sq = p_xyz * p_xyz;
        return sqrt(p_sq.x+p_sq.y+p_sq.z+1e-8)-1.f;
    }
    
    __device__ static float3 grad_sdf(float *grad_params, float grad_SDF, float3 pos, const float *params)
    {
        float p_y= pos.y-fminf(fmaxf(pos.y, 0.0f), params[0]);
        float3 p_xyz= make_float3(pos.x, p_y, pos.z);
        float3 p_sq = p_xyz * p_xyz;
        float3 grad_p_xyz= p_xyz*rsqrt(p_sq.x+p_sq.y+p_sq.z+1e-8);
        if(pos.y>params[0]){
            atomicAdd(grad_params+0, -grad_SDF*p_y*rsqrt(p_sq.x+p_sq.y+p_sq.z+1e-8));
        }
        return make_float3(grad_p_xyz.x, grad_p_xyz.y, grad_p_xyz.z);
    }

};

template <> 
struct BaseSDF<UNIT_LINE>
{
    __device__ static float sdf(float3 pos,const float *params){
        float t=pos.y/params[0];
        //t = fminf(fmaxf(t, 0.0f), 1.0f);
        t = clamp(t, 0.0f, 1.0f);
        float p_y= pos.y-fminf(fmaxf(pos.y, 0.0f), params[0]);
        float r=(1+params[1])*t+(1-params[1])*(1-t);
        return sqrtf(pos.x*pos.x+p_y*p_y+pos.z*pos.z+1e-8f)-r;
    }

    __device__ static float3 grad_sdf(float *grad_params,float grad_SDF,float3 pos,const float *params){
        float t=pos.y/params[0];
        t = fminf(fmaxf(t, 0.0f), 1.0f);
        float p_y= pos.y-fminf(fmaxf(pos.y, 0.0f), params[0]);
        float r=(1+params[1])*t+(1-params[1])*(1-t);
        float inv_norm = rsqrt(pos.x*pos.x+p_y*p_y+pos.z*pos.z+1e-8f);
        float grad_t_y=t>0.0f&&t<1.0f?1.0f/params[0]:0.0f;
        float3 grad_t_xyz=make_float3(0.0f,grad_t_y,0.0f);
        float grad_r_t=2*params[1];
        float3 grad_r=grad_r_t*grad_t_xyz;
        float grad_p_y=p_y==0?0.0f:1.0f;
        float3 grad_p_xyz=make_float3(pos.x,grad_p_y*p_y,pos.z)*inv_norm;
        float3 grad_xyz=grad_p_xyz-grad_r;

        float grad_t_p_0=t>0.0f&&t<1.0f?-1.0f/(params[0]*params[0]):0.0f;
        float grad_r_p_0=grad_r_t*grad_t_p_0;
        float grad_y_p_0=pos.y>params[0]?-1.0f:0.0f;
        float grad_dis_p_0=grad_y_p_0*p_y*inv_norm;
        atomicAdd(grad_params+0, -grad_SDF*grad_dis_p_0-grad_r_p_0);

        float grad_r_p_1=2*t-1.0f;
        float grad_dis_p_1=0.0f;
        atomicAdd(grad_params+1, -grad_SDF*(grad_dis_p_1-grad_r_p_1));

        return grad_xyz;
    }
};

template <>
struct BaseSDF<UNIT_TRIPRISM>
{
    __device__ static float sdf(float3 pos,const float *params){
        float3 q=fabs(pos);
        return fmaxf(q.y-params[0],fmaxf(q.x*0.866025f+pos.z*0.5f,-pos.z)-0.5f);
    }

    __device__ static float3 grad_sdf(float *grad_params,float grad_SDF,float3 pos,const float *params){
        float3 q=fabs(pos);
        float3 grad_q=make_float3(pos.x<0.0f?-1.0f:1.0f,pos.y<0.0f?-1.0f:1.0f,pos.z<0.0f?-1.0f:1.0f);
        float grad_x=0.0f;
        float grad_y=0.0f;
        float grad_z=0.0f;
        if(q.x*0.866025f+pos.z*0.5f>-pos.z){
            if(q.y-params[0]>q.x*0.866025f+pos.z*0.5f-0.5f){
                grad_y=grad_q.y;
                atomicAdd(grad_params+0, -grad_SDF);
            }
            else{
                grad_x=0.866025f*grad_q.x;
                grad_z=0.5f;
            }
        }
        else{
            if(q.y-params[0]>-pos.z-0.5f){
                grad_y=grad_q.y;
                atomicAdd(grad_params+0, -grad_SDF);
            }
            else{
                grad_z=-1.0f;
            }
        }
        return make_float3(grad_x,grad_y,grad_z);
    }
};

template <>
struct BaseSDF<UNIT_OCTAHEDRON>{
    __device__ static float sdf(float3 pos,const float *params){
        float3 p_abs=fabs(pos);
        return (p_abs.x+p_abs.y+p_abs.z-1.0f)*0.57735027f;
    }

    __device__ static float3 grad_sdf(float *grad_params,float grad_SDF,float3 pos,const float *params){
        float3 p_abs=fabs(pos);
        float grad_x=pos.x<0.0f?-0.57735027f:0.57735027f;
        float grad_y=pos.y<0.0f?-0.57735027f:0.57735027f;
        float grad_z=pos.z<0.0f?-0.57735027f:0.57735027f;
        return make_float3(grad_x,grad_y,grad_z);
    }
};

template <>
struct BaseSDF<UNIT_BEZIER>{
    __device__ static float sdf_bezier(float3 pos, float3 A, float3 B, float3 C, float r1, float r2) {
        float2 res=make_float2(0.0f,0.0f);

        float3 a=B-A;
        float3 b=A-2.0f*B+C;
        float3 c=a*2.0f;
        float3 d=A-pos;

        float kk=1.0f/dot(b,b);
        float kx=kk*dot(a,b);
        float ky=kk*(2.0f*dot(a,a)+dot(d,b))/3.0f;
        float kz=kk*dot(d,a);

        float p=ky-kx*kx;
        float p3=p*p*p;
        float q=kx*(2.0f*kx*kx-3.0f*ky)+kz;
        float h=q*q+4.0f*p3;

        if(h>=0.0f){
            h=sqrtf(h+1e-8f);
            float2 x=make_float2((h-q)*0.5f,-(h+q)*0.5f);
            float2 uv = make_float2(
                copysignf(powf(fabs(x.x),1.0f/3.0f),x.x),
                copysignf(powf(fabs(x.y),1.0f/3.0f),x.y)
            );
            float t=clamp((uv.x+uv.y)-kx,0.0,1.0);

            res=make_float2(dot(d+(c+b*t)*t,d+(c+b*t)*t),t);
        }
        else{
            float z=sqrtf(-p+1e-8f);
            float v=acosf(q/(p*z*2.0f+1e-8f))/3.0f;
            float m=cosf(v);
            float n=sinf(v)*1.732050808f;
            float3 t = clamp(make_float3(m+n,-n-m,n-m)*z-kx,0.0f,1.0f);

            float3 dis3 = d+(c+b*t.x)*t.x;
            float dis = dot(dis3,dis3);
            res = make_float2(dis,t.x);

            dis3 = d+(c+b*t.y)*t.y;
            dis = dot(dis3,dis3);
            res = dis<res.x?make_float2(dis,t.y):res;
        }

        res.x=sqrtf(res.x+1e-8f);
        float r = lerp(r1, r2, res.y);
        return res.x - r;
    }


    __device__ static float sdf(float3 pos,const float *params){
        float3 A=make_float3(params[0],params[1],params[2]);
        float3 B=make_float3(params[3],params[4],params[5]);
        float3 C=make_float3(params[6],params[7],params[8]);
        return sdf_bezier(pos, A, B, C, params[9], params[10]);
    }

    // __device__ static float3 other_grad_sdf(float *grad_params,float grad_SDF,float3 pos,const float *params){
    //     float3 A=make_float3(params[0],params[1],params[2]);
    //     float3 B=make_float3(params[3],params[4],params[5]);
    //     float3 C=make_float3(params[6],params[7],params[8]);
    //     float2 res=make_float2(0.0f,0.0f);

    //     float3 a=B-A;
    //     float3 b=A-2.0f*B+C;
    //     float3 c=a*2.0f;
    //     float3 d=A-pos;

    //     float kk=1.0f/dot(b,b);
    //     float kx=kk*dot(a,b);
    //     float ky=kk*(2.0f*dot(a,a)+dot(d,b))/3.0f;
    //     float kz=kk*dot(d,a);

    //     float p=ky-kx*kx;
    //     float p3=p*p*p;
    //     float q=kx*(2.0f*kx*kx-3.0f*ky)+kz;
    //     float h=q*q+4.0f*p3;

    //     if(h>=0.0f){
    //         h=sqrtf(h+1e-8f);
    //         float2 x=make_float2((h-q)*0.5f,-(h+q)*0.5f);
    //         float2 uv = make_float2(
    //             copysignf(powf(fabs(x.x),1.0f/3.0f),x.x),
    //             copysignf(powf(fabs(x.y),1.0f/3.0f),x.y)
    //         );
    //         float t=clamp((uv.x+uv.y)-kx,0.0,1.0);

    //         res=make_float2(dot(d+(c+b*t)*t,d+(c+b*t)*t),t);

    //         float ansx=res.x;
    //         float ansy=res.y;
    //         float grad_d_xyz=-1.0f;
    //         if(t==0.0f||t==1.0f){
    //             float3 grad_ansx_xyz=grad_d_xyz*2.0f*d;
    //             float3 grad_ansy_xyz=make_float3(0.0f,0.0f,0.0f);
    //         }
    //         else {
    //             float3 m=d+(c+b*t)*t;
    //             float3 grad_ansx_m=2.0f*m;
    //             float grad_m_d=1.0f;
    //             float3 grad_m_t=c+2.0f*b*t;
    //             float grad_d_xyz=-1.0f;
    //             float grad_uvx_h=copysignf(powf(x.x,-2.0f/3.0f),x.x)/3.0f;
    //             float grad_uvy_h=-1.0f*copysignf(powf(x.y,-2.0f/3.0f),x.y)/3.0f;
    //             float grad_h_q=2.0f*q;
    //             float grad_uvx_q=-1.0f*grad_uvx_h;
    //             float grad_uvy_q=-1.0f*grad_uvy_h;
    //             float grad_q_xyz=kk*a*(-1.0f);
    //             float grad_ansx_xyz=grad_ansx_m*(grad_m_d+grad_m_t*grad_)
    //         }
    //     }
    //     else{
    //         float z=sqrtf(-p);
    //         float v=acosf(q/(p*z*2.0f+1e-8f))/3.0f;
    //         float m=cosf(v);
    //         float n=sinf(v)*1.732050808f;
    //         float3 t = clamp(make_float3(m+n,-m-n,n-m)*z-kx,0.0f,1.0f);

    //         float dis=dot(d+(c+b*t.x)*t.x,d+(c+b*t.x)*t.x);
    //         res = make_float2(dis,t.x);

    //         dis=dot(d+(c+b*t.y)*t.y,d+(c+b*t.y)*t.y);

    //         if(dis<res.x){
    //             res = make_float2(dis,t.y);
    //             float ansx=res.x;
    //             float ansy=res.y;
    //             float grad_d_xyz=-1.0f;
    //             if(t.y==0.0f||t.y==1.0f){
    //                 float3 grad_ansx_xyz=grad_d_xyz*2.0f*d;
    //                 float3 grad_ansy_xyz=make_float3(0.0f,0.0f,0.0f);
    //             }
    //             else {
    //                 float3 m=d+(c+b*t.y)*t.y;
    //                 float3 grad_ansx_m=2.0f*m;
    //                 float grad_m_d=1.0f;
    //                 float3 grad_m_t=c+2.0f*b*t.y;
    //                 float grad_d_xyz=-1.0f;
    //                 float grad_ty_n=-1.0f*z;
    //                 float grad_ty_m=-1.0f*z;
    //                 float grad_n_v=cosf(v)*1.732050808f;
    //                 float grad_m_v=-sinf(v);
    //                 float grad_v_q=1.0f/(p*z*2.0f+1e-8f)*(-1.0f)*rsqrtf(1.0f-q*q/(p*p*z*z*4.0f)+1e-8f)/3.0f;
    //                 float grad_q_xyz=kk*a*(-1.0f);
    //             }
    //         }
    //         else {
    //             float ansx=res.x;
    //             float ansy=res.y;
    //             float grad_d_xyz=-1.0f;
    //             if(t.x==0.0f||t.x==1.0f){
    //                 float3 grad_ansx_xyz=grad_d_xyz*2.0f*d;
    //                 float3 grad_ansy_xyz=make_float3(0.0f,0.0f,0.0f);
    //             }
    //             else {
    //                 float3 m=d+(c+b*t.x)*t.x;
    //                 float3 grad_ansx_m=2.0f*m;
    //                 float grad_m_d=1.0f;
    //                 float3 grad_m_t=c+2.0f*b*t.x;
    //                 float grad_d_xyz=-1.0f;
    //                 float grad_tx_n=1.0f*z;
    //                 float grad_tx_m=1.0f*z;
    //                 float grad_n_v=cosf(v)*1.732050808f;
    //                 float grad_m_v=-sinf(v);
    //                 float grad_v_q=1.0f/(p*z*2.0f+1e-8f)*(-1.0f)*rsqrtf(1.0f-q*q/(p*p*z*z*4.0f)+1e-8f)/3.0f;
    //                 float grad_q_xyz=kk*a*(-1.0f);
    //             }
    //         }

    //     }

    // }

    template <int i, int j>
    __device__ static float get_param(const float *params, float delta) {
        return params[j] + (j == i ? delta : 0.0f);
    }

    template <int i>
    __device__ static float sdf_delta(float3 pos,const float *params, float delta){
        float3 A=make_float3(get_param<i, 0>(params, delta),get_param<i, 1>(params, delta),get_param<i, 2>(params, delta));
        float3 B=make_float3(get_param<i, 3>(params, delta),get_param<i, 4>(params, delta),get_param<i, 5>(params, delta));
        float3 C=make_float3(get_param<i, 6>(params, delta),get_param<i, 7>(params, delta),get_param<i, 8>(params, delta));
        return sdf_bezier(pos, A, B, C, get_param<i, 9>(params, delta), get_param<i, 10>(params, delta));
    }

    template <int i>
    __device__ static float grad_param_i(float3 pos, const float *params, float delta) {
        float sdf_positive = sdf_delta<i>(pos, params, +delta);
        float sdf_negative = sdf_delta<i>(pos, params, -delta);
        float grad_params_i = (sdf_positive - sdf_negative) / (2.0f * delta);
        return grad_params_i;
    }
    
    __device__ static float3 grad_sdf(float *grad_params,float grad_SDF,float3 pos,const float *params){
        float delta=1e-4f;

        atomicAdd(grad_params+0, grad_SDF*grad_param_i<0>(pos,params,delta));
        atomicAdd(grad_params+1, grad_SDF*grad_param_i<1>(pos,params,delta));
        atomicAdd(grad_params+2, grad_SDF*grad_param_i<2>(pos,params,delta));
        atomicAdd(grad_params+3, grad_SDF*grad_param_i<3>(pos,params,delta));
        atomicAdd(grad_params+4, grad_SDF*grad_param_i<4>(pos,params,delta));
        atomicAdd(grad_params+5, grad_SDF*grad_param_i<5>(pos,params,delta));
        atomicAdd(grad_params+6, grad_SDF*grad_param_i<6>(pos,params,delta));
        atomicAdd(grad_params+7, grad_SDF*grad_param_i<7>(pos,params,delta));
        atomicAdd(grad_params+8, grad_SDF*grad_param_i<8>(pos,params,delta));
        atomicAdd(grad_params+9, grad_SDF*grad_param_i<9>(pos,params,delta));
        atomicAdd(grad_params+10, grad_SDF*grad_param_i<10>(pos,params,delta));

        float grad_x=(sdf(pos+make_float3(delta,0.0f,0.0f),params)-sdf(pos-make_float3(delta,0.0f,0.0f),params))/(2.0f*delta);
        float grad_y=(sdf(pos+make_float3(0.0f,delta,0.0f),params)-sdf(pos-make_float3(0.0f,delta,0.0f),params))/(2.0f*delta);
        float grad_z=(sdf(pos+make_float3(0.0f,0.0f,delta),params)-sdf(pos-make_float3(0.0f,0.0f,delta),params))/(2.0f*delta);
        return make_float3(grad_x,grad_y,grad_z);

    }
};
/////////////////////////////////////////////////////////////////////
// Color Fields
/////////////////////////////////////////////////////////////////////

template <ColorType color_type>
struct ColorField
{
    static constexpr int color_dim = 3;
    // Stores color value into color_out for the given pos and shape params.
    __device__ static void get_color(float *color_out, float3 pos, const float *params);
    // Stores grad_params for given pos and color params.
    // Note: use atomic operation on grad_params
    __device__ static void grad_color(float *grad_params, const float *grad_color, float3 pos, const float *params);
};

template <>
struct ColorField<CONSTANT_RGB>
{
    static constexpr int color_dim = 3;
    __device__ static void get_color(float *color_out, float3 pos, const float *params)
    {
        *(float3 *)color_out = *(float3 *)params;
    }
    __device__ static void grad_color(float *grad_params, const float *grad_color, float3 pos, const float *params)
    {
        atomicAdd3(grad_params, *(float3 *)grad_color);
    }
};

template <>
struct ColorField<GRADIENT_RGB>
{
    static constexpr int color_dim = 3;
    __device__ static void get_color(float *color_out, float3 pos, const float *params)
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
    __device__ static void grad_color(float *grad_params, const float *grad_color, float3 pos, const float *params)
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

/////////////////////////////////////////////////////////////////////
// Forward
/////////////////////////////////////////////////////////////////////

template <BaseSDFType sdf_type,
          ColorType color_type,
          bool enable_translation,
          bool enable_rotation,
          bool enable_singlescale,
          bool enable_multiscale>
__global__ void stroke_forward_kernel(float *__restrict__ alpha_output,
                                      float *__restrict__ color_output,
                                      float *__restrict__ sdf_output,
                                      const float *__restrict__ x,
                                      const float *__restrict__ shape_params,
                                      const float *__restrict__ color_params,
                                      const int64_t n_points,
                                      const int64_t n_strokes,
                                      const int64_t n_shape_params,
                                      const int64_t n_color_params,
                                      const float sdf_delta,
                                      const bool use_sigmoid_clamping)
{
    const uint32_t idx_thread = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t idx_point = idx_thread / n_strokes;
    const uint32_t idx_stroke = idx_thread % n_strokes;
    if (idx_point >= n_points)
        return;

    alpha_output += idx_thread;
    color_output += idx_thread * ColorField<color_type>::color_dim;
    if (sdf_output)
        sdf_output += idx_thread;
    x += idx_point * 3;
    shape_params += idx_stroke * n_shape_params;
    color_params += idx_stroke * n_color_params;

    // Apply inverse shape transformation if required
    float3 pos = *(const float3 *)x;
    const float *sp_reverse = shape_params + n_shape_params;
    pos = inverse_transform<enable_translation,
                            enable_rotation,
                            enable_singlescale,
                            enable_multiscale>(pos, sp_reverse);

    // Query unit space SDF value with shape parameters
    float sdf_value = BaseSDF<sdf_type>::sdf(pos, shape_params);
    if (sdf_output)
        *sdf_output = sdf_value;

    // Soft clamping the SDF to compute the blending weight alpha
    const float sdf_scale = (use_sigmoid_clamping ? 4.0f : 0.5f) / sdf_delta;
    float alpha = sdf_delta > 0.0f ? (use_sigmoid_clamping
                                          ? sigmoid(-sdf_value * sdf_scale)
                                          : clamp(-sdf_value * sdf_scale + 0.5f, 0.0f, 0.9999f))
                                   : float(sdf_value <= 0.0f);
    *alpha_output = alpha;

    // Compute the color output with color parameters
    ColorField<color_type>::get_color(color_output, pos, color_params);
}

template <uint32_t id>
void stroke_forward_warpper(float *alpha_output,
                            float *color_output,
                            float *sdf_output,
                            const float *x,
                            const float *shape_params,
                            const float *color_params,
                            const int64_t n_points,
                            const int64_t n_strokes,
                            const int64_t n_shape_params,
                            const int64_t n_color_params,
                            const float sdf_delta,
                            const bool use_sigmoid_clamping)
{
    constexpr uint32_t sdf_id = id / NB_COLORS;
    constexpr uint32_t color_id = id % NB_COLORS;
    constexpr ColorType color_type = ColorType(color_id);
    constexpr BaseSDFType base_sdf_type = BaseSDFType(sdf_id >> 4);
    constexpr bool enable_translation = (sdf_id & 0b0001) != 0;
    constexpr bool enable_rotation = (sdf_id & 0b0010) != 0;
    constexpr bool enable_singlescale = (sdf_id & 0b0100) != 0;
    constexpr bool enable_multiscale = (sdf_id & 0b1000) != 0;
    constexpr int64_t n_threads = 1024;
    const int64_t n_blocks = div_round_up(n_points * n_strokes, n_threads);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    stroke_forward_kernel<
        base_sdf_type,
        color_type,
        enable_translation,
        enable_rotation,
        enable_singlescale,
        enable_multiscale>
        <<<n_blocks, n_threads, 0, stream>>>(
            alpha_output,
            color_output,
            sdf_output,
            x,
            shape_params,
            color_params,
            n_points,
            n_strokes,
            n_shape_params,
            n_color_params,
            sdf_delta,
            use_sigmoid_clamping);
}

DECLARE_INT_TEMPLATE_ARG_LUT(stroke_forward_warpper)
void stroke_forward(at::Tensor alpha_output,
                    at::Tensor color_output,
                    at::Tensor sdf_output,
                    const at::Tensor x,
                    const at::Tensor shape_params,
                    const at::Tensor color_params,
                    const uint32_t sdf_id,
                    const uint32_t color_id,
                    const float sdf_delta,
                    const bool use_sigmoid_clamping)
{
    CHECK_FLOAT_INPUT(alpha_output);
    CHECK_FLOAT_INPUT(color_output);
    CHECK_FLOAT_INPUT(sdf_output);
    CHECK_FLOAT_INPUT(x);
    CHECK_FLOAT_INPUT(shape_params);
    CHECK_FLOAT_INPUT(color_params);

    const int64_t n_points = x.size(0);
    const int64_t n_strokes = shape_params.size(0);
    const int64_t n_shape_params = shape_params.size(1);
    const int64_t n_color_params = color_params.size(1);

    constexpr uint32_t num_fn_ids = NB_BASE_SDFS * 16 * NB_COLORS;
    const uint32_t fn_id = sdf_id * NB_COLORS + color_id;
    TORCH_CHECK(fn_id < num_fn_ids, "fn_id must be in [0, num_fn_ids]")
    static const auto fn_table = MAKE_INT_TEMPLATE_ARG_LUT(stroke_forward_warpper, num_fn_ids);

    fn_table[fn_id](
        alpha_output.data_ptr<float>(),
        color_output.data_ptr<float>(),
        sdf_output.numel() ? sdf_output.data_ptr<float>() : nullptr,
        x.data_ptr<float>(),
        shape_params.data_ptr<float>(),
        color_params.data_ptr<float>(),
        n_points,
        n_strokes,
        n_shape_params,
        n_color_params,
        sdf_delta,
        use_sigmoid_clamping);
}

/////////////////////////////////////////////////////////////////////
// Backward
/////////////////////////////////////////////////////////////////////

template <BaseSDFType sdf_type,
          ColorType color_type,
          bool enable_translation,
          bool enable_rotation,
          bool enable_singlescale,
          bool enable_multiscale>
__global__ void stroke_backward_kernel(float *__restrict__ grad_shape_params,
                                       float *__restrict__ grad_color_params,
                                       float *__restrict__ grad_x,
                                       const float *__restrict__ grad_alpha,
                                       const float *__restrict__ grad_color,
                                       const float *__restrict__ grad_sdf,
                                       const float *__restrict__ x,
                                       const float *__restrict__ alpha,
                                       const float *__restrict__ shape_params,
                                       const float *__restrict__ color_params,
                                       const int64_t n_points,
                                       const int64_t n_strokes,
                                       const int64_t n_shape_params,
                                       const int64_t n_color_params,
                                       const float sdf_delta,
                                       const bool use_sigmoid_clamping)
{
    const uint32_t idx_thread = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t idx_point = idx_thread / n_strokes;
    const uint32_t idx_stroke = idx_thread % n_strokes;
    if (idx_point >= n_points)
        return;

    grad_shape_params += idx_stroke * n_shape_params;
    grad_color_params += idx_stroke * n_color_params;
    if (grad_x)
        grad_x += idx_point * 3;
    grad_alpha += idx_thread;
    grad_color += idx_thread * ColorField<color_type>::color_dim;
    if (grad_sdf)
        grad_sdf += idx_thread;
    x += idx_point * 3;
    shape_params += idx_stroke * n_shape_params;
    color_params += idx_stroke * n_color_params;

    // Apply inverse shape transformation if required
    float3 pos = *(const float3 *)x;
    const float *sp_reverse = shape_params + n_shape_params;
    pos = inverse_transform<enable_translation,
                            enable_rotation,
                            enable_singlescale,
                            enable_multiscale>(pos, sp_reverse);

    // Compute dL/dSDF from dL/dAlpha
    const float sdf_scale = (use_sigmoid_clamping ? 4.0f : 0.5f) / sdf_delta;
    float dAlpha_dSDF = 0.0f;
    if (sdf_delta > 0.0f)
    {
        float alpha_val = *(alpha + idx_thread);
        if (use_sigmoid_clamping)
            dAlpha_dSDF = alpha_val * (1.0f - alpha_val) * -sdf_scale;
        else
            dAlpha_dSDF = 0.0f < alpha_val && alpha_val < 0.9999f ? -sdf_scale : 0.0f;
    }
    float dL_dSDF = *grad_alpha * dAlpha_dSDF + (grad_sdf ? *grad_sdf : 0.0f);

    // Compute dL/dColorParams from dL/dColor
    ColorField<color_type>::grad_color(grad_color_params, grad_color, pos, color_params);

    // Compute dL/dShapeParams and dL/dPos from dL/dSDF
    float3 dSDF_dPos = BaseSDF<sdf_type>::grad_sdf(grad_shape_params, dL_dSDF, pos, shape_params);
    float3 dL_dPos = dL_dSDF * dSDF_dPos;

    // Compute dL/dShapeParams from dL/dPos, by forward transformation
    float *grad_sp_reverse = grad_shape_params + (sp_reverse - shape_params);
    if constexpr (enable_singlescale)
    {
        float scale = *sp_reverse;
        dL_dPos /= scale;
        float3 dL_dScale = -dL_dPos * pos;
        pos *= scale;
        atomicAdd(grad_sp_reverse, dL_dScale.x + dL_dScale.y + dL_dScale.z);
        sp_reverse += 1;
        grad_sp_reverse += 1;
    }
    else if constexpr (enable_multiscale)
    {
        float3 scale = *(const float3 *)sp_reverse;
        dL_dPos /= scale;
        float3 dL_dScale = -dL_dPos * pos;
        pos *= scale;
        atomicAdd3(grad_sp_reverse, dL_dScale);
        sp_reverse += 3;
        grad_sp_reverse += 3;
    }
    if constexpr (enable_rotation)
    {
        float3 eular_angle = *(const float3 *)sp_reverse;
        pos = rotate_point<false>(pos, eular_angle);
        float3 dL_dAngle = grad_angle_rotate_point(dL_dPos, pos, eular_angle);
        dL_dPos = grad_point_rotate_point(dL_dPos, pos, eular_angle);
        atomicAdd3(grad_sp_reverse, dL_dAngle);
        sp_reverse += 3;
        grad_sp_reverse += 3;
    }
    if constexpr (enable_translation)
    {
        float3 translation = *(const float3 *)sp_reverse;
        pos += translation;
        float3 dL_dTranslation = -dL_dPos;
        atomicAdd3(grad_sp_reverse, dL_dTranslation);
        sp_reverse += 3;
        grad_sp_reverse += 3;
    }

    if (grad_x)
        atomicAdd3(grad_x, dL_dPos);
}

template <uint32_t id>
void stroke_backward_warpper(float *grad_shape_params,
                             float *grad_color_params,
                             float *grad_x,
                             const float *grad_alpha,
                             const float *grad_color,
                             const float *grad_sdf,
                             const float *x,
                             const float *alpha,
                             const float *shape_params,
                             const float *color_params,
                             const int64_t n_points,
                             const int64_t n_strokes,
                             const int64_t n_shape_params,
                             const int64_t n_color_params,
                             const float sdf_delta,
                             const bool use_sigmoid_clamping)
{
    constexpr uint32_t sdf_id = id / NB_COLORS;
    constexpr uint32_t color_id = id % NB_COLORS;
    constexpr ColorType color_type = ColorType(color_id);
    constexpr BaseSDFType base_sdf_type = BaseSDFType(sdf_id >> 4);
    constexpr bool enable_translation = (sdf_id & 0b0001) != 0;
    constexpr bool enable_rotation = (sdf_id & 0b0010) != 0;
    constexpr bool enable_singlescale = (sdf_id & 0b0100) != 0;
    constexpr bool enable_multiscale = (sdf_id & 0b1000) != 0;
    constexpr int64_t n_threads = 1024;
    const int64_t n_blocks = div_round_up(n_points * n_strokes, n_threads);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    stroke_backward_kernel<
        base_sdf_type,
        color_type,
        enable_translation,
        enable_rotation,
        enable_singlescale,
        enable_multiscale>
        <<<n_blocks, n_threads, 0, stream>>>(
            grad_shape_params,
            grad_color_params,
            grad_x,
            grad_alpha,
            grad_color,
            grad_sdf,
            x,
            alpha,
            shape_params,
            color_params,
            n_points,
            n_strokes,
            n_shape_params,
            n_color_params,
            sdf_delta,
            use_sigmoid_clamping);
}

DECLARE_INT_TEMPLATE_ARG_LUT(stroke_backward_warpper)
void stroke_backward(at::Tensor grad_shape_params,
                     at::Tensor grad_color_params,
                     at::Tensor grad_x,
                     const at::Tensor grad_alpha,
                     const at::Tensor grad_color,
                     const at::Tensor grad_sdf,
                     const at::Tensor x,
                     const at::Tensor alpha,
                     const at::Tensor shape_params,
                     const at::Tensor color_params,
                     const uint32_t sdf_id,
                     const uint32_t color_id,
                     const float sdf_delta,
                     const bool use_sigmoid_clamping)
{
    CHECK_FLOAT_INPUT(grad_shape_params);
    CHECK_FLOAT_INPUT(grad_color_params);
    CHECK_FLOAT_INPUT(grad_x);
    CHECK_FLOAT_INPUT(grad_alpha);
    CHECK_FLOAT_INPUT(grad_color);
    CHECK_FLOAT_INPUT(grad_sdf);
    CHECK_FLOAT_INPUT(x);
    CHECK_FLOAT_INPUT(shape_params);
    CHECK_FLOAT_INPUT(color_params);

    const int64_t n_points = x.size(0);
    const int64_t n_strokes = shape_params.size(0);
    const int64_t n_shape_params = shape_params.size(1);
    const int64_t n_color_params = color_params.size(1);

    constexpr uint32_t num_fn_ids = NB_BASE_SDFS * 16 * NB_COLORS;
    const uint32_t fn_id = sdf_id * NB_COLORS + color_id;
    TORCH_CHECK(fn_id < num_fn_ids, "fn_id must be in [0, num_fn_ids]")
    static const auto fn_table = MAKE_INT_TEMPLATE_ARG_LUT(stroke_backward_warpper, num_fn_ids);

    fn_table[fn_id](
        grad_shape_params.data_ptr<float>(),
        grad_color_params.data_ptr<float>(),
        grad_x.numel() ? grad_x.data_ptr<float>() : nullptr,
        grad_alpha.data_ptr<float>(),
        grad_color.data_ptr<float>(),
        grad_sdf.numel() ? grad_sdf.data_ptr<float>() : nullptr,
        x.data_ptr<float>(),
        alpha.data_ptr<float>(),
        shape_params.data_ptr<float>(),
        color_params.data_ptr<float>(),
        n_points,
        n_strokes,
        n_shape_params,
        n_color_params,
        sdf_delta,
        use_sigmoid_clamping);
}