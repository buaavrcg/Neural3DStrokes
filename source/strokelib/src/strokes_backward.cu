#include "strokes.h"
#include "strokes_sdf.h"
#include "strokes_color.h"
#include <array>
#include <utility>

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
                                       const float *__restrict__ viewdir,
                                       const float *__restrict__ alpha,
                                       const float *__restrict__ shape_params,
                                       const float *__restrict__ color_params,
                                       const int64_t n_points,
                                       const int64_t n_strokes,
                                       const int64_t n_samples_per_ray,
                                       const int64_t n_shape_params,
                                       const int64_t n_color_params,
                                       const float sdf_delta,
                                       const bool use_laplace_transform)
{
    const uint32_t idx_thread = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t idx_point = idx_thread / n_strokes;
    const uint32_t idx_stroke = idx_thread % n_strokes;
    if (idx_point >= n_points)
        return;

    grad_shape_params += idx_stroke * n_shape_params;
    grad_color_params += idx_stroke * n_color_params;
    grad_color += idx_thread * ColorField<color_type>::color_dim;
    shape_params += idx_stroke * n_shape_params;
    color_params += idx_stroke * n_color_params;

    float3 pos = ((const float3 *)x)[idx_point];
    float3 dir = ColorField<color_type>::use_viewdir ? ((const float3 *)viewdir)[idx_point / n_samples_per_ray]
                                                     : make_float3(0.0f);
    if constexpr (!ColorField<color_type>::use_unit_pos)
    {
        // Compute dL/dColorParams from dL/dColor with raw pos
        ColorField<color_type>::grad_color(grad_color_params, grad_color, pos, dir, color_params, idx_stroke);
    }

    // Apply inverse shape transformation if required
    const float *sp_reverse = shape_params + n_shape_params;
    pos = inverse_transform<enable_translation,
                            enable_rotation,
                            enable_singlescale,
                            enable_multiscale>(pos, sp_reverse);

    // Compute dL/dSDF from dL/dAlpha
    const float sdf_scale = (use_laplace_transform ? 2.0f : 0.5f) / sdf_delta;
    float dAlpha_dSDF = 0.0f;
    if (sdf_delta > 0.0f)
    {
        float alpha_val = *(alpha + idx_thread);
        if (use_laplace_transform)
            dAlpha_dSDF = (alpha_val < 0.5f ? alpha_val : 1.0f - alpha_val) * -sdf_scale;
        else
            dAlpha_dSDF = 0.0f < alpha_val && alpha_val < 0.9999f ? -sdf_scale : 0.0f;
    }
    float dL_dSDF = grad_alpha[idx_thread] * dAlpha_dSDF + (grad_sdf ? grad_sdf[idx_thread] : 0.0f);

    if constexpr (ColorField<color_type>::use_unit_pos)
    {
        if constexpr (ColorField<color_type>::use_viewdir)
        {
            // Transform the viewdir to unit space
            const float *sp_reverse = shape_params + n_shape_params;
            dir = inverse_transform_direction<enable_rotation, enable_multiscale>(dir, sp_reverse);
        }

        // Compute dL/dColorParams from dL/dColor with raw pos
        ColorField<color_type>::grad_color(grad_color_params, grad_color, pos, dir, color_params, idx_stroke);
    }

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
        atomicAdd3(grad_x + idx_point * 3, dL_dPos);
}

template <uint32_t id>
void stroke_backward_warpper(float *grad_shape_params,
                             float *grad_color_params,
                             float *grad_x,
                             const float *grad_alpha,
                             const float *grad_color,
                             const float *grad_sdf,
                             const float *x,
                             const float *viewdir,
                             const float *alpha,
                             const float *shape_params,
                             const float *color_params,
                             const int64_t n_points,
                             const int64_t n_strokes,
                             const int64_t n_samples_per_ray,
                             const int64_t n_shape_params,
                             const int64_t n_color_params,
                             const float sdf_delta,
                             const bool use_laplace_transform)
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
            viewdir,
            alpha,
            shape_params,
            color_params,
            n_points,
            n_strokes,
            n_samples_per_ray,
            n_shape_params,
            n_color_params,
            sdf_delta,
            use_laplace_transform);
}

DECLARE_INT_TEMPLATE_ARG_LUT(stroke_backward_warpper)
void stroke_backward(at::Tensor grad_shape_params,
                     at::Tensor grad_color_params,
                     at::Tensor grad_x,
                     const at::Tensor grad_alpha,
                     const at::Tensor grad_color,
                     const at::Tensor grad_sdf,
                     const at::Tensor x,
                     const at::Tensor viewdir,
                     const at::Tensor alpha,
                     const at::Tensor shape_params,
                     const at::Tensor color_params,
                     const uint32_t sdf_id,
                     const uint32_t color_id,
                     const float sdf_delta,
                     const bool use_laplace_transform)
{
    CHECK_FLOAT_INPUT(grad_shape_params);
    CHECK_FLOAT_INPUT(grad_color_params);
    CHECK_FLOAT_INPUT(grad_x);
    CHECK_FLOAT_INPUT(grad_alpha);
    CHECK_FLOAT_INPUT(grad_color);
    CHECK_FLOAT_INPUT(grad_sdf);
    CHECK_FLOAT_INPUT(x);
    CHECK_FLOAT_INPUT(viewdir);
    CHECK_FLOAT_INPUT(shape_params);
    CHECK_FLOAT_INPUT(color_params);

    const int64_t n_points = x.size(0);
    const int64_t n_viewdirs = viewdir.size(0);
    const int64_t n_samples_per_ray = n_points / n_viewdirs;
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
        viewdir.data_ptr<float>(),
        alpha.data_ptr<float>(),
        shape_params.data_ptr<float>(),
        color_params.data_ptr<float>(),
        n_points,
        n_strokes,
        n_samples_per_ray,
        n_shape_params,
        n_color_params,
        sdf_delta,
        use_laplace_transform);
}
