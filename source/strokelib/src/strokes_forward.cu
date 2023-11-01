#include "strokes.h"
#include "strokes_sdf.h"
#include "strokes_color.h"
#include <array>
#include <utility>

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
                                      const float *__restrict__ viewdir,
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

    alpha_output += idx_thread;
    color_output += idx_thread * ColorField<color_type>::color_dim;
    shape_params += idx_stroke * n_shape_params;
    color_params += idx_stroke * n_color_params;

    float3 pos = ((const float3 *)x)[idx_point];
    float3 dir = ColorField<color_type>::use_viewdir ? ((const float3 *)viewdir)[idx_point / n_samples_per_ray]
                                                     : make_float3(0.0f);
    if constexpr (!ColorField<color_type>::use_unit_pos)
    {
        // Compute the color output with raw pos and color parameters
        ColorField<color_type>::get_color(color_output, pos, dir, color_params, idx_stroke);
    }

    // Apply inverse shape transformation if required
    const float *sp_reverse = shape_params + n_shape_params;
    pos = inverse_transform<enable_translation,
                            enable_rotation,
                            enable_singlescale,
                            enable_multiscale>(pos, sp_reverse);

    // Query unit space SDF value with shape parameters
    float sdf_value = BaseSDF<sdf_type>::sdf(pos, shape_params);
    if (sdf_output)
        sdf_output[idx_thread] = sdf_value;

    // Transform the SDF to compute the blending weight alpha
    const float sdf_scale = (use_laplace_transform ? 2.0f : 0.5f) / sdf_delta;
    float alpha = sdf_delta > 0.0f ? (use_laplace_transform
                                          ? laplace_cdf(-sdf_value * sdf_scale)
                                          : clamp(-sdf_value * sdf_scale + 0.5f, 0.0f, 0.9999f))
                                   : float(sdf_value <= 0.0f);
    *alpha_output = alpha;

    if constexpr (ColorField<color_type>::use_unit_pos)
    {
        if constexpr (ColorField<color_type>::use_viewdir)
        {
            // Transform the viewdir to unit space
            const float *sp_reverse = shape_params + n_shape_params;
            dir = inverse_transform_direction<enable_rotation, enable_multiscale>(dir, sp_reverse);
        }

        // Compute the color output with unit pos and color parameters
        ColorField<color_type>::get_color(color_output, pos, dir, color_params, idx_stroke);
    }
}

template <uint32_t id>
void stroke_forward_warpper(float *alpha_output,
                            float *color_output,
                            float *sdf_output,
                            const float *x,
                            const float *viewdir,
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
    constexpr int64_t n_threads = 512;
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
            viewdir,
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

DECLARE_INT_TEMPLATE_ARG_LUT(stroke_forward_warpper)
void stroke_forward(at::Tensor alpha_output,
                    at::Tensor color_output,
                    at::Tensor sdf_output,
                    const at::Tensor x,
                    const at::Tensor viewdir,
                    const at::Tensor shape_params,
                    const at::Tensor color_params,
                    const uint32_t sdf_id,
                    const uint32_t color_id,
                    const float sdf_delta,
                    const bool use_laplace_transform)
{
    CHECK_FLOAT_INPUT(alpha_output);
    CHECK_FLOAT_INPUT(color_output);
    CHECK_FLOAT_INPUT(sdf_output);
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
    static const auto fn_table = MAKE_INT_TEMPLATE_ARG_LUT(stroke_forward_warpper, num_fn_ids);

    fn_table[fn_id](
        alpha_output.data_ptr<float>(),
        color_output.data_ptr<float>(),
        sdf_output.numel() ? sdf_output.data_ptr<float>() : nullptr,
        x.data_ptr<float>(),
        viewdir.data_ptr<float>(),
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
