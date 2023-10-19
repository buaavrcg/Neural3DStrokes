#include <cstdint>
#include <utility>
#include <array>
#include <math.h>
#include "common.h"
#include "helper_math.h"
#include "compositing.h"

template <int color_dim>
__global__ void compose_forward_kernel(float *__restrict__ density_output,
                                       float *__restrict__ color_output,
                                       const float *__restrict__ alphas,
                                       const float *__restrict__ colors,
                                       const float *__restrict__ density_params,
                                       const int64_t n_points,
                                       const int64_t n_strokes)
{
    const uint32_t idx_point = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_point >= n_points)
        return;

    alphas += idx_point * n_strokes;
    colors += idx_point * n_strokes * color_dim;

    // Initialize T, density and color
    float T = 1.0f;
    float density = 0.0f;
    float color[color_dim];
#pragma unroll
    for (int i = 0; i < color_dim; ++i)
        color[i] = 0.0f;

    // Compute accumulated density and color
    for (int idx_stroke = n_strokes - 1; idx_stroke >= 0; --idx_stroke)
    {
        const float alpha = alphas[idx_stroke];
        if (alpha == 0.0f) // skip zero alpha for speedup
            continue;

        const float weight = alpha * T;
        T *= (1.0f - alpha);
        density += density_params[idx_stroke] * weight;
#pragma unroll
        for (int i = 0; i < color_dim; ++i)
            color[i] += colors[idx_stroke * color_dim + i] * weight;
    }

    // Compute final color
    float final_color_scale = 1.0f / (1.0f + 1e-6f - T); // 1 / (1 - T_n)
#pragma unroll
    for (int i = 0; i < color_dim; ++i)
        color[i] = clamp(color[i] * final_color_scale, 0.0f, 1.0f);

    // Store density and color
    density_output[idx_point] = density;
#pragma unroll
    for (int i = 0; i < color_dim; ++i)
        color_output[idx_point * color_dim + i] = color[i];
}

template <int color_dim>
__global__ void compose_backward_kernel(float *__restrict__ grad_alphas,
                                        float *__restrict__ grad_colors,
                                        float *__restrict__ grad_density_params,
                                        const float *__restrict__ grad_density_output,
                                        const float *__restrict__ grad_color_output,
                                        const float *__restrict__ alphas,
                                        const float *__restrict__ colors,
                                        const float *__restrict__ density_params,
                                        const int64_t n_points,
                                        const int64_t n_strokes)
{
    const uint32_t idx_point = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_point >= n_points)
        return;

    alphas += idx_point * n_strokes;
    colors += idx_point * n_strokes * color_dim;
    grad_alphas += idx_point * n_strokes;
    grad_colors += idx_point * n_strokes * color_dim;

    // Recompute density and color outputs
    float T = 1.0f;
    float density = 0.0f;
    float color[color_dim];
#pragma unroll
    for (int i = 0; i < color_dim; ++i)
        color[i] = 0.0f;
    for (int idx_stroke = n_strokes - 1; idx_stroke >= 0; --idx_stroke)
    {
        const float alpha = alphas[idx_stroke];
        if (alpha == 0.0f) // skip zero alpha for speedup
            continue;

        const float weight = alpha * T;
        T *= (1.0f - alpha);
        density += density_params[idx_stroke] * weight;
#pragma unroll
        for (int i = 0; i < color_dim; ++i)
            color[i] += colors[idx_stroke * color_dim + i] * weight;
    }

    // Load gradients
    float dL_ddensity = grad_density_output[idx_point];
    float dL_dcolor[color_dim];
    float final_opacity = 1.0f + 1e-6f - T;         // (1 - T_n)
    float final_color_scale = 1.0f / final_opacity; // 1 / (1 - T_n)
#pragma unroll
    for (int i = 0; i < color_dim; ++i)
    {
        float scaled_color = color[i] * final_color_scale;
        bool in_range = 0.0f <= scaled_color && scaled_color <= 1.0f;
        dL_dcolor[i] = in_range ? grad_color_output[idx_point * color_dim + i] : 0.0f;
    }

    // Compute accumulated density and color
    T = 1.0f;
    float density2 = 0.0f;
    float color2[color_dim];
#pragma unroll
    for (int i = 0; i < color_dim; ++i)
        color2[i] = 0.0f;
    for (int idx_stroke = n_strokes - 1; idx_stroke >= 0; --idx_stroke)
    {
        const float alpha = alphas[idx_stroke];

        // Calculate gradients for density_params and colors
        if (alpha > 0.0f)
        {
            const float weight = alpha * T;
            T *= (1.0f - alpha);
            density2 += density_params[idx_stroke] * weight;
#pragma unroll
            for (int i = 0; i < color_dim; ++i)
                color2[i] += colors[idx_stroke * color_dim + i] * weight;

            atomicAdd(grad_density_params + idx_stroke, dL_ddensity * weight);
#pragma unroll
            for (int i = 0; i < color_dim; ++i)
                atomicAdd(grad_colors + idx_stroke * color_dim + i, dL_dcolor[i] * final_color_scale * weight);
        }

        // Calculate gradients for alphas
        float scale_dT_dalpha = 1.0f / max(1.0f - alpha, 1e-6f);
        float density_suffix = density - density2;
        float dL_dalpha = dL_ddensity * (T * density_params[idx_stroke] - density_suffix) * scale_dT_dalpha;

        scale_dT_dalpha *= final_color_scale * final_color_scale;
#pragma unroll
        for (int i = 0; i < color_dim; ++i)
        {
            float color_suffix = color[i] - color2[i];
            dL_dalpha += dL_dcolor[i] * (T * colors[idx_stroke * color_dim + i] * (final_opacity - alpha) - color_suffix) * scale_dT_dalpha;
        }
        atomicAdd(grad_alphas + idx_stroke, dL_dalpha);
    }
}

void compose_forward(at::Tensor density_output,
                     at::Tensor color_output,
                     const at::Tensor alphas,
                     const at::Tensor colors,
                     const at::Tensor density_params)
{
    CHECK_FLOAT_INPUT(density_output);
    CHECK_FLOAT_INPUT(color_output);
    CHECK_FLOAT_INPUT(alphas);
    CHECK_FLOAT_INPUT(colors);
    CHECK_FLOAT_INPUT(density_params);

    const int64_t n_points = alphas.size(0);
    const int64_t n_strokes = alphas.size(1);
    const int64_t color_dim = colors.size(2);

    constexpr int64_t n_threads = 512;
    const int64_t n_blocks = div_round_up(n_points, n_threads);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    switch (color_dim)
    {
    case 1:
        compose_forward_kernel<1><<<n_blocks, n_threads, 0, stream>>>(
            density_output.data_ptr<float>(),
            color_output.data_ptr<float>(),
            alphas.data_ptr<float>(),
            colors.data_ptr<float>(),
            density_params.data_ptr<float>(),
            n_points,
            n_strokes);
        break;
    case 3:
        compose_forward_kernel<3><<<n_blocks, n_threads, 0, stream>>>(
            density_output.data_ptr<float>(),
            color_output.data_ptr<float>(),
            alphas.data_ptr<float>(),
            colors.data_ptr<float>(),
            density_params.data_ptr<float>(),
            n_points,
            n_strokes);
        break;
    default:
        throw std::runtime_error("Unsupported color dimension: " + std::to_string(color_dim));
    }
}

void compose_backward(at::Tensor grad_alphas,
                      at::Tensor grad_colors,
                      at::Tensor grad_density_params,
                      const at::Tensor grad_density_output,
                      const at::Tensor grad_color_output,
                      const at::Tensor alphas,
                      const at::Tensor colors,
                      const at::Tensor density_params)
{
    CHECK_FLOAT_INPUT(grad_alphas);
    CHECK_FLOAT_INPUT(grad_colors);
    CHECK_FLOAT_INPUT(grad_density_params);
    CHECK_FLOAT_INPUT(grad_density_output);
    CHECK_FLOAT_INPUT(grad_color_output);
    CHECK_FLOAT_INPUT(alphas);
    CHECK_FLOAT_INPUT(colors);
    CHECK_FLOAT_INPUT(density_params);

    const int64_t n_points = alphas.size(0);
    const int64_t n_strokes = alphas.size(1);
    const int64_t color_dim = colors.size(2);

    constexpr int64_t n_threads = 512;
    const int64_t n_blocks = div_round_up(n_points, n_threads);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    switch (color_dim)
    {
    case 1:
        compose_backward_kernel<1><<<n_blocks, n_threads, 0, stream>>>(
            grad_alphas.data_ptr<float>(),
            grad_colors.data_ptr<float>(),
            grad_density_params.data_ptr<float>(),
            grad_density_output.data_ptr<float>(),
            grad_color_output.data_ptr<float>(),
            alphas.data_ptr<float>(),
            colors.data_ptr<float>(),
            density_params.data_ptr<float>(),
            n_points,
            n_strokes);
        break;
    case 3:
        compose_backward_kernel<3><<<n_blocks, n_threads, 0, stream>>>(
            grad_alphas.data_ptr<float>(),
            grad_colors.data_ptr<float>(),
            grad_density_params.data_ptr<float>(),
            grad_density_output.data_ptr<float>(),
            grad_color_output.data_ptr<float>(),
            alphas.data_ptr<float>(),
            colors.data_ptr<float>(),
            density_params.data_ptr<float>(),
            n_points,
            n_strokes);
        break;
    default:
        throw std::runtime_error("Unsupported color dimension: " + std::to_string(color_dim));
    }
}