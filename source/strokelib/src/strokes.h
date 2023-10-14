#pragma once

#include <cstdint>
#include <torch/torch.h>

// alpha_output: [N_Points, N_Strokes], float
// color_output: [N_Points, N_Strokes, D_Color], float
// sdf_output: [N_Points, N_Strokes], float
// x: [N_Points, 3], float
// shape_params: [N_Strokes, N_ShapeParams], float
// color_params: [N_Strokes, N_ColorParams], float
void stroke_forward(at::Tensor alpha_output,
                    at::Tensor color_output,
                    at::Tensor sdf_output,
                    const at::Tensor x,
                    const at::Tensor shape_params,
                    const at::Tensor color_params,
                    const uint32_t sdf_id,
                    const uint32_t color_id,
                    const float sdf_delta,
                    const bool use_sigmoid_clamping);

// grad_shape_params: [N_Strokes, N_ShapeParams], float
// grad_color_params: [N_Strokes, N_ColorParams], float
// grad_x: [N_Points, 3], float
// grad_alpha: [N_Points, N_Strokes], float
// grad_color: [N_Points, N_Strokes, D_Color], float
// grad_sdf: [N_Points, N_Strokes], float
// x: [N_Points, 3], float
// shape_params: [N_Strokes, N_ShapeParams], float
// color_params: [N_Strokes, N_ColorParams], float
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
                     const bool use_sigmoid_clamping);
