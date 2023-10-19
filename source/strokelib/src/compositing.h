#pragma once

#include <cstdint>
#include <torch/torch.h>

// density_output: [N_Points], float
// color_output: [N_Points, D_Color], float
// alphas: [N_Points, N_Strokes], float
// colors: [N_Points, N_Strokes, D_Color], float
// density_params: [N_Strokes], float
void compose_forward(at::Tensor density_output,
                     at::Tensor color_output,
                     const at::Tensor alphas,
                     const at::Tensor colors,
                     const at::Tensor density_params);

// grad_alphas: [N_Points, N_Strokes], float
// grad_colors: [N_Points, N_Strokes, D_Color], float
// grad_density_params: [N_Strokes], float
// grad_density_output: [N_Points], float
// grad_color_output: [N_Points, D_Color], float
// alphas: [N_Points, N_Strokes], float
// colors: [N_Points, N_Strokes, D_Color], float
// density_params: [N_Strokes], float
void compose_backward(at::Tensor grad_alphas,
                      at::Tensor grad_colors,
                      at::Tensor grad_density_params,
                      const at::Tensor grad_density_output,
                      const at::Tensor grad_color_output,
                      const at::Tensor alphas,
                      const at::Tensor colors,
                      const at::Tensor density_params);
