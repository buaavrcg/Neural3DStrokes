#include <torch/extension.h>

#include "strokes.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stroke_forward", &stroke_forward, "stroke_forward (CUDA)");
    m.def("stroke_backward", &stroke_backward, "stroke_backward (CUDA)");
}