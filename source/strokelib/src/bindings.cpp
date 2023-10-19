#include <torch/extension.h>

#include "strokes.h"
#include "compositing.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stroke_forward", &stroke_forward, "stroke_forward (CUDA)");
    m.def("stroke_backward", &stroke_backward, "stroke_backward (CUDA)");

    m.def("compose_forward", &compose_forward, "compose_forward (CUDA)");
    m.def("compose_backward", &compose_backward, "compose_backward (CUDA)");
}