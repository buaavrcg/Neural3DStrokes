#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float32 tensor")
#define CHECK_IS_SAME_TYPE(x, y) TORCH_CHECK(x.scalar_type() == y.scalar_type(), #x " and " #y " must have the same type")
#define CHECK_FLOATING_INPUT(x) \
    CHECK_CUDA(x);           \
    CHECK_CONTIGUOUS(x);     \
    CHECK_IS_FLOATING(x)
#define CHECK_FLOAT_INPUT(x) \
    CHECK_CUDA(x);           \
    CHECK_CONTIGUOUS(x);     \
    CHECK_IS_FLOAT(x)

#define DECLARE_INT_TEMPLATE_ARG_LUT(fname)                        \
    template <size_t... N>                                         \
    static constexpr auto fname##_lut(std::index_sequence<N...> s) \
    {                                                              \
        return std::array{(&fname<N>)...};                         \
    }

#define MAKE_INT_TEMPLATE_ARG_LUT(fname, N) \
    fname##_lut(std::make_index_sequence<N>{})

template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor)
{
    return (val + divisor - 1) / divisor;
}

template <typename T>
__host__ __device__ inline T sigmoid(const T v)
{
    return T(1) / (T(1) + exp(-v));
}

template <typename T>
__host__ __device__ inline T laplace_cdf(const T s)
{
    return s <= T(0) ? T(0.5) * exp(s) : T(1) - T(0.5) * exp(-s);
}

__device__ inline void atomicAdd3(float *address, float3 val)
{
    atomicAdd(address + 0, val.x);
    atomicAdd(address + 1, val.y);
    atomicAdd(address + 2, val.z);
}
