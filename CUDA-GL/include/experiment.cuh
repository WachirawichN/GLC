#pragma once

#include "vector.cuh"
#include "matrix.cuh"

namespace CUDA_GL
{
    namespace experimental
    {
        __global__ void allOutGPUMatMul();
        __global__ void partialGPUMatMul();
        __global__ void GPUMatMul();
    }
}