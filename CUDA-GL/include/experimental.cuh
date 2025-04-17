#pragma once

#include "vector.cuh"
#include "matrix.cuh"
#include "utility.cuh"

namespace CUDA_GL
{
    namespace experimental
    {
        __global__ void allOutGPUMatMul();
        __global__ void partialGPUMatMul(mat4* a, mat4* b, mat4* c);
        __global__ void gpuMatMul(mat4* a, mat4* b, mat4* c);
    }
}