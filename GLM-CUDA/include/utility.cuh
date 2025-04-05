#pragma once

#include "matrix.cuh"
#include "vector.cuh"

namespace GLM_CUDA
{
    __host__ __device__ mat3 dot(mat3 a, mat3 b);
    __host__ __device__ mat4 dot(mat4 a, mat4 b);
    __host__ __device__ float dot(vec2 a, vec2 b);
    __host__ __device__ float dot(vec3 a, vec3 b);
    __host__ __device__ float dot(vec4 a, vec4 b);
}