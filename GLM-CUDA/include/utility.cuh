#pragma once

#include "matrix.cuh"
#include "vector.cuh"

namespace GLM_CUDA
{
    __host__ __device__ mat2 dot(const mat2& a, const mat2& b);
    __host__ __device__ mat3 dot(const mat3& a, const mat3& b);
    __host__ __device__ mat4 dot(const mat4& a, const mat4& b);
    __host__ __device__ float dot(const vec2& a, const vec2& b);
    __host__ __device__ float dot(const vec3& a, const vec3& b);
    __host__ __device__ float dot(const vec4& a, const vec4& b);
}