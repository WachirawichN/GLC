#pragma once

#include <cmath>

#include "matrix.cuh"
#include "vector.cuh"

namespace CUDA_GL
{
    // Common math function for vector and matrix
    // abs, floor, ceil
    __host__ __device__ vec2 pow(const vec2& vector, float exponent);
    __host__ __device__ vec3 pow(const vec3& vector, float exponent);
    __host__ __device__ vec4 pow(const vec4& vector, float exponent);

    // Graphic function
    // Projection (Host only)
    __host__ mat4 perspective(float fov, float aspect, float near, float far);
    __host__ mat4 ortho(float left, float right, float bottom, float top, float near, float far);
    
    // View (Host only)
    __host__ mat4 lookAt(const vec3& position, const vec3& target, const vec3& up);

    // Model
    __host__ __device__ mat4 translate(const vec3& offset);
    __host__ __device__ mat4 rotate(float angle, const vec3& axis);
    __host__ __device__ mat4 scale(const vec3& factor);
}