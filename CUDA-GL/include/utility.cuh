#pragma once

#include "matrix.cuh"
#include "vector.cuh"

namespace CUDA_GL
{
    // Common math function for vector and matrix
    // abs, floor, ceil

    // Graphic function
    // Projection (Host only)
    __host__ mat4 perspective(float fov, float aspect, float near, float far);
    __host__ mat4 ortho(float left, float right, float bottom, float top, float near, float far);
    
    // View (Host only)
    __host__ mat4 lookAt(vec3 position, vec3 target, vec3 up);

    // Model
    __host__ __device__ mat4 translate(vec3 offset);
    __host__ __device__ mat4 rotate(float angle, vec3 axis);
    __host__ __device__ mat4 scale(vec3 factor);
}