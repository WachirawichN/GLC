#include "../../include/utility.cuh"

namespace CUDA_GL
{
    // Projection (Host only)
    __host__ mat4 perspective(float fov, float aspect, float near, float far)
    {
        return mat4(1.0f);
    }
    __host__ mat4 ortho(float left, float right, float bottom, float top, float near, float far)
    {
        return mat4(1.0f);
    }
    
    // View (Host only)
    __host__ mat4 lookAt(const vec3& position, const vec3& target, const vec3& up)
    {
        return mat4(1.0f);
    }

    // Model
    __host__ __device__ mat4 translate(const vec3& offset)
    {
        mat4 out(1.0f);
        out[3][0] = offset[0];
        out[3][1] = offset[1];
        out[3][2] = offset[2];
        return out;
    }
    __host__ __device__ mat4 rotate(float angle, const vec3& axis)
    {
        #ifdef __CUDA_ARCH__
            float c = cosf(angle);
            float s = sinf(angle);
        #else
            float c = std::cosf(angle);
            float s = std::sinf(angle);
        #endif
        float oneMinusCos = 1.0f - c;
        vec3 normAxis = normalize(axis);
        vec3 squareAxis = pow(axis, 2.0f);

        mat4 out(1.0f);
        // Column 0
        out[0][0] = c + squareAxis[0] * oneMinusCos;
        out[0][1] = axis[1] * axis[0] * oneMinusCos + axis[2] * s;
        out[0][2] = axis[2] * axis[0] * oneMinusCos - axis[1] * s;
        // Column 1
        out[1][0] = axis[0] * axis[1] * oneMinusCos - axis[2] * s;
        out[1][1] = c + squareAxis[1] * oneMinusCos;
        out[1][2] = axis[2] * axis[1] * oneMinusCos + axis[0] * s;
        // Column 2
        out[2][0] = axis[0] * axis[2] * oneMinusCos + axis[1] * s;
        out[2][1] = axis[1] * axis[2] * oneMinusCos - axis[0] * s;
        out[2][2] = c + squareAxis[2] * oneMinusCos;

        return out;
    }
    __host__ __device__ mat4 scale(const vec3& factor)
    {
        mat4 out(1.0f);
        out[0][0] = factor[0];
        out[1][1] = factor[1];
        out[2][2] = factor[2];
        return out;
    }
}