#include "../../include/utility.cuh"

namespace GLM_CUDA
{
    __host__ __device__ mat2 dot(const mat2& a, const mat2& b)
    {
        mat2 out;
        for (int column = 0; column < 2; column++)
        {
            for (int row = 0; row < 2; row++)
            {
                out[column][row] = a[column][row] * b[column][row];
            }
        }
        return out;
    }
    __host__ __device__ mat3 dot(const mat3& a, const mat3& b)
    {
        mat3 out;
        for (int column = 0; column < 3; column++)
        {
            for (int row = 0; row < 3; row++)
            {
                out[column][row] = a[column][row] * b[column][row];
            }
        }
        return out;
    }
    __host__ __device__ mat4 dot(const mat4& a, const mat4& b)
    {
        mat4 out;
        for (int column = 0; column < 4; column++)
        {
            for (int row = 0; row < 4; row++)
            {
                out[column][row] = a[column][row] * b[column][row];
            }
        }
        return out;
    }
    __host__ __device__ float dot(const vec2& a, const vec2& b)
    {
        float sum = 0;
        for (int axis = 0; axis < 2; axis++)
        {
            sum += a[axis] * b[axis];
        }
        return sum;
    }
    __host__ __device__ float dot(const vec3& a, const vec3& b)
    {
        float sum = 0;
        for (int axis = 0; axis < 3; axis++)
        {
            sum += a[axis] * b[axis];
        }
        return sum;
    }
    __host__ __device__ float dot(const vec4& a, const vec4& b)
    {
        float sum = 0;
        for (int axis = 0; axis < 4; axis++)
        {
            sum += a[axis] * b[axis];
        }
        return sum;
    }
}