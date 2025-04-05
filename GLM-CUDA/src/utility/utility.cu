#include "../../include/utility.cuh"

namespace GLM_CUDA
{
    __host__ __device__ GLM_CUDA::mat3 dot(GLM_CUDA::mat3 a, GLM_CUDA::mat3 b)
    {
        GLM_CUDA::mat3 out;
        for (int column = 0; column < 3; column++)
        {
            for (int row = 0; row < 3; row++)
            {
                out[column][row] = a[column][row] * b[column][row];
            }
        }
        return out;
    }
    __host__ __device__ GLM_CUDA::mat4 dot(GLM_CUDA::mat4 a, GLM_CUDA::mat4 b)
    {
        GLM_CUDA::mat4 out;
        for (int column = 0; column < 4; column++)
        {
            for (int row = 0; row < 4; row++)
            {
                out[column][row] = a[column][row] * b[column][row];
            }
        }
        return out;
    }
    __host__ __device__ float dot(GLM_CUDA::vec2 a, GLM_CUDA::vec2 b)
    {
        float sum = 0;
        for (int axis = 0; axis < 2; axis++)
        {
            sum += a[axis] * b[axis];
        }
        return sum;
    }
    __host__ __device__ float dot(GLM_CUDA::vec3 a, GLM_CUDA::vec3 b)
    {
        float sum = 0;
        for (int axis = 0; axis < 3; axis++)
        {
            sum += a[axis] * b[axis];
        }
        return sum;
    }
    __host__ __device__ float dot(GLM_CUDA::vec4 a, GLM_CUDA::vec4 b)
    {
        float sum = 0;
        for (int axis = 0; axis < 4; axis++)
        {
            sum += a[axis] * b[axis];
        }
        return sum;
    }
}