#include "../../include/matrix.cuh"

#include "../../include/utility.cuh"

namespace GLM_CUDA
{
    __host__ __device__ mat2::mat2()
    {
        value = new vec2[2];
        for (int i = 0; i < 2; i++)
        {
            value[i] = vec2();
        }
    }
    __host__ __device__ mat2::mat2(float v0)
    {
        value = new vec2[2];
        for (int i = 0; i < 2; i++)
        {
            value[i] = vec2(v0);
        }
    }
    __host__ __device__ mat2::mat2(vec2 v0, vec2 v1)
    {
        value = new vec2[2];
        value[0] = vec2(v0);
        value[1] = vec2(v1);
    }
    __host__ __device__ mat2::~mat2()
    {
        delete[] value;
    }
}