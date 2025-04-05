#include "../../include/matrix.cuh"

#include "../../include/utility.cuh"

namespace GLM_CUDA
{
    __host__ __device__ mat4::mat4()
    {
        value = new vec4[4];
        value[0] = vec4();
        value[1] = vec4();
        value[2] = vec4();
        value[3] = vec4();
    }
    __host__ __device__ mat4::mat4(float v0)
    {
        value = new vec4[4];
        value[0] = vec4(v0);
        value[1] = vec4(v0);
        value[2] = vec4(v0);
        value[3] = vec4(v0);
    }
    __host__ __device__ mat4::mat4(vec4 v0, vec4 v1, vec4 v2, vec4 v3)
    {
        value = new vec4[4];
        value[0] = vec4(v0);
        value[1] = vec4(v1);
        value[2] = vec4(v2);
        value[3] = vec4(v3);
    }
    __host__ __device__ mat4::~mat4()
    {
        delete[] value;
    }
}