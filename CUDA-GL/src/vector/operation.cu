#include "../../include/vector.cuh"

namespace CUDA_GL
{
    __host__ __device__ vec2 cross(vec2 a, vec2 b)
    {
        // [Ax]   [Bx]   [Ay⋅Bx - Ay⋅Bx]
        // [Ay] x [By] = [Ax⋅By − Ax⋅By]
        // Return 0 for vec2
        return vec2();
    }
    __host__ __device__ vec3 cross(vec3 a, vec3 b)
    {
        // [Ax]   [Bx]   [Ay⋅Bz - Az⋅By]
        // [Ay] x [By] = [Az⋅Bx − Ax⋅Bz]
        // [Az]   [Bz]   [Ax⋅By − Ay⋅Bx]
        return vec3(
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        );
    }
    __host__ __device__ vec4 cross(vec4 a, vec4 b)
    {
        // [Ax]   [Bx]   [Ay⋅Bz - Aω⋅Bz]
        // [Ay] x [By] = [Az⋅Bω − Ax⋅Bω]
        // [Az]   [Bz]   [Aω⋅Bx − Ay⋅Bx]
        // [Aω]   [Bω]   [Ax⋅By - Az⋅By]
        return vec4(
            b[2] * (a[1] - a[3]),
            b[3] * (a[2] - a[0]),
            b[0] * (a[3] - a[1]),
            b[1] * (a[0] - a[2])
        );
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