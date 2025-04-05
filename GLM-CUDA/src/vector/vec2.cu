#include "../../include/vector.cuh"

namespace GLM_CUDA
{
    __host__ __device__ vec2::vec2()
    {
        value = new float[2];

        value[0] = 0.0f;
        value[1] = 0.0f;
    }
    __host__ __device__ vec2::vec2(float v0)
    {
        value = new float[2];

        value[0] = v0;
        value[1] = v0;
    }
    __host__ __device__ vec2::vec2(float v0, float v1)
    {
        value = new float[2];

        value[0] = v0;
        value[1] = v1;
    }
    __host__ __device__ vec2::~vec2()
    {
        delete[] value;
    }


    __host__ __device__ float vec2::operator[](unsigned int index) const
    {
        return value[index];
    }
    __host__ __device__ indexProxy vec2::operator[](unsigned int index)
    {
        return indexProxy(value[index]);
    }
    __host__ __device__ vec2& vec2::operator=(const vec2& input)
    {
        value[0] = input[0];
        value[1] = input[1];
        return *this;
    }


    __host__ __device__ vec2 vec2::operator+(vec2 input)
    {
        return vec2(value[0] + input[0], value[1] + input[1]);
    }
    __host__ __device__ vec2& vec2::operator+=(vec2& input)
    {
        value[0] += input[0];
        value[1] += input[1];
        return *this;
    }

    __host__ __device__ vec2 vec2::operator-(vec2 input)
    {
        return vec2(value[0] - input[0], value[1] - input[1]);
    }
    __host__ __device__ vec2& vec2::operator-=(vec2& input)
    {
        value[0] -= input[0];
        value[1] -= input[1];
        return *this;
    }

    __host__ __device__ vec2 vec2::operator*(float scalar)
    {
        return vec2(value[0] * scalar, value[1] * scalar);
    }
    __host__ __device__ vec2& vec2::operator*=(float scalar)
    {
        value[0] *= scalar;
        value[1] *= scalar;
        return *this;
    }

    __host__ __device__ vec2 vec2::operator/(float scalar)
    {
        return vec2(value[0] / scalar, value[1] / scalar);
    }
    __host__ __device__ vec2& vec2::operator/=(float scalar)
    {
        value[0] /= scalar;
        value[1] /= scalar;
        return *this;
    }
}