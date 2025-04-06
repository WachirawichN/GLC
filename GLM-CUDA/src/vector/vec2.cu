#include "../../include/vector.cuh"

namespace GLM_CUDA
{
    __host__ __device__ vec2::vec2()
    {
        value = new float[2];
        for (int i = 0; i < 2; i++)
        {
            value[i] = 0.0f;
        }
    }
    __host__ __device__ vec2::vec2(float v0)
    {
        value = new float[2];
        for (int i = 0; i < 2; i++)
        {
            value[i] = v0;
        }
    }
    __host__ __device__ vec2::vec2(float v0, float v1)
    {
        value = new float[2] {v0, v1};
    }
    __host__ __device__ vec2::vec2(const vec2& vector)
    {
        value = new float[2];
        for (int i = 0; i < 2; i++)
        {
            value[i] = vector[i];
        }
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
    __host__ __device__ vec2& vec2::operator=(const vec2& vector)
    {
        if (this != &vector)
        {
            // Prevent memory leak
            if (value)
            {
                delete[] value;
                value = new float[2];
            }

            for (int i = 0; i < 2; i++)
            {
                value[i] = vector[i];
            }

            // Copy data - Using CUDA's memcpy if possible (More efficient, still contain a bug)
            /*
            #ifdef __CUDA_ARCH__
                cudaMemcpy(
                    value,
                    matrix,
                    3 * sizeof(vec3),
                    cudaMemcpyDeviceToDevice
                );
            #else
                std::copy(matrix[0], matrix[2], value);
            #endif
            */
        }
        return *this;
    }


    __host__ __device__ vec2 vec2::operator+(vec2 vector)
    {
        vec2 out;
        for (int i = 0; i < 2; i++)
        {
            out[i] = value[i] + vector[i];
        }
        return out;
    }
    __host__ __device__ vec2& vec2::operator+=(vec2& vector)
    {
        for (int i = 0; i < 2; i++)
        {
            value[i] += vector[i];
        }
        return *this;
    }

    __host__ __device__ vec2 vec2::operator-(vec2 vector)
    {
        vec2 out;
        for (int i = 0; i < 2; i++)
        {
            out[i] = value[i] - vector[i];
        }
        return out;
    }
    __host__ __device__ vec2& vec2::operator-=(vec2& vector)
    {
        for (int i = 0; i < 2; i++)
        {
            value[i] -= vector[i];
        }
        return *this;
    }

    __host__ __device__ vec2 vec2::operator*(float scalar)
    {
        vec2 out;
        for (int i = 0; i < 2; i++)
        {
            out[i] = value[i] * scalar;
        }
        return out;
    }
    __host__ __device__ vec2& vec2::operator*=(float scalar)
    {
        for (int i = 0; i < 2; i++)
        {
            value[i] *= scalar;
        }
        return *this;
    }

    __host__ __device__ vec2 vec2::operator/(float scalar)
    {
        vec2 out;
        for (int i = 0; i < 2; i++)
        {
            out[i] = value[i] / scalar;
        }
        return out;
    }
    __host__ __device__ vec2& vec2::operator/=(float scalar)
    {
        for (int i = 0; i < 2; i++)
        {
            value[i] /= scalar;
        }
        return *this;
    }

    __host__ std::ostream& operator << (std::ostream& os, const vec2& vector)
    {
        return os << "[" << vector[0] << ", " << vector[1] << "]";
    }
}