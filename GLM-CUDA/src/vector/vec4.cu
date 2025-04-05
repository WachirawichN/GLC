#include "../../include/vector.cuh"

namespace GLM_CUDA
{
    __host__ __device__ vec4::vec4()
    {
        value = new float[4];
        for (int i = 0; i < 4; i++)
        {
            value[i] = 0.0f;
        }
    }
    __host__ __device__ vec4::vec4(float v0)
    {
        value = new float[4];
        for (int i = 0; i < 4; i++)
        {
            value[i] = v0;
        }
    }
    __host__ __device__ vec4::vec4(float v0, float v1, float v2, float v3)
    {
        value = new float[4];
        value[0] = v0;
        value[1] = v1;
        value[2] = v2;
        value[3] = v3;
    }
    __host__ __device__ vec4::~vec4()
    {
        delete[] value;
    }


    __host__ __device__ float vec4::operator[](unsigned int index) const
    {
        return value[index];
    }
    __host__ __device__ indexProxy vec4::operator[](unsigned int index)
    {
        return indexProxy(value[index]);
    }
    __host__ __device__ vec4& vec4::operator=(const vec4& vector)
    {
        if (this != &vector)
        {
            // Prevent memory leak
            delete[] value;
            value = new float[4];

            for (int i = 0; i < 4; i++)
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


    __host__ __device__ vec4 vec4::operator+(vec4 vector)
    {
        vec4 out;
        for (int i = 0; i < 4; i++)
        {
            out[i] = value[i] + vector[i];
        }
        return out;
    }
    __host__ __device__ vec4& vec4::operator+=(vec4& vector)
    {
        for (int i = 0; i < 4; i++)
        {
            value[i] += vector[i];
        }
        return *this;
    }

    __host__ __device__ vec4 vec4::operator-(vec4 vector)
    {
        vec4 out;
        for (int i = 0; i < 4; i++)
        {
            out[i] = value[i] - vector[i];
        }
        return out;
    }
    __host__ __device__ vec4& vec4::operator-=(vec4& vector)
    {
        for (int i = 0; i < 4; i++)
        {
            value[i] -= vector[i];
        }
        return *this;
    }

    __host__ __device__ vec4 vec4::operator*(float scalar)
    {
        vec4 out;
        for (int i = 0; i < 4; i++)
        {
            out[i] = value[i] * scalar;
        }
        return out;
    }
    __host__ __device__ vec4& vec4::operator*=(float scalar)
    {
        for (int i = 0; i < 4; i++)
        {
            value[i] *= scalar;
        }
        return *this;
    }

    __host__ __device__ vec4 vec4::operator/(float scalar)
    {
        vec4 out;
        for (int i = 0; i < 4; i++)
        {
            out[i] = value[i] / scalar;
        }
        return out;
    }
    __host__ __device__ vec4& vec4::operator/=(float scalar)
    {
        for (int i = 0; i < 4; i++)
        {
            value[i] /= scalar;
        }
        return *this;
    }

    __host__ __device__ std::ostream& operator << (std::ostream& os, const vec4& vector)
    {
        return os << "[" << vector[0] << ", " << vector[1] << ", " << vector[2] << ", " << vector[3] << "]";
    }
}