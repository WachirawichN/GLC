#include "../../include/vector.cuh"

namespace GLM_CUDA
{
    __host__ __device__ vec3::vec3()
    {
        value = new float[3];
        for (int i = 0; i < 3; i++)
        {
            value[i] = 0.0f;
        }
    }
    __host__ __device__ vec3::vec3(float v0)
    {
        value = new float[3];
        for (int i = 0; i < 3; i++)
        {
            value[i] = v0;
        }
    }
    __host__ __device__ vec3::vec3(float v0, float v1, float v2)
    {
        value = new float[3] {v0, v1, v2};
    }
    __host__ __device__ vec3::vec3(const vec3& vector)
    {
        value = new float[3];
        for (int i = 0; i < 3; i++)
        {
            value[i] = vector[i];
        }
    }
    __host__ __device__ vec3::~vec3()
    {
        delete[] value;
    }


    __host__ __device__ float vec3::operator[](unsigned int index) const
    {
        return value[index];
    }
    __host__ __device__ indexProxy vec3::operator[](unsigned int index)
    {
        return indexProxy(value[index]);
    }
    __host__ __device__ vec3& vec3::operator=(const vec3& vector)
    {
        if (this != &vector)
        {
            // Prevent memory leak
            if (value)
            {
                delete[] value;
                value = new float[3];
            }

            for (int i = 0; i < 3; i++)
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


    __host__ __device__ vec3 vec3::operator+(const vec3& vector)
    {
        vec3 out;
        for (int i = 0; i < 3; i++)
        {
            out[i] = value[i] + vector[i];
        }
        return out;
    }
    __host__ __device__ vec3& vec3::operator+=(const vec3& vector)
    {
        for (int i = 0; i < 3; i++)
        {
            value[i] += vector[i];
        }
        return *this;
    }

    __host__ __device__ vec3 vec3::operator-(const vec3& vector)
    {
        vec3 out;
        for (int i = 0; i < 3; i++)
        {
            out[i] = value[i] - vector[i];
        }
        return out;
    }
    __host__ __device__ vec3& vec3::operator-=(const vec3& vector)
    {
        for (int i = 0; i < 3; i++)
        {
            value[i] -= vector[i];
        }
        return *this;
    }

    __host__ __device__ vec3 vec3::operator*(float scalar)
    {
        vec3 out;
        for (int i = 0; i < 3; i++)
        {
            out[i] = value[i] * scalar;
        }
        return out;
    }
    __host__ __device__ vec3& vec3::operator*=(float scalar)
    {
        for (int i = 0; i < 3; i++)
        {
            value[i] *= scalar;
        }
        return *this;
    }

    __host__ __device__ vec3 vec3::operator/(float scalar)
    {
        vec3 out;
        for (int i = 0; i < 3; i++)
        {
            out[i] = value[i] / scalar;
        }
        return out;
    }
    __host__ __device__ vec3& vec3::operator/=(float scalar)
    {
        for (int i = 0; i < 3; i++)
        {
            value[i] /= scalar;
        }
        return *this;
    }

    __host__ std::ostream& operator << (std::ostream& os, const vec3& vector)
    {
        return os << "[" << vector[0] << ", " << vector[1] << ", " << vector[2] << "]";
    }
}