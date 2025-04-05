#pragma once

#include "vector.cuh"

namespace GLM_CUDA
{
    class mat3
    {
        private:
            vec3* value;
        public:
            __host__ __device__ mat3();
            __host__ __device__ mat3(float v0);
            __host__ __device__ mat3(vec3 v0, vec3 v1, vec3 v2);
            __host__ __device__ ~mat3();

            __host__ __device__ vec3& operator [] (unsigned int index);
            __host__ __device__ const vec3& operator [] (unsigned int index) const;
            __host__ __device__ mat3& operator = (const mat3& matrix);

            __host__ __device__ mat3 operator + (mat3 matrix);
            __host__ __device__ mat3& operator += (mat3& matrix);

            __host__ __device__ mat3 operator - (mat3 matrix);
            __host__ __device__ mat3& operator -= (mat3& matrix);

            __host__ __device__ mat3 operator * (float scalar);
            __host__ __device__ mat3 operator * (mat3 matrix);
            __host__ __device__ mat3& operator *= (float scalar);
            __host__ __device__ mat3& operator *= (mat3 matrix);

            __host__ __device__ mat3 operator / (float scalar);
            __host__ __device__ mat3& operator /= (float scalar);
    };
    class mat4
    {
        private:
            vec4* value;
        public:
            __host__ __device__ mat4();
            __host__ __device__ mat4(float v0);
            __host__ __device__ mat4(vec4 v0, vec4 v1, vec4 v2, vec4 v3);
            __host__ __device__ ~mat4();

            __host__ __device__ vec4 operator [] (unsigned int index) const;
            __host__ __device__ mat4& operator = (const mat4& matrix);

            __host__ __device__ mat4 operator + (mat4 matrix);
            __host__ __device__ mat4& operator += (mat4& matrix);

            __host__ __device__ mat4 operator - (mat4 matrix);
            __host__ __device__ mat4& operator -= (mat4& matrix);

            __host__ __device__ mat4 operator * (float scalar);
            __host__ __device__ mat4 operator * (mat4 matrix);
            __host__ __device__ mat4& operator *= (float scalar);
            __host__ __device__ mat4& operator *= (mat4 matrix);

            __host__ __device__ mat4 operator / (float scalar);
            __host__ __device__ mat4& operator /= (float scalar);
    };

    // Matrix exclusive operation
    __host__ __device__ mat3 transpose(mat3 matrix);
    __host__ __device__ mat4 transpose(mat4 matrix);
}