#pragma once

#include <iostream>
#include <string>
#include "vector.cuh"

// This will be weird due to how weird GLSL's matrix structure is. (Column first then Row)
// 3x3 Matrix example bruh
// ┌ ┌   ┐ ┌   ┐ ┌   ┐ ┐
// | | a | | d | | g | |
// | | b | | e | | h | |
// | | c | | f | | i | |
// └ └   ┘ └   ┘ └   ┘ ┘
// Printing out matrix will result in slighty different formation than this tho, console doesn't support printing those special char.

namespace GLM_CUDA
{
    class mat2
    {
        private:
            vec2* value;
        public:
            __host__ __device__ mat2();
            __host__ __device__ mat2(float v0);
            __host__ __device__ mat2(vec2 v0, vec2 v1);
            __host__ __device__ ~mat2();

            __host__ __device__ vec3& operator [] (unsigned int index);
            __host__ __device__ const vec3& operator [] (unsigned int index) const;
            __host__ __device__ mat2& operator = (const mat2& matrix);

            __host__ __device__ mat2 operator + (mat2 matrix);
            __host__ __device__ mat2& operator += (mat2& matrix);

            __host__ __device__ mat2 operator - (mat2 matrix);
            __host__ __device__ mat2& operator -= (mat2& matrix);

            __host__ __device__ mat2 operator * (float scalar);
            __host__ __device__ mat2 operator * (mat2 matrix);
            __host__ __device__ mat2& operator *= (float scalar);
            __host__ __device__ mat2& operator *= (mat2 matrix);

            __host__ __device__ mat2 operator / (float scalar);
            __host__ __device__ mat2& operator /= (float scalar);

            __host__ __device__ friend std::ostream& operator << (std::ostream& os, const mat2& matrix);
    };
    class mat3
    {
        private:
            vec3* value;
        public:
            __host__ __device__ mat3();
            __host__ __device__ mat3(float v0);
            __host__ __device__ mat3(const vec3& v0, const vec3& v1, const vec3& v2);
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

            __host__ __device__ friend std::ostream& operator << (std::ostream& os, const mat3& matrix);
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

            __host__ __device__ friend std::ostream& operator << (std::ostream& os, const mat4& matrix);
    };

    // Matrix exclusive operation
    __host__ __device__ mat2 transpose(mat2 matrix);
    __host__ __device__ mat3 transpose(mat3 matrix);
    __host__ __device__ mat4 transpose(mat4 matrix);
}