#pragma once

#include <iostream>
#include <cmath>
#include <stdio.h>

namespace CUDA_GL
{
    class vec2
    {
        private:
            float value[2];
        public:
            __host__ __device__ vec2();
            __host__ __device__ vec2(float v0);
            __host__ __device__ vec2(float v0, float v1);
            __host__ __device__ vec2(const vec2& vector);

            __host__ __device__ float& operator [] (unsigned int index);
            __host__ __device__ float operator [] (unsigned int index) const;
            __host__ __device__ vec2& operator = (const vec2& vector);

            __host__ __device__ vec2 operator + (float scalar) const;
            __host__ __device__ vec2& operator += (float scalar);
            __host__ __device__ vec2 operator + (const vec2& vector) const;
            __host__ __device__ vec2& operator += (const vec2& vector);

            __host__ __device__ vec2 operator - (float scalar) const;
            __host__ __device__ vec2& operator -= (float scalar);
            __host__ __device__ vec2 operator - (const vec2& vector) const;
            __host__ __device__ vec2& operator -= (const vec2& vector);

            __host__ __device__ vec2 operator * (float scalar) const;
            __host__ __device__ vec2& operator *= (float scalar);

            __host__ __device__ vec2 operator / (float scalar) const;
            __host__ __device__ vec2& operator /= (float scalar);

            __host__ friend std::ostream& operator << (std::ostream& os, const vec2& vector);
        };
    class vec3
    {
        private:
            float value[3];
        public:
            __host__ __device__ vec3();
            __host__ __device__ vec3(float v0);
            __host__ __device__ vec3(float v0, float v1, float v2);
            __host__ __device__ vec3(const vec3& vector);

            __host__ __device__ float& operator [] (unsigned int index);
            __host__ __device__ float operator [] (unsigned int index) const;
            __host__ __device__ vec3& operator = (const vec3& vector);

            __host__ __device__ vec3 operator + (float scalar) const;
            __host__ __device__ vec3& operator += (float scalar);
            __host__ __device__ vec3 operator + (const vec3& vector) const;
            __host__ __device__ vec3& operator += (const vec3& vector);

            __host__ __device__ vec3 operator - (float scalar) const;
            __host__ __device__ vec3& operator -= (float scalar);
            __host__ __device__ vec3 operator - (const vec3& vector) const;
            __host__ __device__ vec3& operator -= (const vec3& vector);

            __host__ __device__ vec3 operator * (float scalar) const;
            __host__ __device__ vec3& operator *= (float scalar);

            __host__ __device__ vec3 operator / (float scalar) const;
            __host__ __device__ vec3& operator /= (float scalar);
            
            __host__ friend std::ostream& operator << (std::ostream& os, const vec3& vector);
        };
    class vec4
    {
        private:
            float value[4];
        public:
            __host__ __device__ vec4();
            __host__ __device__ vec4(float v0);
            __host__ __device__ vec4(float v0, float v1, float v2, float v3);
            __host__ __device__ vec4(const vec4& vector);

            __host__ __device__ float& operator [] (unsigned int index);
            __host__ __device__ float operator [] (unsigned int index) const;
            __host__ __device__ vec4& operator = (const vec4& vector);

            __host__ __device__ vec4 operator + (float scalar) const;
            __host__ __device__ vec4& operator += (float scalar);
            __host__ __device__ vec4 operator + (const vec4& vector) const;
            __host__ __device__ vec4& operator += (const vec4& vector);

            __host__ __device__ vec4 operator - (float scalar) const;
            __host__ __device__ vec4& operator -= (float scalar);
            __host__ __device__ vec4 operator - (const vec4& vector) const;
            __host__ __device__ vec4& operator -= (const vec4& vector);

            __host__ __device__ vec4 operator * (float scalar) const;
            __host__ __device__ vec4& operator *= (float scalar);

            __host__ __device__ vec4 operator / (float scalar) const;
            __host__ __device__ vec4& operator /= (float scalar);

            __host__ friend std::ostream& operator << (std::ostream& os, const vec4& vector);
        };

    // Vector exclusive operation
    __host__ __device__ vec2 cross(const vec2& a, const vec2& b);
    __host__ __device__ vec3 cross(const vec3& a, const vec3& b);
    __host__ __device__ vec4 cross(const vec4& a, const vec4& b);

    __host__ __device__ float dot(const vec2& a, const vec2& b);
    __host__ __device__ float dot(const vec3& a, const vec3& b);
    __host__ __device__ float dot(const vec4& a, const vec4& b);

    __host__ __device__ float length(const vec2& vector);
    __host__ __device__ float length(const vec3& vector);
    __host__ __device__ float length(const vec4& vector);

    __host__ __device__ vec2 normalize(const vec2& vector);
    __host__ __device__ vec3 normalize(const vec3& vector);
    __host__ __device__ vec4 normalize(const vec4& vector);
}