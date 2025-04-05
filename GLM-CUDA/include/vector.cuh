#pragma once

#include <iostream>

class indexProxy
{
    private:
        float& ref;
    public:
        __host__ __device__ indexProxy(float& vector);

        __host__ __device__ indexProxy& operator = (float vector);
        __host__ __device__ indexProxy& operator += (float vector);
        __host__ __device__ indexProxy& operator -= (float vector);
        __host__ __device__ indexProxy& operator *= (float vector);
        __host__ __device__ indexProxy& operator /= (float vector);

        __host__ __device__ operator float() const;
};

namespace GLM_CUDA
{
    class vec2
    {
        private:
            float* value;
        public:
            __host__ __device__ vec2();
            __host__ __device__ vec2(float v0);
            __host__ __device__ vec2(float v0, float v1);
            __host__ __device__ ~vec2();


            __host__ __device__ float operator [] (unsigned int index) const;
            __host__ __device__ indexProxy operator [] (unsigned int index);
            __host__ __device__ vec2& operator = (const vec2& vector);

            __host__ __device__ vec2 operator + (vec2 vector);
            __host__ __device__ vec2& operator += (vec2& vector);

            __host__ __device__ vec2 operator - (vec2 vector);
            __host__ __device__ vec2& operator -= (vec2& vector);

            __host__ __device__ vec2 operator * (float scalar);
            __host__ __device__ vec2& operator *= (float scalar);

            __host__ __device__ vec2 operator / (float scalar);
            __host__ __device__ vec2& operator /= (float scalar);
        };

    class vec3
    {
        private:
            float* value;
        public:
            __host__ __device__ vec3();
            __host__ __device__ vec3(float v0);
            __host__ __device__ vec3(float v0, float v1, float v2);
            __host__ __device__ ~vec3();

            __host__ __device__ float operator [] (unsigned int index) const;
            __host__ __device__ indexProxy operator [] (unsigned int index);
            __host__ __device__ vec3& operator = (const vec3& vector);

            __host__ __device__ vec3 operator + (vec3 vector);
            __host__ __device__ vec3& operator += (vec3& vector);

            __host__ __device__ vec3 operator - (vec3 vector);
            __host__ __device__ vec3& operator -= (vec3& vector);

            __host__ __device__ vec3 operator * (float scalar);
            __host__ __device__ vec3& operator *= (float scalar);

            __host__ __device__ vec3 operator / (float scalar);
            __host__ __device__ vec3& operator /= (float scalar);
            
            __host__ __device__ friend std::ostream& operator << (std::ostream& os, const vec3& vector);
        };

    class vec4
    {
        private:
            float* value;
        public:
            __host__ __device__ vec4();
            __host__ __device__ vec4(float v0);
            __host__ __device__ vec4(float v0, float v1, float v2, float v3);
            __host__ __device__ ~vec4();


            __host__ __device__ float operator [] (unsigned int index) const;
            __host__ __device__ indexProxy operator [] (unsigned int index);
            __host__ __device__ vec4& operator = (const vec4& vector);

            __host__ __device__ vec4 operator + (vec4 vector);
            __host__ __device__ vec4& operator += (vec4& vector);

            __host__ __device__ vec4 operator - (vec4 vector);
            __host__ __device__ vec4& operator -= (vec4& vector);

            __host__ __device__ vec4 operator * (float scalar);
            __host__ __device__ vec4& operator *= (float scalar);

            __host__ __device__ vec4 operator / (float scalar);
            __host__ __device__ vec4& operator /= (float scalar);
        };

    // Vector exclusive operation
    __host__ __device__ vec2 cross(vec2 a, vec2 b);
    __host__ __device__ vec3 cross(vec3 a, vec3 b);
    __host__ __device__ vec4 cross(vec4 a, vec4 b);
}