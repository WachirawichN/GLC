#include "../../include/utility.cuh"

namespace CUDA_GL
{
    __host__ __device__ vec2 pow(const vec2& vector, float exponent)
    {
        #ifdef __CUDA_ARCH__
            return vec2(powf(vector[0], exponent), powf(vector[1], exponent));
        #else
            return vec2(std::powf(vector[0], exponent), std::powf(vector[1], exponent));
        #endif
    }
    __host__ __device__ vec3 pow(const vec3& vector, float exponent)
    {
        #ifdef __CUDA_ARCH__
            return vec3(powf(vector[0], exponent), powf(vector[1], exponent), powf(vector[2], exponent));
        #else
            return vec3(std::powf(vector[0], exponent), std::powf(vector[1], exponent), std::powf(vector[2], exponent));
        #endif
    }
    __host__ __device__ vec4 pow(const vec4& vector, float exponent)
    {
        #ifdef __CUDA_ARCH__
            return vec4(powf(vector[0], exponent), powf(vector[1], exponent), powf(vector[2], exponent), powf(vector[3], exponent));
        #else
            return vec4(std::powf(vector[0], exponent), std::powf(vector[1], exponent), std::powf(vector[2], exponent), std::powf(vector[3], exponent));
        #endif
    }
}