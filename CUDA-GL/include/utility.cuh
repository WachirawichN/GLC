#pragma once

#include <cmath>
#include <type_traits>
#include <concepts>

#include "matrix.cuh"
#include "vector.cuh"

namespace CUDA_GL
{
    /*------------------------------------------
        Common math funtions
        Available for both matrix and vector
    ------------------------------------------*/

    __host__ __device__ vec2 pow(const vec2& vector, float exponent);
    __host__ __device__ vec3 pow(const vec3& vector, float exponent);
    __host__ __device__ vec4 pow(const vec4& vector, float exponent);

    /*------------------------------------------
        Graphic funtions
    ------------------------------------------*/

    /**
     * @brief Generate perspective projection matrix.
     * @param fov Fov of the camera.
     * @param aspect Aspect ratio of the camera.
     * @param near Length before the object will be clipped with camera.
     * @param far Length before the object will be unrendered due to how far it is.
     */
    __host__ mat4 perspective(float fov, float aspect, float near, float far);
    /**
     * @brief Generate orthographic projection matrix. (Untested)
     */
    __host__ mat4 ortho(float left, float right, float bottom, float top, float near, float far);
    /**
     * @brief Generate matrix of where camera look at.
     * @param position Position of camera.
     * @param target Target that where camera look at.
     * @param up Up vector.
     */
    __host__ mat4 lookAt(const vec3& position, const vec3& target, const vec3& up);

    /*------------------------------------------
        Model funtions
    ------------------------------------------*/

    /**
     * @brief Translate the input matrix or generate translation matrix.
     * @param offset Vector of how much translation we want.
     * @param matrix Matrix that we want to add translation to. (OPTIONAL)
     */
    __host__ __device__ mat4 translate(const vec3& offset);
    __host__ __device__ mat4 translate(const vec3& offset, const mat4& matrix);
    /**
     * @brief Rotate the input matrix or generate rotation matrix.
     * @param angle How much angle we want to rotate.
     * @param axis Vector of axis we want to rotate.
     * @param matrix Matrix that we want to rotate. (OPTIONAL)
     */
    __host__ __device__ mat4 rotate(float angle, const vec3& axis);
    __host__ __device__ mat4 rotate(float angle, const vec3& axis, const mat4& matrix);
    /**
     * @brief Scale the input matrix or generate scaling matrix.
     * @param factor How much we want to scale on each axis.
     * @param matrix Matrix that we add scale to. (Scaling will be multiply on the old one) (OPTIONAL)
     */
    __host__ __device__ mat4 scale(const vec3& factor);
    __host__ __device__ mat4 scale(const vec3& factor, const mat4& matrix);

    /*------------------------------------------
        Other funtions
    ------------------------------------------*/
    
    template<typename T>
    concept vectorType = std::same_as<T, CUDA_GL::vec2> || std::same_as<T, CUDA_GL::vec3> || std::same_as<T, CUDA_GL::vec4>;
    template<typename T>
    concept matrixType = std::same_as<T, CUDA_GL::mat2> || std::same_as<T, CUDA_GL::mat3> || std::same_as<T, CUDA_GL::mat4>;

    template<vectorType T>
    /**
     * @brief Unpack vector or matrix into array of float.
     * @param vector V e c t o r .
     */
    __host__ __device__ float* unpack(T vector)
    {
        static float out[sizeof(T) / sizeof(float)];
        for (int i = 0; i < sizeof(T) / sizeof(float); i++)
        {
            out[i] = vector[i];
        }
        return out;
    }
    template<matrixType T>
    /**
     * @brief Unpack vector or matrix into array of float.
     * @param matrix M a t r i x .
     */
    __host__ __device__ float* unpack(T matrix)
    {
        static float out[sizeof(T) / sizeof(float)];
        #ifdef __CUDA_ARCH__
            int size = (int)sqrtf(sizeof(T) / sizeof(float));
        #else
            int size = (int)std::sqrtf(sizeof(T) / sizeof(float));
        #endif
        for (int column = 0; column < size; column++)
        {
            for (int row = 0; row < size; row++)
            {
                out[column * size + row] = matrix[column][row];
            }
        }
        return out;
    }
    __device__ int threadID();
}