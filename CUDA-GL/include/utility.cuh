#pragma once

#include <cmath>
#include <type_traits>
#include <concepts>
#include <stdio.h>

#include "matrix.cuh"
#include "vector.cuh"

namespace CUDA_GL
{
    /*------------------------------------------
        Common math funtions
        Available for both matrix and vector
    ------------------------------------------*/

    template<vectorType T>
    /**
     * @brief Raised vector to power of exponent.
     * 
     * Raised every element inside vector or matrix to the power of the exponent.
     * 
     * @param vector V e c t o r .
     * @param exponent Exponent of the vector.
     * @return Raised vector.
     */
    __host__ __device__ T pow(const T& vector, float exponent)
    {
        T out;
        for (int i = 0; i < (int)(sizeof(T) / sizeof(float)); i++)
        {
            #ifdef __CUDA_ARCH__
                out[i] = powf(vector[i], exponent);
            #else
                out[i] = std::powf(vector[i], exponent);
            #endif
        }
        return out;
    }
    template<matrixType T>
    /**
     * @brief Raised matrix to power of exponent.
     * 
     * Raised every element inside vector or matrix to the power of the exponent.
     * 
     * @param matrix M a t r i x .
     * @param exponent Exponent of the matrix.
     * @return Raised matrix.
     */
    __host__ __device__ T pow(const T& matrix, float exponent)
    {
        T out;
        #ifdef __CUDA_ARCH__
            int size = (int)sqrtf(sizeof(T) / sizeof(float));
        #else
            int size = (int)std::sqrtf(sizeof(T) / sizeof(float));
        #endif
        for (int i = 0; i < size; i++)
        {
            out[i] = pow(matrix[i], exponent);
        }
        return out;
    }

    /*------------------------------------------
        Graphic funtions
    ------------------------------------------*/

    /**
     * @brief Generate perspective projection matrix.
     * @param fov Fov of the camera.
     * @param aspect Aspect ratio of the camera.
     * @param near Length before the object will be clipped with camera.
     * @param far Length before the object will be unrendered due to how far it is.
     * @return Perspective projection matrix.
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
     * @return matrix where camera look at.
     */
    __host__ mat4 lookAt(const vec3& position, const vec3& target, const vec3& up);

    /*------------------------------------------
        Model funtions
    ------------------------------------------*/

    /**
     * @brief Translate the input matrix or generate translation matrix.
     * @param offset vector of how much translation we want.
     * @param matrix matrix that we want to add translation to. (OPTIONAL)
     * @return Translation matrix or translated version of the input matrix.
     */
    __host__ __device__ mat4 translate(const vec3& offset);
    __host__ __device__ mat4 translate(const vec3& offset, const mat4& matrix);
    /**
     * @brief Rotate the input matrix or generate rotation matrix.
     * @param angle How much angle we want to rotate.
     * @param axis vector of axis we want to rotate.
     * @param matrix matrix that we want to rotate. (OPTIONAL)
     * @return Rotation matrix or rotated version of the input matrix.
     */
    __host__ __device__ mat4 rotate(float angle, const vec3& axis);
    __host__ __device__ mat4 rotate(float angle, const vec3& axis, const mat4& matrix);
    /**
     * @brief Scale the input matrix or generate scaling matrix.
     * @param factor How much we want to scale on each axis.
     * @param matrix matrix that we add scale to. (Scaling will be multiply on the old one) (OPTIONAL)
     * @return Scaling matrix or scaled version of the input matrix.
     */
    __host__ __device__ mat4 scale(const vec3& factor);
    __host__ __device__ mat4 scale(const vec3& factor, const mat4& matrix);

    /*------------------------------------------
        Other funtions
    ------------------------------------------*/

    template<vectorType T>
    /**
     * @brief Unpack vector or matrix into array of float. Similar to value_ptr of GLM.
     * @param vector V e c t o r .
     * @return Pointer to array of float containing every values of the input vector.
     */
    __host__ __device__ float* unpack(const T& vector)
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
     * @brief Unpack vector or matrix into array of float. Similar to value_ptr of GLM.
     * @param matrix M a t r i x .
     * @return Pointer to array of float containing every values of the input matrix.
     */
    __host__ __device__ float* unpack(const T& matrix)
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