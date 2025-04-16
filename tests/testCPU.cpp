#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

#include "../CUDA-GL/include/matrix.cuh"
#include "../CUDA-GL/include/utility.cuh"
#include "../CUDA-GL/include/vector.cuh"

template <typename T>
void testMat(T& matrix1, T& matrix2)
{
    std::cout << "First matrix" << std::endl;
    std::cout << matrix1 << std::endl;
    std::cout << "Second matrix" << std::endl;
    std::cout << matrix2 << std::endl;

    {
        std::cout << "Addition" << std::endl;
        std::cout << matrix1 + matrix2 << std::endl;
        matrix1 += matrix2;
        std::cout << matrix1 << std::endl;
    }
    {
        std::cout << "Subtraction" << std::endl;
        std::cout << matrix1 - matrix2 << std::endl;
        matrix1 -= matrix2;
        std::cout << matrix1 << std::endl;
    }
    {
        std::cout << "Scalar multiplication" << std::endl;
        std::cout << matrix1 * 2.0f << std::endl;
        matrix1 *= 2.0f;
        std::cout << matrix1 << std::endl;
    }
    {
        std::cout << "Scalar division" << std::endl;
        std::cout << matrix1 / 2.0f << std::endl;
        matrix1 /= 2.0f;
        std::cout << matrix1 << std::endl;
    }
    {
        std::cout << "Matrix multiplication" << std::endl;
        std::cout << matrix1 * matrix2 << std::endl;
        matrix1 *= matrix2;
        std::cout << matrix1 << std::endl;
    }
}

template <typename mat, typename vec>
void testMatVec(mat& matrix, const vec& vector)
{
    std::cout << "Matrix:" << std::endl;
    std::cout << matrix;
    std::cout << "Vector: " << vector << std::endl;
    std::cout << "Result of multiplication: " << matrix * vector << std::endl;
}

float RNG()
{
    return ((rand() % 20001) - 10000.0f) / 1000.0f;
}

int main()
{
    std::srand(1);

    // Random
    {
        std::cout << "2x2 Random matrix x Random 2D vector" << std::endl;
        {
            CUDA_GL::vec2 val1(RNG(), RNG());
            CUDA_GL::mat2 testMat1(
                CUDA_GL::vec2(RNG(), RNG()),
                CUDA_GL::vec2(RNG(), RNG())
            );
            testMatVec<CUDA_GL::mat2, CUDA_GL::vec2>(testMat1, val1);
            std::cout << std::endl;
        }
        
        std::cout << "3x3 Random matrix x Random 3D vector" << std::endl;
        {
            CUDA_GL::vec3 val1(RNG(), RNG(), RNG());
            CUDA_GL::mat3 testMat1(
                CUDA_GL::vec3(RNG(), RNG(), RNG()),
                CUDA_GL::vec3(RNG(), RNG(), RNG()),
                CUDA_GL::vec3(RNG(), RNG(), RNG())
            );
            testMatVec<CUDA_GL::mat3, CUDA_GL::vec3>(testMat1, val1);
            std::cout << std::endl;
        }

        std::cout << "4x4 Random matrix x Random 4D vector" << std::endl;
        {
            CUDA_GL::vec4 val1(RNG(), RNG(), RNG(), RNG());
            CUDA_GL::mat4 testMat1(
                CUDA_GL::vec4(RNG(), RNG(), RNG(), RNG()),
                CUDA_GL::vec4(RNG(), RNG(), RNG(), RNG()),
                CUDA_GL::vec4(RNG(), RNG(), RNG(), RNG()),
                CUDA_GL::vec4(RNG(), RNG(), RNG(), RNG())
            );
            testMatVec<CUDA_GL::mat4, CUDA_GL::vec4>(testMat1, val1);
            std::cout << std::endl;
        }
    }
    
    // Identity matrix
    {
        std::cout << "2x2 Identity matrix x Random 2D vector" << std::endl;
        {
            CUDA_GL::vec2 val1(RNG(), RNG());
            CUDA_GL::mat2 testMat1(
                CUDA_GL::vec2(1.0f, 0.0f),
                CUDA_GL::vec2(0.0f, 1.0f)
            );
            testMatVec<CUDA_GL::mat2, CUDA_GL::vec2>(testMat1, val1);
            std::cout << std::endl;
        }
        
        std::cout << "3x3 Identity matrix x Random 3D vector" << std::endl;
        {
            CUDA_GL::vec3 val1(RNG(), RNG(), RNG());
            CUDA_GL::mat3 testMat1(
                CUDA_GL::vec3(1.0f, 0.0f, 0.0f),
                CUDA_GL::vec3(0.0f, 1.0f, 0.0f),
                CUDA_GL::vec3(0.0f, 0.0f, 1.0f)
            );
            testMatVec<CUDA_GL::mat3, CUDA_GL::vec3>(testMat1, val1);
            std::cout << std::endl;
        }

        std::cout << "4x4 Identity matrix x Random 4D vector" << std::endl;
        {
            CUDA_GL::vec4 val1(RNG(), RNG(), RNG(), RNG());
            CUDA_GL::mat4 testMat1(
                CUDA_GL::vec4(1.0f, 0.0f, 0.0f, 0.0f),
                CUDA_GL::vec4(0.0f, 1.0f, 0.0f, 0.0f),
                CUDA_GL::vec4(0.0f, 0.0f, 1.0f, 0.0f),
                CUDA_GL::vec4(0.0f, 0.0f, 0.0f, 1.0f)
            );
            testMatVec<CUDA_GL::mat4, CUDA_GL::vec4>(testMat1, val1);
            std::cout << std::endl;
        }
    }

    // Use case identity matrix
    {
        std::cout << "Use case 2x2 Identity matrix x Random 2D vector" << std::endl;
        {
            CUDA_GL::vec2 val1(RNG(), 1.0f);
            CUDA_GL::mat2 testMat1(
                CUDA_GL::vec2(1.0f, 0.0f),
                CUDA_GL::vec2(RNG(), 1.0f)
            );
            testMatVec<CUDA_GL::mat2, CUDA_GL::vec2>(testMat1, val1);
            std::cout << std::endl;
        }
        
        std::cout << "Use case 3x3 Identity matrix x Random 3D vector" << std::endl;
        {
            CUDA_GL::vec3 val1(RNG(), RNG(), 1.0f);
            CUDA_GL::mat3 testMat1(
                CUDA_GL::vec3(1.0f, 0.0f, 0.0f),
                CUDA_GL::vec3(0.0f, 1.0f, 0.0f),
                CUDA_GL::vec3(RNG(), RNG(), 1.0f)
            );
            testMatVec<CUDA_GL::mat3, CUDA_GL::vec3>(testMat1, val1);
            std::cout << std::endl;
        }

        std::cout << "Use case 4x4 Identity matrix x Random 4D vector" << std::endl;
        {
            CUDA_GL::vec4 val1(RNG(), RNG(), RNG(), 1.0f);
            CUDA_GL::mat4 testMat1(
                CUDA_GL::vec4(1.0f, 0.0f, 0.0f, 0.0f),
                CUDA_GL::vec4(0.0f, 1.0f, 0.0f, 0.0f),
                CUDA_GL::vec4(0.0f, 0.0f, 1.0f, 0.0f),
                CUDA_GL::vec4(RNG(), RNG(), RNG(), 1.0f)
            );
            testMatVec<CUDA_GL::mat4, CUDA_GL::vec4>(testMat1, val1);
            std::cout << std::endl;
        }
    }

    return 0;
}