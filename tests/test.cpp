#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

#include "../GLM-CUDA/include/matrix.cuh"
#include "../GLM-CUDA/include/utility.cuh"
#include "../GLM-CUDA/include/vector.cuh"

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
            GLM_CUDA::vec2 val1(RNG(), RNG());
            GLM_CUDA::mat2 testMat1(
                GLM_CUDA::vec2(RNG(), RNG()),
                GLM_CUDA::vec2(RNG(), RNG())
            );
            testMatVec<GLM_CUDA::mat2, GLM_CUDA::vec2>(testMat1, val1);
            std::cout << std::endl;
        }
        
        std::cout << "3x3 Random matrix x Random 3D vector" << std::endl;
        {
            GLM_CUDA::vec3 val1(RNG(), RNG(), RNG());
            GLM_CUDA::mat3 testMat1(
                GLM_CUDA::vec3(RNG(), RNG(), RNG()),
                GLM_CUDA::vec3(RNG(), RNG(), RNG()),
                GLM_CUDA::vec3(RNG(), RNG(), RNG())
            );
            testMatVec<GLM_CUDA::mat3, GLM_CUDA::vec3>(testMat1, val1);
            std::cout << std::endl;
        }

        std::cout << "4x4 Random matrix x Random 4D vector" << std::endl;
        {
            GLM_CUDA::vec4 val1(RNG(), RNG(), RNG(), RNG());
            GLM_CUDA::mat4 testMat1(
                GLM_CUDA::vec4(RNG(), RNG(), RNG(), RNG()),
                GLM_CUDA::vec4(RNG(), RNG(), RNG(), RNG()),
                GLM_CUDA::vec4(RNG(), RNG(), RNG(), RNG()),
                GLM_CUDA::vec4(RNG(), RNG(), RNG(), RNG())
            );
            testMatVec<GLM_CUDA::mat4, GLM_CUDA::vec4>(testMat1, val1);
            std::cout << std::endl;
        }
    }
    
    // Identity matrix
    {
        std::cout << "2x2 Identity matrix x Random 2D vector" << std::endl;
        {
            GLM_CUDA::vec2 val1(RNG(), RNG());
            GLM_CUDA::mat2 testMat1(
                GLM_CUDA::vec2(1.0f, 0.0f),
                GLM_CUDA::vec2(0.0f, 1.0f)
            );
            testMatVec<GLM_CUDA::mat2, GLM_CUDA::vec2>(testMat1, val1);
            std::cout << std::endl;
        }
        
        std::cout << "3x3 Identity matrix x Random 3D vector" << std::endl;
        {
            GLM_CUDA::vec3 val1(RNG(), RNG(), RNG());
            GLM_CUDA::mat3 testMat1(
                GLM_CUDA::vec3(1.0f, 0.0f, 0.0f),
                GLM_CUDA::vec3(0.0f, 1.0f, 0.0f),
                GLM_CUDA::vec3(0.0f, 0.0f, 1.0f)
            );
            testMatVec<GLM_CUDA::mat3, GLM_CUDA::vec3>(testMat1, val1);
            std::cout << std::endl;
        }

        std::cout << "4x4 Identity matrix x Random 4D vector" << std::endl;
        {
            GLM_CUDA::vec4 val1(RNG(), RNG(), RNG(), RNG());
            GLM_CUDA::mat4 testMat1(
                GLM_CUDA::vec4(1.0f, 0.0f, 0.0f, 0.0f),
                GLM_CUDA::vec4(0.0f, 1.0f, 0.0f, 0.0f),
                GLM_CUDA::vec4(0.0f, 0.0f, 1.0f, 0.0f),
                GLM_CUDA::vec4(0.0f, 0.0f, 0.0f, 1.0f)
            );
            testMatVec<GLM_CUDA::mat4, GLM_CUDA::vec4>(testMat1, val1);
            std::cout << std::endl;
        }
    }

    // Use case identity matrix
    {
        std::cout << "Use case 2x2 Identity matrix x Random 2D vector" << std::endl;
        {
            GLM_CUDA::vec2 val1(RNG(), 1.0f);
            GLM_CUDA::mat2 testMat1(
                GLM_CUDA::vec2(1.0f, 0.0f),
                GLM_CUDA::vec2(RNG(), 1.0f)
            );
            testMatVec<GLM_CUDA::mat2, GLM_CUDA::vec2>(testMat1, val1);
            std::cout << std::endl;
        }
        
        std::cout << "Use case 3x3 Identity matrix x Random 3D vector" << std::endl;
        {
            GLM_CUDA::vec3 val1(RNG(), RNG(), 1.0f);
            GLM_CUDA::mat3 testMat1(
                GLM_CUDA::vec3(1.0f, 0.0f, 0.0f),
                GLM_CUDA::vec3(0.0f, 1.0f, 0.0f),
                GLM_CUDA::vec3(RNG(), RNG(), 1.0f)
            );
            testMatVec<GLM_CUDA::mat3, GLM_CUDA::vec3>(testMat1, val1);
            std::cout << std::endl;
        }

        std::cout << "Use case 4x4 Identity matrix x Random 4D vector" << std::endl;
        {
            GLM_CUDA::vec4 val1(RNG(), RNG(), RNG(), 1.0f);
            GLM_CUDA::mat4 testMat1(
                GLM_CUDA::vec4(1.0f, 0.0f, 0.0f, 0.0f),
                GLM_CUDA::vec4(0.0f, 1.0f, 0.0f, 0.0f),
                GLM_CUDA::vec4(0.0f, 0.0f, 1.0f, 0.0f),
                GLM_CUDA::vec4(RNG(), RNG(), RNG(), 1.0f)
            );
            testMatVec<GLM_CUDA::mat4, GLM_CUDA::vec4>(testMat1, val1);
            std::cout << std::endl;
        }
    }


    return 0;
}