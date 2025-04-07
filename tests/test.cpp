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

float RNG()
{
    return ((rand() % 20001) - 10000.0f) / 1000.0f;
}

int main()
{
    std::srand(1);

    std::cout << "2x2 Matrix" << std::endl;
    {
        GLM_CUDA::vec2 val1(RNG(), RNG());
        GLM_CUDA::mat2 testMat1(
            val1,
            GLM_CUDA::vec2(RNG(), RNG())
        );
        GLM_CUDA::mat2 testMat2(
            GLM_CUDA::vec2(RNG(), RNG()),
            GLM_CUDA::vec2(RNG())
        );
        testMat<GLM_CUDA::mat2>(testMat1, testMat2);
    }
    
    std::cout << "3x3 Matrix" << std::endl;
    {
        GLM_CUDA::vec3 val1(RNG(), RNG(), RNG());

        GLM_CUDA::mat3 testMat1(
            val1,
            GLM_CUDA::vec3(RNG()),
            GLM_CUDA::vec3(RNG(), RNG(), RNG())
        );

        GLM_CUDA::mat3 testMat2(
            GLM_CUDA::vec3(RNG(), RNG(), RNG()),
            GLM_CUDA::vec3(RNG(), RNG(), RNG()),
            GLM_CUDA::vec3()
        );
        testMat<GLM_CUDA::mat3>(testMat1, testMat2);
    }

    std::cout << "4x4 Matrix" << std::endl;
    {
        GLM_CUDA::vec4 val1(RNG(), RNG(), RNG(), RNG());

        GLM_CUDA::mat4 testMat1(
            val1,
            GLM_CUDA::vec4(RNG()),
            GLM_CUDA::vec4(RNG(), RNG(), RNG(), RNG()),
            GLM_CUDA::vec4()
        );

        GLM_CUDA::mat4 testMat2(
            GLM_CUDA::vec4(RNG(), RNG(), RNG(), RNG()),
            GLM_CUDA::vec4(RNG(), RNG(), RNG(), RNG()),
            GLM_CUDA::vec4(RNG(), RNG(), RNG(), RNG()),
            GLM_CUDA::vec4(RNG(), RNG(), RNG(), RNG())
        );
        testMat<GLM_CUDA::mat4>(testMat1, testMat2);
    }
    return 0;
}