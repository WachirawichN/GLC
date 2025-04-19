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
    CUDA_GL::vec4* testVec = new CUDA_GL::vec4[10];
    CUDA_GL::mat4 testMat;
    std::cout << sizeof(testVec) << std::endl;
    std::cout << sizeof(testMat) << std::endl;

    return 0;
}