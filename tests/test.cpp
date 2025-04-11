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

template <typename T>
void testVec(T& vector1, T& vector2)
{
    std::cout << "Vector 1: " << vector1 << std::endl;
    std::cout << "Vector 2: " << vector2 << std::endl;
    {
        std::cout << "Scalar addition" << std::endl;
        std::cout << vector1 + 1.0f << std::endl;
        vector1 += 1.0f;
        std::cout << vector1 << std::endl;
    }
    {
        std::cout << "Scalar subtraction" << std::endl;
        std::cout << vector1 - 1.0f << std::endl;
        vector1 -= 1.0f;
        std::cout << vector1 << std::endl;
    }
    {
        std::cout << "Scalar multiplication" << std::endl;
        std::cout << vector1 * 2.0f << std::endl;
        vector1 *= 2.0f;
        std::cout << vector1 << std::endl;
    }
    {
        std::cout << "Scalar division" << std::endl;
        std::cout << vector1 / 2.0f << std::endl;
        vector1 /= 2.0f;
        std::cout << vector1 << std::endl;
    }
    {
        std::cout << "Vector addition" << std::endl;
        std::cout << vector1 + vector2 << std::endl;
        vector1 += vector2;
        std::cout << vector1 << std::endl;
    }
    {
        std::cout << "Vector subtraction" << std::endl;
        std::cout << vector1 - vector2 << std::endl;
        vector1 -= vector2;
        std::cout << vector1 << std::endl << std::endl;
    }
}

float RNG()
{
    return ((rand() % 20001) - 10000.0f) / 1000.0f;
}

int main()
{
    std::srand(1);

    std::cout << "2D Vector" << std::endl;
    {
        GLM_CUDA::vec2 vec1(RNG(), RNG());
        GLM_CUDA::vec2 vec2(RNG(), RNG());
        testVec<GLM_CUDA::vec2>(vec1, vec2);
    }
    
    std::cout << "3D Vector" << std::endl;
    {
        GLM_CUDA::vec3 vec1(RNG(), RNG(), RNG());
        GLM_CUDA::vec3 vec2(RNG(), RNG(), RNG());
        testVec<GLM_CUDA::vec3>(vec1, vec2);
    }

    std::cout << "4D Vector" << std::endl;
    {
        GLM_CUDA::vec4 vec1(RNG(), RNG(), RNG(), RNG());
        GLM_CUDA::vec4 vec2(RNG(), RNG(), RNG(), RNG());
        testVec<GLM_CUDA::vec4>(vec1, vec2);
    }
    return 0;
}