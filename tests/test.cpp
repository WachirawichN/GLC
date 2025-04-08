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
void testVec(T& vector)
{
    std::cout << vector << std::endl;
    {
        std::cout << "Scalar addition" << std::endl;
        std::cout << vector + 1.0f << std::endl;
        vector += 1.0f;
        std::cout << vector << std::endl;
    }
    {
        std::cout << "Scalar subtraction" << std::endl;
        std::cout << vector - 1.0f << std::endl;
        vector -= 1.0f;
        std::cout << vector << std::endl << std::endl;
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
        GLM_CUDA::vec2 vector(RNG(), RNG());
        testVec(vector);
    }
    
    std::cout << "3D Vector" << std::endl;
    {
        GLM_CUDA::vec3 vector(RNG(), RNG(), RNG());
        testVec(vector);
    }

    std::cout << "4D Vector" << std::endl;
    {
        GLM_CUDA::vec4 vector(RNG(), RNG(), RNG(), RNG());
        testVec(vector);
    }
    return 0;
}