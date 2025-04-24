#include <iostream>

#include <cuda_runtime.h>

#include "matrix.cuh"
#include "utility.cuh"
#include "vector.cuh"

int main()
{
    CUDA_GL::vec3 testVector(1.0f, 2.0f, 3.0f);
    float* vectorValues = CUDA_GL::unpack(testVector);
    std::cout << "Total values: " << sizeof(testVector) / sizeof(float) << ", Vector values: ";
    for (int i = 0; i < sizeof(testVector) / sizeof(float); i++)
    {
        std::cout << vectorValues[i] << ", ";
    }
    std::cout << std::endl;

    //CUDA_GL::mat4 testMatrix(
    //    CUDA_GL::vec4(0.0f, 0.1f, 0.2f, 0.3f),
    //    CUDA_GL::vec4(0.4f, 0.5f, 0.6f, 0.7f),
    //    CUDA_GL::vec4(0.8f, 0.9f, 1.0f, 1.1f),
    //    CUDA_GL::vec4(1.2f, 1.3f, 1.4f, 1.5f)
    //);
    CUDA_GL::mat3 testMatrix(
        CUDA_GL::vec3(1.0f, 2.0f, 3.0f),
        CUDA_GL::vec3(4.0f, 5.0f, 6.0f),
        CUDA_GL::vec3(7.0f, 8.0f, 9.0f)
    );
    float* matrixValues = CUDA_GL::unpack(testMatrix);
    std::cout << "Total values: " << sizeof(testMatrix) / sizeof(float) << ", Matrix values: ";
    for (int i = 0; i < sizeof(testMatrix) / sizeof(float); i++)
    {
        std::cout << matrixValues[i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}