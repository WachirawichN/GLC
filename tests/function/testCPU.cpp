#include <iostream>

#include <cuda_runtime.h>

#include "matrix.cuh"
#include "utility.cuh"
#include "vector.cuh"

int main()
{
    CUDA_GL::vec4 testVector(1.0f, 2.0f, 3.0f, 4.0f);
    testVector = CUDA_GL::pow(testVector, 2.0f);
    float* unpackedVec = CUDA_GL::unpack(testVector);
    for (int i = 0; i < (int)(sizeof(testVector) / sizeof(float)); i++)
    {
        std::cout << unpackedVec[i] << " ";
    }
    std::cout << std::endl;

    CUDA_GL::mat4 testMatrix(
        CUDA_GL::vec4(0.0f, 0.1f, 0.2f, 0.3f),
        CUDA_GL::vec4(0.4f, 0.5f, 0.6f, 0.7f),
        CUDA_GL::vec4(0.8f, 0.9f, 1.0f, 1.1f),
        CUDA_GL::vec4(1.2f, 1.3f, 1.4f, 1.5f)
    );
    testMatrix = CUDA_GL::pow(testMatrix, 2.0f);
    float* unpackedMat = CUDA_GL::unpack(testMatrix);
    for (int i = 0; i < (int)(sizeof(testMatrix) / sizeof(float)); i++)
    {
        std::cout << unpackedMat[i] << " ";
    }
    std::cout << std::endl;

    CUDA_GL::mat4 tranformMat(1.0f);
    tranformMat = CUDA_GL::scale(3.0f, tranformMat);
    std::cout << tranformMat << std::endl;
    tranformMat = CUDA_GL::translate(CUDA_GL::vec3(1.0f, 2.0f, 3.0f), tranformMat);
    std::cout << tranformMat << std::endl;
    tranformMat = CUDA_GL::rotate(1.0f, CUDA_GL::vec3(1.0f, 2.0f, 1.0f), tranformMat);
    std::cout << tranformMat << std::endl;

    return 0;
}