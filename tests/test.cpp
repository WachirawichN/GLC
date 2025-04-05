#include <iostream>

#include <cuda_runtime.h>

#include "../GLM-CUDA/include/matrix.cuh"
#include "../GLM-CUDA/include/utility.cuh"
#include "../GLM-CUDA/include/vector.cuh"

int main()
{
    GLM_CUDA::vec3 testVec1(1.0f);
    GLM_CUDA::vec3 testVec2(5.0f, 6.0f, 7.0f);

    testVec1 += testVec2;
    std::cout << testVec1 << std::endl << std::endl;

    testVec1 -= testVec2;
    std::cout << testVec1 << std::endl << std::endl;

    testVec1 *= 2.5f;
    std::cout << testVec1 << std::endl << std::endl;

    testVec1 /= 2.5f;
    std::cout << testVec1 << std::endl << std::endl;
    return 0;
}