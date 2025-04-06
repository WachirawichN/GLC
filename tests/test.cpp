#include <iostream>

#include <cuda_runtime.h>

#include "../GLM-CUDA/include/matrix.cuh"
#include "../GLM-CUDA/include/utility.cuh"
#include "../GLM-CUDA/include/vector.cuh"

int main()
{
    GLM_CUDA::vec3 val1(1.1f, 4.4f, 7.7f);

    GLM_CUDA::mat3 testMat1(
        val1,
        GLM_CUDA::vec3(4.0f, 5.0f, 6.0f),
        GLM_CUDA::vec3(7.0f, 8.0f, 9.0f)
    );
    std::cout << testMat1 << std::endl;
    std::cout << GLM_CUDA::transpose(testMat1) << std::endl;
    return 0;
}