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
        GLM_CUDA::vec3(1.23f),
        GLM_CUDA::vec3(7.0f, 8.0f, 9.0f)
    );

    GLM_CUDA::mat3 testMat2(
        GLM_CUDA::vec3(0.4f, 1.0f, 9.8f),
        GLM_CUDA::vec3(7.8f, 0.0f, 7.4f),
        GLM_CUDA::vec3(2.25f, 0.9f, 2.3f)
    );

    GLM_CUDA::mat3 matMul = testMat1 * testMat2;
    std::cout << matMul << std::endl;
    return 0;
}