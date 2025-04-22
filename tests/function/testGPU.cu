#include <iostream>

#include "matrix.cuh"
#include "utility.cuh"
#include "vector.cuh"

__global__ void testKernel(CUDA_GL::mat4* translationMat, CUDA_GL::mat4* scalingMat, CUDA_GL::mat4* rotationMat, CUDA_GL::vec3* vecPow)
{
    translationMat[0] = CUDA_GL::translate(CUDA_GL::vec3(1.0f, 2.0f, 3.0f));
    scalingMat[0] = CUDA_GL::scale(CUDA_GL::vec3(11.0f, 2.0f, 9.0f));
    rotationMat[0] = CUDA_GL::rotate(45, CUDA_GL::vec3(1.0f, 2.0f, 0.0f));
    vecPow[0] = CUDA_GL::pow(CUDA_GL::vec3(2.0f, 3.0f, 4.0f), 2.0f);
}

int main()
{
    CUDA_GL::mat4 h_translation;
    CUDA_GL::mat4 h_scaling;
    CUDA_GL::mat4 h_rotation;
    CUDA_GL::vec3 h_vecPow;

    CUDA_GL::mat4* d_translation;
    CUDA_GL::mat4* d_scaling;
    CUDA_GL::mat4* d_rotation;
    CUDA_GL::vec3* d_vecPow;

    cudaMalloc((void**)&d_translation, sizeof(CUDA_GL::mat4));
    cudaMalloc((void**)&d_scaling, sizeof(CUDA_GL::mat4));
    cudaMalloc((void**)&d_rotation, sizeof(CUDA_GL::mat4));
    cudaMalloc((void**)&d_vecPow, sizeof(CUDA_GL::vec3));

    testKernel<<<1, 1>>>(d_translation, d_scaling, d_rotation, d_vecPow);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_translation, d_translation, sizeof(CUDA_GL::mat4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_scaling, d_scaling, sizeof(CUDA_GL::mat4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_rotation, d_rotation, sizeof(CUDA_GL::mat4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vecPow, d_vecPow, sizeof(CUDA_GL::vec3), cudaMemcpyDeviceToHost);

    cudaFree(d_translation);
    cudaFree(d_scaling);
    cudaFree(d_rotation);
    cudaFree(d_vecPow);

    std::cout << h_translation << std::endl;
    std::cout << h_scaling << std::endl;
    std::cout << h_rotation << std::endl;
    std::cout << h_vecPow << std::endl;

    return 0;
}