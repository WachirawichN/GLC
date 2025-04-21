#include <iostream>

#include <stdio.h>

#include "../CUDA-GL/include/matrix.cuh"
#include "../CUDA-GL/include/utility.cuh"
#include "../CUDA-GL/include/vector.cuh"

#include "../CUDA-GL/include/experimental.cuh"

__global__ void testKernel(CUDA_GL::vec2* vec2, CUDA_GL::vec3* vec3, CUDA_GL::vec4* vec4, float* length)
{
    length[0] = CUDA_GL::length(vec2[0]);
    length[1] = CUDA_GL::length(vec3[0]);
    length[2] = CUDA_GL::length(vec4[0]);
}

int main()
{
    CUDA_GL::vec2 h_vector2(2.0f, 10.0f);
    CUDA_GL::vec3 h_vector3(1.0f, 2.0f, 3.0f);
    CUDA_GL::vec4 h_vector4(8.2f, 9.7f, 34.0f, 90.0f);
    float h_length[3];

    {
        CUDA_GL::vec2* d_vector2;
        CUDA_GL::vec3* d_vector3;
        CUDA_GL::vec4* d_vector4;
        float* d_length;

        cudaMalloc((void**)&d_vector2, sizeof(CUDA_GL::vec2));
        cudaMalloc((void**)&d_vector3, sizeof(CUDA_GL::vec3));
        cudaMalloc((void**)&d_vector4, sizeof(CUDA_GL::vec4));
        cudaMalloc((void**)&d_length, sizeof(float) * 3);

        cudaMemcpy(d_vector2, &h_vector2, sizeof(CUDA_GL::vec2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vector3, &h_vector3, sizeof(CUDA_GL::vec3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vector4, &h_vector4, sizeof(CUDA_GL::vec4), cudaMemcpyHostToDevice);
        
        testKernel<<<1, 1>>>(d_vector2, d_vector3, d_vector4, d_length);
        cudaDeviceSynchronize();

        cudaMemcpy(h_length, d_length, sizeof(float) * 3, cudaMemcpyDeviceToHost);

        cudaFree(d_vector2);
        cudaFree(d_vector3);
        cudaFree(d_vector4);
        cudaFree(d_length);
    }

    std::cout << h_length[0] << ", " << h_length[1] << ", " << h_length[2] << std::endl;

    return 0;
}