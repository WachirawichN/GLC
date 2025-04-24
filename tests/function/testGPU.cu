#include <iostream>

#include "matrix.cuh"
#include "utility.cuh"
#include "vector.cuh"

__global__ void unpackKernel(CUDA_GL::vec4* vec, CUDA_GL::mat4* mat, float* unpackedVec, float* unpackedMat)
{
    float* vecValues = CUDA_GL::unpack(CUDA_GL::pow(vec[0], 2.0f));
    for (int i = 0; i < (int)(sizeof(CUDA_GL::vec4) / sizeof(float)); i++)
    {
        unpackedVec[i] = vecValues[i];
    }
    float* matValues = CUDA_GL::unpack(CUDA_GL::pow(mat[0], 2.0f));
    for (int i = 0; i < (int)(sizeof(CUDA_GL::mat4) / sizeof(float)); i++)
    {
        unpackedMat[i] = matValues[i];
    }    
}
__global__ void transformKernel(CUDA_GL::mat4* scaledMatrix, CUDA_GL::mat4* translatedMatrix, CUDA_GL::mat4* rotatedMatrix)
{
    scaledMatrix[0] = CUDA_GL::scale(3.0f, CUDA_GL::mat4(1.0f));
    translatedMatrix[0] = CUDA_GL::translate(CUDA_GL::vec3(1.0f, 2.0f, 3.0f), scaledMatrix[0]);
    rotatedMatrix[0] = CUDA_GL::rotate(1.0f, CUDA_GL::vec3(1.0f, 2.0f, 1.0f), translatedMatrix[0]);
}

int main()
{
    {
        CUDA_GL::vec4 h_vec(1.0f, 2.0f, 3.0f, 4.0f);
        CUDA_GL::mat4 h_mat(
            CUDA_GL::vec4(0.0f, 0.1f, 0.2f, 0.3f),
            CUDA_GL::vec4(0.4f, 0.5f, 0.6f, 0.7f),
            CUDA_GL::vec4(0.8f, 0.9f, 1.0f, 1.1f),
            CUDA_GL::vec4(1.2f, 1.3f, 1.4f, 1.5f)
        );
        float* h_unpackedVec = new float[sizeof(CUDA_GL::vec4) / sizeof(float)];
        float* h_unpackedMat = new float[sizeof(CUDA_GL::mat4) / sizeof(float)];

        CUDA_GL::vec4* d_vec;
        CUDA_GL::mat4* d_mat;
        float* d_unpackedVec;
        float* d_unpackedMat;
        cudaMalloc((void**)&d_vec, sizeof(CUDA_GL::vec4));
        cudaMalloc((void**)&d_mat, sizeof(CUDA_GL::mat4));
        cudaMalloc((void**)&d_unpackedVec, sizeof(CUDA_GL::vec4));
        cudaMalloc((void**)&d_unpackedMat, sizeof(CUDA_GL::mat4));

        cudaMemcpy(d_vec, &h_vec, sizeof(CUDA_GL::vec4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mat, &h_mat, sizeof(CUDA_GL::mat4), cudaMemcpyHostToDevice);
        
        unpackKernel<<<1, 1>>>(d_vec, d_mat, d_unpackedVec, d_unpackedMat);
        cudaDeviceSynchronize();

        cudaMemcpy(h_unpackedVec, d_unpackedVec, sizeof(CUDA_GL::vec4), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_unpackedMat, d_unpackedMat, sizeof(CUDA_GL::mat4), cudaMemcpyDeviceToHost);
        
        cudaFree(d_vec);
        cudaFree(d_mat);
        cudaFree(d_unpackedVec);
        cudaFree(d_unpackedMat);

        for (int i = 0; i < (int)(sizeof(CUDA_GL::vec4) / sizeof(float)); i++)
        {
            std::cout << h_unpackedVec[i] << " ";
        }
        std::cout << std::endl;
        for (int i = 0; i < (int)(sizeof(CUDA_GL::mat4) / sizeof(float)); i++)
        {
            std::cout << h_unpackedMat[i] << " ";
        }
        std::cout << std::endl;

        delete[] h_unpackedVec;
        h_unpackedVec = NULL;
        delete[] h_unpackedMat;
        h_unpackedMat = NULL;
    }
    {
        CUDA_GL::mat4* h_scaledMat = new CUDA_GL::mat4[1];
        CUDA_GL::mat4* h_translatedMat = new CUDA_GL::mat4[1];
        CUDA_GL::mat4* h_rotatedMat = new CUDA_GL::mat4[1];

        CUDA_GL::mat4* d_scaledMat;
        CUDA_GL::mat4* d_translatedMat;
        CUDA_GL::mat4* d_rotatedMat;

        cudaMalloc((void**)&d_scaledMat, sizeof(CUDA_GL::mat4));
        cudaMalloc((void**)&d_translatedMat, sizeof(CUDA_GL::mat4));
        cudaMalloc((void**)&d_rotatedMat, sizeof(CUDA_GL::mat4));

        transformKernel<<<1, 1>>>(d_scaledMat, d_translatedMat, d_rotatedMat);
        cudaDeviceSynchronize();

        cudaMemcpy(h_scaledMat, d_scaledMat, sizeof(CUDA_GL::mat4), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_translatedMat, d_translatedMat, sizeof(CUDA_GL::mat4), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rotatedMat, d_rotatedMat, sizeof(CUDA_GL::mat4), cudaMemcpyDeviceToHost);

        cudaFree(d_scaledMat);
        cudaFree(d_translatedMat);
        cudaFree(d_rotatedMat);

        std::cout << *h_scaledMat << std::endl;
        std::cout << *h_translatedMat << std::endl;
        std::cout << *h_rotatedMat << std::endl;

        delete[] h_scaledMat;
        h_scaledMat = NULL;
        delete[] h_translatedMat;
        h_translatedMat = NULL;
        delete[] h_rotatedMat;
        h_rotatedMat = NULL;
    }
    return 0;
}