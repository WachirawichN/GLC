#include <iostream>

#include <GLC/GLC.cuh>

__global__ void unpackKernel(GLC::vec4* vec, GLC::mat4* mat, float* unpackedVec, float* unpackedMat)
{
    float* vecValues = GLC::unpack(GLC::pow(vec[0], 2.0f));
    for (int i = 0; i < (int)(sizeof(GLC::vec4) / sizeof(float)); i++)
    {
        unpackedVec[i] = vecValues[i];
    }
    float* matValues = GLC::unpack(GLC::pow(mat[0], 2.0f));
    for (int i = 0; i < (int)(sizeof(GLC::mat4) / sizeof(float)); i++)
    {
        unpackedMat[i] = matValues[i];
    }    
}
__global__ void transformKernel(GLC::mat4* scaledMatrix, GLC::mat4* translatedMatrix, GLC::mat4* rotatedMatrix)
{
    scaledMatrix[0] = GLC::scale(3.0f, GLC::mat4(1.0f));
    translatedMatrix[0] = GLC::translate(GLC::vec3(1.0f, 2.0f, 3.0f), scaledMatrix[0]);
    rotatedMatrix[0] = GLC::rotate(1.0f, GLC::vec3(1.0f, 2.0f, 1.0f), translatedMatrix[0]);
}
__global__ void radiansKernel(float* radians)
{
    int id = GLC::threadID();
    radians[id] = GLC::radians(id * 10.0f);
}

int main()
{
    {
        GLC::vec4 h_vec(1.0f, 2.0f, 3.0f, 4.0f);
        GLC::mat4 h_mat(
            GLC::vec4(0.0f, 0.1f, 0.2f, 0.3f),
            GLC::vec4(0.4f, 0.5f, 0.6f, 0.7f),
            GLC::vec4(0.8f, 0.9f, 1.0f, 1.1f),
            GLC::vec4(1.2f, 1.3f, 1.4f, 1.5f)
        );
        float* h_unpackedVec = new float[sizeof(GLC::vec4) / sizeof(float)];
        float* h_unpackedMat = new float[sizeof(GLC::mat4) / sizeof(float)];

        GLC::vec4* d_vec;
        GLC::mat4* d_mat;
        float* d_unpackedVec;
        float* d_unpackedMat;
        cudaMalloc((void**)&d_vec, sizeof(GLC::vec4));
        cudaMalloc((void**)&d_mat, sizeof(GLC::mat4));
        cudaMalloc((void**)&d_unpackedVec, sizeof(GLC::vec4));
        cudaMalloc((void**)&d_unpackedMat, sizeof(GLC::mat4));

        cudaMemcpy(d_vec, &h_vec, sizeof(GLC::vec4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mat, &h_mat, sizeof(GLC::mat4), cudaMemcpyHostToDevice);
        
        unpackKernel<<<1, 1>>>(d_vec, d_mat, d_unpackedVec, d_unpackedMat);
        cudaDeviceSynchronize();

        cudaMemcpy(h_unpackedVec, d_unpackedVec, sizeof(GLC::vec4), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_unpackedMat, d_unpackedMat, sizeof(GLC::mat4), cudaMemcpyDeviceToHost);
        
        cudaFree(d_vec);
        cudaFree(d_mat);
        cudaFree(d_unpackedVec);
        cudaFree(d_unpackedMat);

        for (int i = 0; i < (int)(sizeof(GLC::vec4) / sizeof(float)); i++)
        {
            std::cout << h_unpackedVec[i] << " ";
        }
        std::cout << std::endl;
        for (int i = 0; i < (int)(sizeof(GLC::mat4) / sizeof(float)); i++)
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
        GLC::mat4* h_scaledMat = new GLC::mat4[1];
        GLC::mat4* h_translatedMat = new GLC::mat4[1];
        GLC::mat4* h_rotatedMat = new GLC::mat4[1];

        GLC::mat4* d_scaledMat;
        GLC::mat4* d_translatedMat;
        GLC::mat4* d_rotatedMat;

        cudaMalloc((void**)&d_scaledMat, sizeof(GLC::mat4));
        cudaMalloc((void**)&d_translatedMat, sizeof(GLC::mat4));
        cudaMalloc((void**)&d_rotatedMat, sizeof(GLC::mat4));

        transformKernel<<<1, 1>>>(d_scaledMat, d_translatedMat, d_rotatedMat);
        cudaDeviceSynchronize();

        cudaMemcpy(h_scaledMat, d_scaledMat, sizeof(GLC::mat4), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_translatedMat, d_translatedMat, sizeof(GLC::mat4), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rotatedMat, d_rotatedMat, sizeof(GLC::mat4), cudaMemcpyDeviceToHost);

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
    {
        int size = 10;
        float* h_radians = new float[size];

        float* d_radians;
        cudaMalloc((void**)&d_radians, sizeof(float) * size);
        
        radiansKernel<<<1, size>>>(d_radians);
        cudaDeviceSynchronize();

        cudaMemcpy(h_radians, d_radians, sizeof(float) * size, cudaMemcpyDeviceToHost);
        cudaFree(d_radians);

        for (int i = 0; i < size; i++)
        {
            std::cout << h_radians[i] << " ";
        }
        std::cout << std::endl;

        delete[] h_radians;
        h_radians = NULL;
    }
    return 0;
}