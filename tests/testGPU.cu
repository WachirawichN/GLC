#include <iostream>
#include <cstdlib>
#include <chrono>

#include <curand_kernel.h>

#include "../CUDA-GL/include/matrix.cuh"
#include "../CUDA-GL/include/utility.cuh"
#include "../CUDA-GL/include/vector.cuh"

#include "../CUDA-GL/include/experimental.cuh"

__host__ float cpuRNG() {return ((rand() % 20001) - 10000.0f) / 1000.0f;}

__device__ int threadID()
{
    int blockId = 
        blockIdx.x +
        blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y
    ;
    int threadOffset = blockId * blockDim.x * blockDim.y * blockDim.z;
    int threadId = 
        threadIdx.x + 
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y
    ;
    return threadOffset + threadId;
}
__global__ void gpuMatricesInit(CUDA_GL::mat4* a, CUDA_GL::mat4* b, curandState* states)
{
    int id = threadID();
    for (int column = 0; column < 4; column++)
    {
        for (int row = 0; row < 4; row++)
        {
            a[id] = curand_uniform(&states[id]);
            b[id] = curand_uniform(&states[id]);
        }
    }
}
__global__ void gpuRNGInit(curandState* states, unsigned long seed)
{
    int id = threadID();
    curand_init(seed, id, 0, &states[id]);
}

int main()
{
    size_t totalMatrices = 1000000;
    std::cout << "Total matrices: " << totalMatrices << std::endl;

    std::srand(1);
    
    CUDA_GL::mat4* h_a = new CUDA_GL::mat4[totalMatrices];
    CUDA_GL::mat4* h_b = new CUDA_GL::mat4[totalMatrices];
    CUDA_GL::mat4* h_c = new CUDA_GL::mat4[totalMatrices];

    // Initialize matrices values
    std::cout << "Initializing matrices using: ";
    auto initializeTime0 = std::chrono::high_resolution_clock::now();
    {
        std::cout << "GPU" << std::endl;

        dim3 gridSize(10, 10, 10); // Thousand blocks
        dim3 blockSize(10, 10, 10); // Thousand threads
        
        std::cout << "Initializing states" << std::endl;
        curandState* d_states;
        cudaMalloc(&d_states, totalMatrices * sizeof(curandState));
        gpuRNGInit<<<gridSize, blockSize>>>(d_states, 0);

        std::cout << "Finish initializing states, initializing values" << std::endl;
        size_t byteSize = sizeof(CUDA_GL::mat4) * totalMatrices;
        CUDA_GL::mat4* d_a;
        CUDA_GL::mat4* d_b;
        cudaMalloc((void**)&d_a, byteSize);
        cudaMalloc((void**)&d_b, byteSize);
        gpuMatricesInit<<<gridSize, blockSize>>>(d_a, d_b, d_states);
        cudaMemcpy(h_a, d_a, byteSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b, d_b, byteSize, cudaMemcpyDeviceToHost);

        cudaFree(d_states);
        cudaFree(d_a);
        cudaFree(d_b);
    }
    auto initializeTime1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> initializeTime = initializeTime1 - initializeTime0;
    std::cout << "Variable have been set using " << initializeTime << " ms, executing CPU base line" << std::endl;

    // CPU base line
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < totalMatrices; i++)
        {
            h_c[i] = h_a[i] * h_b[i];
        }

        auto finishTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> usageTimeMs = finishTime - startTime;
        std::cout << "CPU usage time: " << usageTimeMs.count() << " ms" << std::endl;
    }

    // GPU execution
    /*
    {
        dim3 blockPerGrid(std::cbrt(totalMatrices));
        dim3 threadPerBlock(4, 4, 1);

        auto startTime = std::chrono::high_resolution_clock::now();

        //CUDA_GL::experimental::allOutGPUMatMul<<<blockPerGrid, threadPerBlock>>>();

        auto finishTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> usageTimeMs = finishTime - startTime;
        std::cout << "All out GPU usage time: " << usageTimeMs.count() << " ms" << std::endl;
    }
    */
    /*
    {
        // Setting up step
        dim3 gridSize(10, 10, 10); // Thousand blocks
        dim3 blockSize(10, 10, 10); // Thousand threads

        size_t byteSize = sizeof(CUDA_GL::mat4) * totalMatrices;
        CUDA_GL::mat4* d_a;
        CUDA_GL::mat4* d_b;
        CUDA_GL::mat4* d_c;
        cudaMalloc((void**)&d_a, byteSize);
        cudaMalloc((void**)&d_b, byteSize);
        cudaMalloc((void**)&d_c, byteSize);
        cudaMemcpy(d_a, h_a, byteSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, byteSize, cudaMemcpyHostToDevice);

        // Execution step
        auto startTime = std::chrono::high_resolution_clock::now();
        
        CUDA_GL::experimental::partialGPUMatMul<<<gridSize, blockSize>>>(d_a, d_b, d_c);

        auto finishTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> usageTimeMs = finishTime - startTime;
        std::cout << "GPU usage time: " << usageTimeMs.count() << " ms" << std::endl;

        // Verification step
        cudaMemcpy(h_c, d_c, byteSize, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    */
    {
        // Setting up step
        dim3 gridSize(10, 10, 10); // Thousand blocks
        dim3 blockSize(10, 10, 10); // Thousand threads

        size_t byteSize = sizeof(CUDA_GL::mat4) * totalMatrices;
        CUDA_GL::mat4* d_a;
        CUDA_GL::mat4* d_b;
        CUDA_GL::mat4* d_c;
        cudaMalloc((void**)&d_a, byteSize);
        cudaMalloc((void**)&d_b, byteSize);
        cudaMalloc((void**)&d_c, byteSize);
        cudaMemcpy(d_a, h_a, byteSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, byteSize, cudaMemcpyHostToDevice);

        // Execution step
        auto startTime = std::chrono::high_resolution_clock::now();
        
        CUDA_GL::experimental::gpuMatMul<<<gridSize, blockSize>>>(d_a, d_b, d_c);

        auto finishTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> usageTimeMs = finishTime - startTime;
        std::cout << "GPU usage time: " << usageTimeMs.count() << " ms" << std::endl;

        // Verification step
        cudaMemcpy(h_c, d_c, byteSize, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    h_a = 0;
    h_b = 0;
    h_c = 0;

    return 0;
}