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
    // Benchmark setting
    size_t totalMatrices = 1000000;
    dim3 gridSize(10, 10, 10);
    dim3 blockSize(10, 10, 10);
    int totalRepeat = 10;

    std::cout << "Initializing matrices" << std::endl;
    std::cout << "Total matrices: " << totalMatrices << std::endl;

    auto initializeTime0 = std::chrono::high_resolution_clock::now();
    CUDA_GL::mat4* h_a = new CUDA_GL::mat4[totalMatrices];
    CUDA_GL::mat4* h_b = new CUDA_GL::mat4[totalMatrices];
    CUDA_GL::mat4* h_c = new CUDA_GL::mat4[totalMatrices];
    auto initializeTime1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> initializeTime = initializeTime1 - initializeTime0;
    std::cout << "  Matrices initialized, usage time: " << initializeTime.count() << " ms" << std::endl;
    
    // Assigning matrices values
    std::cout << "Assigning matrices values using: ";
    auto assignmentTime0 = std::chrono::high_resolution_clock::now();
    {
        std::cout << "GPU" << std::endl;

        dim3 gridSize(10, 10, 10); // Thousand blocks
        dim3 blockSize(10, 10, 10); // Thousand threads
        
        std::cout << "  Initializing states" << std::endl;
        curandState* d_states;
        cudaMalloc(&d_states, totalMatrices * sizeof(curandState));
        gpuRNGInit<<<gridSize, blockSize>>>(d_states, 0);
        cudaDeviceSynchronize();

        std::cout << "      Finish initializing states, assigning values" << std::endl;
        size_t byteSize = sizeof(CUDA_GL::mat4) * totalMatrices;
        CUDA_GL::mat4* d_a;
        CUDA_GL::mat4* d_b;
        cudaMalloc((void**)&d_a, byteSize);
        cudaMalloc((void**)&d_b, byteSize);
        gpuMatricesInit<<<gridSize, blockSize>>>(d_a, d_b, d_states);
        cudaDeviceSynchronize();
        cudaMemcpy(h_a, d_a, byteSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b, d_b, byteSize, cudaMemcpyDeviceToHost);

        cudaFree(d_states);
        cudaFree(d_a);
        cudaFree(d_b);
    }
    auto assignmentTime1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> assignmentTime = assignmentTime1 - assignmentTime0;
    std::cout << "  Matrices's values have been assigned, usage time: " << assignmentTime.count() << " ms" << std::endl << std::endl;

    std::cout << "Benchmarking result: " << std::endl;
    // CPU base line
    {
        double totalTime = 0.0f;
        for (int i = 0; i < totalRepeat; i++)
        {
            auto startTime = std::chrono::high_resolution_clock::now();
    
            for (size_t j = 0; j < totalMatrices; j++)
            {
                h_c[j] = h_a[j] * h_b[j];
            }
    
            auto finishTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> usageTime = finishTime - startTime;
            totalTime += usageTime.count();
        }
        std::cout << "  AVG CPU usage time: " << totalTime / totalRepeat << " ms" << std::endl;
    }
    // GPU execution
    /*
    {
        dim3 blockPerGrid(std::cbrt(totalMatrices));
        dim3 threadPerBlock(4, 4, 1);

        auto startTime = std::chrono::high_resolution_clock::now();

        //CUDA_GL::experimental::allOutGPUMatMul<<<blockPerGrid, threadPerBlock>>>();
        cudaDeviceSynchronize();

        auto finishTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> usageTime = finishTime - startTime;
        std::cout << "All out GPU usage time: " << usageTime.count() << " ms" << std::endl;
    }
    */
    {
        double totalTime = 0.0f;

        // Setting up step
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
        for (int i = 0; i < totalRepeat; i++)
        {
            // Execution step
            auto startTime = std::chrono::high_resolution_clock::now();
            
            CUDA_GL::experimental::partialGPUMatMul<<<gridSize, blockSize>>>(d_a, d_b, d_c);
            cudaDeviceSynchronize();

            auto finishTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> usageTime = finishTime - startTime;
            totalTime += usageTime.count();
        }
        
        // Verification step
        cudaMemcpy(h_c, d_c, byteSize, cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        
        std::cout << "  AVG Partial GPU usage time: " << totalTime / totalRepeat << " ms" << std::endl;
    }
    {
        // BEST RESULT
        double totalTime = 0.0f;

        // Setting up step
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
        for (int i = 0; i < totalRepeat; i++)
        {
            auto startTime = std::chrono::high_resolution_clock::now();
            
            CUDA_GL::experimental::gpuMatMul<<<gridSize, blockSize>>>(d_a, d_b, d_c);
            cudaDeviceSynchronize();
    
            auto finishTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> usageTime = finishTime - startTime;
            totalTime += usageTime.count();
        }
        
        // Verification step
        cudaMemcpy(h_c, d_c, byteSize, cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        std::cout << "  AVG GPU usage time: " << totalTime / totalRepeat << " ms" << std::endl;
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    h_a = 0;
    h_b = 0;
    h_c = 0;

    return 0;
}