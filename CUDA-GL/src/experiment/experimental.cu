#include "../../include/experimental.cuh"

__global__ void gpuDot(CUDA_GL::mat4* a, CUDA_GL::mat4* b, CUDA_GL::mat4* c, int matrixID)
{
    int row = threadIdx.y;
    int column = threadIdx.x;
    CUDA_GL::vec4 aVec = CUDA_GL::transpose(a[matrixID])[row];
    CUDA_GL::vec4 bVec = b[matrixID][column];
    c[matrixID][column][row] = CUDA_GL::dot(aVec, bVec);
}

namespace CUDA_GL
{
    namespace experimental
    {
        __global__ void allOutGPUMatMul()
        {
            // Utilize dynamic parallelism to the EXTREAM
            // Workflow:
            // 1. Each thread (Output thread) call a number of threads correlate to number of values in the output matrix (those thread are called "Control thread").
            // 2. Each Control thread then call a number of threads correlate to number of multiplication required in the dot product operation for two matrix in those two input (those thread are called "Multiplication thread").
            // 3. After all multiplication are finished, the Control thread then called a number of
        }
        __global__ void partialGPUMatMul(mat4* a, mat4* b, mat4* c)
        {
            // Utilize dynamic parallelism
            // Workflow:
            // 1. Each thread (Output thread) call a number of threads correlate to number of values in the output matrix (those thread are called "Control thread").
            // 2. Each Control thread then perform a dot product operation to both of the vectors needed to perform the output.
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
            int id = threadOffset + threadId;
            dim3 blockSize(4, 4, 1);
            gpuDot<<<1, blockSize>>>(a, b, c, id);
        }
        __global__ void gpuMatMul(mat4* a, mat4* b, mat4* c)
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
            int id = threadOffset + threadId;
            c[id] = a[id] * b[id];
        }
    }
}