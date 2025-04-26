#include <GLC/utility.cuh>

namespace GLC
{
    __host__ __device__ float radians(float degree)
    {
        #ifdef __CUDA_ARCH__
            return degree * CUDART_PI_F / 180.0f;
        #else
            return degree * std::numbers::pi / 180.0f;
        #endif
    }
    __device__ int threadID()
    {
        int blockId = 
        blockIdx.x +
        blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;
        int threadOffset = blockId * blockDim.x * blockDim.y * blockDim.z;
        int threadId = 
            threadIdx.x + 
            threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
        return threadOffset + threadId;
    }
}