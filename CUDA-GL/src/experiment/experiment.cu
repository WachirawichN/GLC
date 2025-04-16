#include "../../include/experiment.cuh"

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
        __global__ void partialGPUMatMul()
        {
            // Utilize dynamic parallelism
            // Workflow:
            // 1. Each thread (Output thread) call a number of threads correlate to number of values in the output matrix (those thread are called "Control thread").
            // 2. Each Control thread then perform a dot product operation to both of the vectors needed to perform the output.
        }
        __global__ void GPUMatMul()
        {

        }
    }
}