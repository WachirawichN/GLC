#include <iostream>
#include <cstdlib>
#include <chrono>

#include "../CUDA-GL/include/matrix.cuh"
#include "../CUDA-GL/include/utility.cuh"
#include "../CUDA-GL/include/vector.cuh"

#include "../CUDA-GL/include/experiment.cuh"

int main()
{
    std::srand(1);
    unsigned int totalMatrix = 1000;

    for (int i = 0; i < totalMatrix; i++)
    {

    }

    CUDA_GL::experimental::allOutGPUMatMul<<<1, 1>>>();

    return 0;
}