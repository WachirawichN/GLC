#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

#include "../CUDA-GL/include/matrix.cuh"
#include "../CUDA-GL/include/utility.cuh"
#include "../CUDA-GL/include/vector.cuh"

int main()
{
    CUDA_GL::mat2 matrix2(1.0f);
    std::cout << matrix2 << std::endl;
    
    CUDA_GL::mat3 matrix3(9.0f);
    std::cout << matrix3 << std::endl;
    
    CUDA_GL::mat4 matrix4(3.0f);
    std::cout << matrix4 << std::endl;

    return 0;
}