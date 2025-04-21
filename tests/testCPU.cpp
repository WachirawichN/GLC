#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

#include "../CUDA-GL/include/matrix.cuh"
#include "../CUDA-GL/include/utility.cuh"
#include "../CUDA-GL/include/vector.cuh"

int main()
{
    CUDA_GL::vec2 vector2(2.0f, 10.0f);
    std::cout << CUDA_GL::normalize(vector2) << std::endl;
    
    CUDA_GL::vec3 vector3(1.0f, 2.0f, 3.0f);
    std::cout << CUDA_GL::normalize(vector3) << std::endl;
    
    CUDA_GL::vec4 vector4(8.2f, 9.7f, 34.0f, 90.0f);
    std::cout << CUDA_GL::normalize(vector4) << std::endl;

    return 0;
}