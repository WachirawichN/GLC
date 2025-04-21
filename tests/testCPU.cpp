#include <iostream>

#include <cuda_runtime.h>

#include "../CUDA-GL/include/matrix.cuh"
#include "../CUDA-GL/include/utility.cuh"
#include "../CUDA-GL/include/vector.cuh"

int main()
{
    CUDA_GL::mat4 translationMat = CUDA_GL::translate(CUDA_GL::vec3(1.0f, 2.0f, 3.0f));
    std::cout << translationMat << std::endl;
    
    CUDA_GL::mat4 scalingMat = CUDA_GL::scale(CUDA_GL::vec3(11.0f, 2.0f, 9.0f));
    std::cout << scalingMat << std::endl;

    CUDA_GL::mat4 rotationMat = CUDA_GL::rotate(45, CUDA_GL::vec3(1.0f, 2.0f, 0.0f));
    std::cout << rotationMat << std::endl;

    std::cout << CUDA_GL::pow(CUDA_GL::vec3(2.0f, 3.0f, 4.0f), 2.0f) << std::endl;

    return 0;
}