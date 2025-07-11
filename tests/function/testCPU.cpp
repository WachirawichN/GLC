#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>

#include <GLC/GLC.cuh>

int main()
{
    GLC::vec4 testVector(1.0f, 2.0f, 3.0f, 4.0f);
    testVector = GLC::pow(testVector, 2.0f);
    float* unpackedVec = GLC::unpack(testVector);
    for (int i = 0; i < (int)(sizeof(testVector) / sizeof(float)); i++)
    {
        std::cout << unpackedVec[i] << " ";
    }
    std::cout << std::endl;

    GLC::mat4 testMatrix(
        GLC::vec4(0.0f, 0.1f, 0.2f, 0.3f),
        GLC::vec4(0.4f, 0.5f, 0.6f, 0.7f),
        GLC::vec4(0.8f, 0.9f, 1.0f, 1.1f),
        GLC::vec4(1.2f, 1.3f, 1.4f, 1.5f)
    );
    testMatrix = GLC::pow(testMatrix, 2.0f);
    float* unpackedMat = GLC::unpack(testMatrix);
    for (int i = 0; i < (int)(sizeof(testMatrix) / sizeof(float)); i++)
    {
        std::cout << unpackedMat[i] << " ";
    }
    std::cout << std::endl;

    GLC::mat4 tranformMat(1.0f);
    tranformMat = GLC::scale(3.0f, tranformMat);
    std::cout << tranformMat << std::endl;
    tranformMat = GLC::translate(GLC::vec3(1.0f, 2.0f, 3.0f), tranformMat);
    std::cout << tranformMat << std::endl;
    tranformMat = GLC::rotate(1.0f, GLC::vec3(1.0f, 2.0f, 1.0f), tranformMat);
    std::cout << tranformMat << std::endl;

    for (int i = 0; i < 10; i++)
    {
        std::cout << GLC::radians(i * 10.0f) << " ";
    }
    std::cout << std::endl;

    std::cout << "Standard layout: " << std::is_standard_layout<GLC::mat2>::value << std::endl;
    std::cout << "Trivially copyable: " << std::is_trivially_copyable<GLC::mat2>::value << std::endl;

    std::cout << "Standard layout: " << std::is_standard_layout<GLC::mat3>::value << std::endl;
    std::cout << "Trivially copyable: " << std::is_trivially_copyable<GLC::mat3>::value << std::endl;

    std::cout << "Standard layout: " << std::is_standard_layout<GLC::mat4>::value << std::endl;
    std::cout << "Trivially copyable: " << std::is_trivially_copyable<GLC::mat4>::value << std::endl;
    return 0;
}