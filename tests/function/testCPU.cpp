#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>

#include <GLC/matrix.cuh>
#include <GLC/vector.cuh>
#include <GLC/utility.cuh>

struct vertex
{
    GLC::vec2 position;
    GLC::vec3 color;
};

int main()
{
    std::cout << sizeof(GLC::mat4) << " " << sizeof(float) * 16 << std::endl;

    return 0;
}