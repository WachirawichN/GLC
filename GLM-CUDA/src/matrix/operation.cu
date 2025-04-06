#include "../../include/matrix.cuh"

namespace GLM_CUDA
{
    /*
    __host__ __device__ mat2 transpose(mat2 matrix)
    {
        mat2 out;
        for (int row = 0; row < 2; row++)
        {
            for (int column = 0; column < 2; column++)
            {
                out[row][column] = (float)matrix[column][row];
            }
        }
        return out;
    }
    */
    __host__ __device__ mat3 transpose(const mat3& matrix)
    {
        mat3 out;
        for (int row = 0; row < 3; row++)
        {
            for (int column = 0; column < 3; column++)
            {
                out[row][column] = (float)matrix[column][row];
            }
        }
        return out;
    }
    /*
    __host__ __device__ mat4 transpose(mat4 matrix)
    {
        mat4 out;
        for (int row = 0; row < 4; row++)
        {
            for (int column = 0; column < 4; column++)
            {
                out[row][column] = (float)matrix[column][row];
            }
        }
        return out;
    }
    */
}