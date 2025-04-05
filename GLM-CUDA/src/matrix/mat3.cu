#include "../../include/matrix.cuh"

#include "../../include/utility.cuh"

namespace GLM_CUDA
{
    __host__ __device__ mat3::mat3()
    {
        value = new vec3[3];
        for (int i = 0; i < 3; i++)
        {
            value[i] = vec3();
        }
    }
    __host__ __device__ mat3::mat3(float v0)
    {
        value = new vec3[3];
        for (int i = 0; i < 3; i++)
        {
            value[i] = vec3(v0);
        }
    }
    __host__ __device__ mat3::mat3(vec3 v0, vec3 v1, vec3 v2)
    {
        value = new vec3[3];
        value[0] = v0;
        value[1] = v1;
        value[2] = v2;
    }
    __host__ __device__ mat3::~mat3()
    {
        delete[] value;
    }

    __host__ __device__ vec3& mat3::operator[](unsigned int index)
    {
        return value[index];
    }
    __host__ __device__ const vec3& mat3::operator[] (unsigned int index) const
    {
        return value[index];
    }
    __host__ __device__ mat3& mat3::operator=(const mat3& matrix)
    {
        if (this != &matrix)
        {
            // Prevent memory leak
            delete[] value;
            value = new vec3[3];

            for (int i = 0; i < 3; i++)
            {
                value[i] = matrix[i];
            }

            // Copy data - Using CUDA's memcpy if possible (More efficient, still contain a bug)
            /*
            #ifdef __CUDA_ARCH__
                cudaMemcpy(
                    value,
                    matrix,
                    3 * sizeof(vec3),
                    cudaMemcpyDeviceToDevice
                );
            #else
                std::copy(matrix[0], matrix[2], value);
            #endif
            */
        }
        return *this;
    }

    __host__ __device__ mat3 mat3::operator+(mat3 matrix)
    {
        mat3 out;
        for (int i = 0; i < 3; ++i)
        {
            out[i] = value[i] + matrix[i];
        }
        return out;
    }
    __host__ __device__ mat3& mat3::operator+=(mat3& matrix)
    {
        for (int i = 0; i < 3; ++i)
        {
            value[i] += matrix[i];
        }
        return *this;
    }

    __host__ __device__ mat3 mat3::operator-(mat3 matrix)
    {
        mat3 out;
        for (int i = 0; i < 3; ++i)
        {
            out[i] = value[i] - matrix[i];
        }
        return out;
    }
    __host__ __device__ mat3& mat3::operator-=(mat3& matrix)
    {
        for (int i = 0; i < 3; ++i)
        {
            value[i] -= matrix[i];
        }
        return *this;
    }

    __host__ __device__ mat3 mat3::operator*(float scalar)
    {
        mat3 out;
        for (int i = 0; i < 3; ++i)
        {
            out[i] = value[i] * scalar;
        }
        return out;
    }
    __host__ __device__ mat3 mat3::operator*(mat3 matrix)
    {
        mat3 out;
        mat3 transposed = transpose(matrix);
        for (int column = 0; column < 3; column++)
        {
            for (int row = 0; row < 3; row++)
            {
                out[column][row] = dot(value[column], transposed[row]);
            }
        }
        return out;
    }
    __host__ __device__ mat3& mat3::operator*=(float scalar)
    {
        for (int i = 0; i < 3; ++i)
        {
            value[i] *= scalar;
        }
        return *this;
    }
    __host__ __device__ mat3& mat3::operator*=(mat3 matrix)
    {
        mat3 transposed = transpose(matrix);
        for (int column = 0; column < 3; column++)
        {
            for (int row = 0; row < 3; row++)
            {
                value[column][row] = dot(value[column], transposed[row]);
            }
        }
        return *this;
    }

    __host__ __device__ mat3 mat3::operator/(float scalar)
    {
        mat3 out;
        for (int i = 0; i < 3; ++i)
        {
            out[i] = value[i] / scalar;
        }
        return out;
    }
    __host__ __device__ mat3& mat3::operator/=(float scalar)
    {
        for (int i = 0; i < 3; ++i)
        {
            value[i] /= scalar;
        }
        return *this;
    }

    __host__ __device__ std::ostream& operator << (std::ostream& os, const mat3& matrix)
    {
        // Expected output
        // ┌ ┌   ┐ ┌   ┐ ┌   ┐ ┐
        // | | a | | d | | g | |
        // | | b | | e | | h | |
        // | | c | | f | | i | |
        // └ └   ┘ └   ┘ └   ┘ ┘

        // Check for maximum length of every number inside matrix
        unsigned int maxLength = 0;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (std::to_string(matrix[i][j]).length() > maxLength)
                {
                    maxLength = std::to_string(matrix[i][j]).length();
                }
            }
        }

        for (int row = 0; row < 5; row++)
        {
            std::string leftBracket = (row == 0) ? "┌" : (row == 5) ? "└" : "|";
            std::string rightBracket = (row == 0) ? "┐" : (row == 5) ? "┘" : "|";
            os << leftBracket << " "; 

            for (int column = 0; column < 3; column++)
            {
                std::string number = (
                    (row == 0 || row == 5) ?
                    std::string(maxLength, ' ') :
                    std::string(maxLength - std::to_string(matrix[column][row]).length(), ' ') + std::to_string(matrix[column][row])
                );
                os << leftBracket << " " << number << " " << rightBracket << " ";
            }

            os << rightBracket;
        }
        return os;
    }
}