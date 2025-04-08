#include "../../include/matrix.cuh"

#include "../../include/utility.cuh"

namespace GLM_CUDA
{
    __host__ __device__ mat2::mat2()
    {
        value = new vec2[2];
        for (int i = 0; i < 2; i++)
        {
            value[i] = vec2();
        }
    }
    __host__ __device__ mat2::mat2(float v0)
    {
        value = new vec2[2];
        for (int i = 0; i < 2; i++)
        {
            value[i] = vec2(v0);
        }
    }
    __host__ __device__ mat2::mat2(const vec2& v0, const vec2& v1)
    {
        value = new vec2[2] {v0, v1};
    }
    __host__ __device__ mat2::~mat2()
    {
        delete[] value;
    }

    __host__ __device__ vec2& mat2::operator[](unsigned int index)
    {
        return value[index];
    }
    __host__ __device__ const vec2& mat2::operator[](unsigned int index) const
    {
        return value[index];
    }
    __host__ __device__ mat2& mat2::operator=(const mat2& matrix)
    {
        if (this != &matrix)
        {
            // Prevent memory leak
            if (value)
            {
                delete[] value;
                value = new vec2[2];
            }

            for (int i = 0; i < 2; i++)
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

    __host__ __device__ mat2 mat2::operator+(const mat2& matrix)
    {
        mat2 out;
        for (int i = 0; i < 2; ++i)
        {
            out[i] = value[i] + matrix[i];
        }
        return out;
    }
    __host__ __device__ mat2& mat2::operator+=(const mat2& matrix)
    {
        for (int i = 0; i < 2; ++i)
        {
            value[i] += matrix[i];
        }
        return *this;
    }

    __host__ __device__ mat2 mat2::operator-(const mat2& matrix)
    {
        mat2 out;
        for (int i = 0; i < 2; ++i)
        {
            out[i] = value[i] - matrix[i];
        }
        return out;
    }
    __host__ __device__ mat2& mat2::operator-=(const mat2& matrix)
    {
        for (int i = 0; i < 2; ++i)
        {
            value[i] -= matrix[i];
        }
        return *this;
    }

    __host__ __device__ mat2 mat2::operator*(float scalar)
    {
        mat2 out;
        for (int i = 0; i < 2; ++i)
        {
            out[i] = value[i] * scalar;
        }
        return out;
    }
    __host__ __device__ mat2& mat2::operator*=(float scalar)
    {
        for (int i = 0; i < 2; ++i)
        {
            value[i] *= scalar;
        }
        return *this;
    }
    __host__ __device__ mat2 mat2::operator*(const mat2& matrix)
    {
        mat2 out;
        mat2 transposed = transpose(*this);
        for (int column = 0; column < 2; column++)
        {
            for (int row = 0; row < 2; row++)
            {
                out[row][column] = dot(transposed[column], matrix[row]);
            }
        }
        return out;
    }
    __host__ __device__ mat2& mat2::operator*=(const mat2& matrix)
    {
        mat2 transposed = transpose(*this);
        for (int column = 0; column < 2; column++)
        {
            for (int row = 0; row < 2; row++)
            {
                value[row][column] = dot(transposed[column], matrix[row]);
            }
        }
        return *this;
    }
    __host__ __device__ vec2 mat2::operator*(const vec2& vector)
    {
        vec2 out;
        mat2 transposed = transpose(*this);
        for (int row = 0; row < 2; row++)
        {
            for (int column = 0; column < 2; column++)
            {
                out[row] += (row == column) ? transposed[row][column] * vector[row] : transposed[row][column];
            }
        }
        return out;
    }

    __host__ __device__ mat2 mat2::operator/(float scalar)
    {
        mat2 out;
        for (int i = 0; i < 2; ++i)
        {
            out[i] = value[i] / scalar;
        }
        return out;
    }
    __host__ __device__ mat2& mat2::operator/=(float scalar)
    {
        for (int i = 0; i < 2; ++i)
        {
            value[i] /= scalar;
        }
        return *this;
    }

    __host__ std::ostream& operator<<(std::ostream& os, const mat2& matrix)
    {
        // Expected output
        // |             |
        // | | a | | c | |
        // | | b | | d | |
        // |             |

        // Check for maximum length of every number inside matrix
        unsigned int maxLength = 0;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                if (std::to_string(matrix[i][j]).length() > maxLength)
                {
                    maxLength = std::to_string(matrix[i][j]).length();
                }
            }
        }

        for (int row = 0; row < 4; row++)
        {
            os << "|" << " "; 

            for (int column = 0; column < 2; column++)
            {
                std::string bracket;
                std::string numStr;
                if (row == 0 || row == 3)
                {
                    bracket = " ";
                    numStr = std::string(maxLength, ' ');
                }
                else
                {
                    bracket = "|";
                    numStr = std::string(maxLength - std::to_string(matrix[column][row - 1]).length(), ' ') + std::to_string(matrix[column][row - 1]);
                }
                os << bracket << " " << numStr << " " << bracket << " ";
            }

            os << "|" << "\n";
        }
        return os;
    }
}