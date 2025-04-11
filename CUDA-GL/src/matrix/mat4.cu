#include "../../include/matrix.cuh"

#include "../../include/utility.cuh"

namespace CUDA_GL
{
    __host__ __device__ mat4::mat4()
    {
        value = new vec4[4];
        for (int i = 0; i < 4; i++)
        {
            value[i] = vec4();
        }
    }
    __host__ __device__ mat4::mat4(float v0)
    {
        value = new vec4[4];
        for (int i = 0; i < 4; i++)
        {
            value[i] = vec4(v0);
        }
    }
    __host__ __device__ mat4::mat4(const vec4& v0, const vec4& v1, const vec4& v4, const vec4& v3)
    {
        value = new vec4[4] {v0, v1, v4, v3};
    }
    __host__ __device__ mat4::~mat4()
    {
        delete[] value;
    }

    __host__ __device__ vec4& mat4::operator[](unsigned int index)
    {
        return value[index];
    }
    __host__ __device__ const vec4& mat4::operator[](unsigned int index) const
    {
        return value[index];
    }
    __host__ __device__ mat4& mat4::operator=(const mat4& matrix)
    {
        if (this != &matrix)
        {
            // Prevent memory leak
            if (value)
            {
                delete[] value;
                value = new vec4[4];
            }

            for (int i = 0; i < 4; i++)
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
                std::copy(matrix[0], matrix[4], value);
            #endif
            */
        }
        return *this;
    }

    __host__ __device__ mat4 mat4::operator+(const mat4& matrix)
    {
        mat4 out;
        for (int i = 0; i < 4; ++i)
        {
            out[i] = value[i] + matrix[i];
        }
        return out;
    }
    __host__ __device__ mat4& mat4::operator+=(const mat4& matrix)
    {
        for (int i = 0; i < 4; ++i)
        {
            value[i] += matrix[i];
        }
        return *this;
    }

    __host__ __device__ mat4 mat4::operator-(const mat4& matrix)
    {
        mat4 out;
        for (int i = 0; i < 4; ++i)
        {
            out[i] = value[i] - matrix[i];
        }
        return out;
    }
    __host__ __device__ mat4& mat4::operator-=(const mat4& matrix)
    {
        for (int i = 0; i < 4; ++i)
        {
            value[i] -= matrix[i];
        }
        return *this;
    }

    __host__ __device__ mat4 mat4::operator*(float scalar)
    {
        mat4 out;
        for (int i = 0; i < 4; ++i)
        {
            out[i] = value[i] * scalar;
        }
        return out;
    }
    __host__ __device__ mat4& mat4::operator*=(float scalar)
    {
        for (int i = 0; i < 4; ++i)
        {
            value[i] *= scalar;
        }
        return *this;
    }
    __host__ __device__ mat4 mat4::operator*(const mat4& matrix)
    {
        mat4 out;
        mat4 transposed = transpose(*this);
        for (int column = 0; column < 4; column++)
        {
            for (int row = 0; row < 4; row++)
            {
                out[row][column] = dot(transposed[column], matrix[row]);
            }
        }
        return out;
    }
    __host__ __device__ mat4& mat4::operator*=(const mat4& matrix)
    {
        mat4 transposed = transpose(*this);
        for (int column = 0; column < 4; column++)
        {
            for (int row = 0; row < 4; row++)
            {
                value[row][column] = dot(transposed[column], matrix[row]);
            }
        }
        return *this;
    }
    __host__ __device__ vec4 mat4::operator*(const vec4& vector)
    {
        vec4 out;
        mat4 transposed = transpose(*this);
        for (int row = 0; row < 4; row++)
        {
            for (int column = 0; column < 4; column++)
            {
                out[row] += transposed[row][column] * vector[column];
            }
        }
        return out;
    }

    __host__ __device__ mat4 mat4::operator/(float scalar)
    {
        mat4 out;
        for (int i = 0; i < 4; ++i)
        {
            out[i] = value[i] / scalar;
        }
        return out;
    }
    __host__ __device__ mat4& mat4::operator/=(float scalar)
    {
        for (int i = 0; i < 4; ++i)
        {
            value[i] /= scalar;
        }
        return *this;
    }

    __host__ std::ostream& operator<<(std::ostream& os, const mat4& matrix)
    {
        // Expected output
        // |                         |
        // | | a | | e | | i | | m | |
        // | | b | | f | | j | | n | |
        // | | c | | g | | k | | o | |
        // | | d | | h | | l | | p | |
        // |                         |

        // Check for maximum length of every number inside matrix
        unsigned int maxLength = 0;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                if (std::to_string(matrix[i][j]).length() > maxLength)
                {
                    maxLength = std::to_string(matrix[i][j]).length();
                }
            }
        }

        for (int row = 0; row < 6; row++)
        {
            os << "|" << " "; 

            for (int column = 0; column < 4; column++)
            {
                std::string bracket;
                std::string numStr;
                if (row == 0 || row == 5)
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