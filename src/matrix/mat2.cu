#include <GLC/matrix.cuh>

namespace GLC
{
    __host__ __device__ vec2& mat2::operator[](unsigned int index)
    {
        return *(&x + index);
    }
    __host__ __device__ const vec2& mat2::operator[](unsigned int index) const
    {
        return *(&x + index);
    }

    __host__ __device__ mat2 mat2::operator+(const mat2& matrix) const
    {
        mat2 out;
        for (int i = 0; i < 2; ++i)
        {
            out[i] = (*this)[i] + matrix[i];
        }
        return out;
    }
    __host__ __device__ mat2& mat2::operator+=(const mat2& matrix)
    {
        for (int i = 0; i < 2; ++i)
        {
            (*this)[i] += matrix[i];
        }
        return *this;
    }

    __host__ __device__ mat2 mat2::operator-(const mat2& matrix) const
    {
        mat2 out;
        for (int i = 0; i < 2; ++i)
        {
            out[i] = (*this)[i] - matrix[i];
        }
        return out;
    }
    __host__ __device__ mat2& mat2::operator-=(const mat2& matrix)
    {
        for (int i = 0; i < 2; ++i)
        {
            (*this)[i] -= matrix[i];
        }
        return *this;
    }

    __host__ __device__ mat2 mat2::operator*(float scalar) const
    {
        mat2 out;
        for (int i = 0; i < 2; ++i)
        {
            out[i] = (*this)[i] * scalar;
        }
        return out;
    }
    __host__ __device__ mat2& mat2::operator*=(float scalar)
    {
        for (int i = 0; i < 2; ++i)
        {
            (*this)[i] *= scalar;
        }
        return *this;
    }
    __host__ __device__ mat2 mat2::operator*(const mat2& matrix) const
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
                (*this)[row][column] = dot(transposed[column], matrix[row]);
            }
        }
        return *this;
    }
    __host__ __device__ vec2 mat2::operator*(const vec2& vector) const
    {
        vec2 out;
        mat2 transposed = transpose(*this);
        for (int row = 0; row < 2; row++)
        {
            for (int column = 0; column < 2; column++)
            {
                out[row] += transposed[row][column] * vector[column];
            }
        }
        return out;
    }

    __host__ __device__ mat2 mat2::operator/(float scalar) const
    {
        mat2 out;
        for (int i = 0; i < 2; ++i)
        {
            out[i] = (*this)[i] / scalar;
        }
        return out;
    }
    __host__ __device__ mat2& mat2::operator/=(float scalar)
    {
        for (int i = 0; i < 2; ++i)
        {
            (*this)[i] /= scalar;
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