#include "../../include/vector.cuh"

__host__ __device__ indexProxy::indexProxy(float& input)
    : ref(input)
{
}

__host__ __device__ indexProxy& indexProxy::operator=(float value)
{
    ref = value;
    return *this;
}
__host__ __device__ indexProxy& indexProxy::operator+=(float value)
{
    ref += value;
    return *this;
}
__host__ __device__ indexProxy& indexProxy::operator-=(float value)
{
    ref -= value;
    return *this;
}
__host__ __device__ indexProxy& indexProxy::operator*=(float value)
{
    ref *= value;
    return *this;
}
__host__ __device__ indexProxy& indexProxy::operator/=(float value)
{
    ref /= value;
    return *this;
}

__host__ __device__ indexProxy::operator float() const
{
    return ref;
}