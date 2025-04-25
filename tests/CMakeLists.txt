cmake_minimum_required(VERSION 4.0.0)
project(GLC VERSION 0.1.0 LANGUAGES C CXX CUDA)

# Enable separable compilation and device linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set standard and CUDA flags
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

# GLC library (compiled from .cu files)
file(GLOB_RECURSE LIB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLC/src/vector/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLC/src/matrix/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLC/src/utility/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLC/src/experiment/*.cu
)

add_library(GLC STATIC ${LIB_SOURCES})
target_include_directories(GLC PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../GLC/include)

add_subdirectory(function)
add_subdirectory(opengl)