cmake_minimum_required(VERSION 4.0.0)
project(GLM-CUD VERSION 0.1.0 LANGUAGES CXX CUDA)

# Set standard and CUDA flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include path
include_directories(${CMAKE_SOURCE_DIR}/GLM-CUDA/include)

# GLM-CUDA library (compiled from .cu files)
file(GLOB_RECURSE LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/GLM-CUDA/src/vector/*.cu
    ${CMAKE_SOURCE_DIR}/GLM-CUDA/src/matrix/*.cu
    ${CMAKE_SOURCE_DIR}/GLM-CUDA/src/utility/*.cu
)

add_library(GLM-CUDA STATIC ${LIB_SOURCES})
target_include_directories(GLM-CUDA PUBLIC ${CMAKE_SOURCE_DIR}/GLM-CUDA/include)

# Test executable using a single .cpp file
add_executable(TestCPU ${CMAKE_SOURCE_DIR}/tests/test.cpp)

# Force CUDA linker for TestCPU (even though it's a .cpp file)
set_target_properties(TestCPU PROPERTIES LINKER_LANGUAGE CUDA)

# Link test executable with the CUDA library
target_link_libraries(TestCPU PRIVATE GLM-CUDA)