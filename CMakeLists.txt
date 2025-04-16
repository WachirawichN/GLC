cmake_minimum_required(VERSION 4.0.0)
project(CUDA-GL VERSION 0.1.0 LANGUAGES CXX CUDA)

# Set standard and CUDA flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include path
include_directories(${CMAKE_SOURCE_DIR}/CUDA-GL/include)

# CUDA-GL library (compiled from .cu files)
file(GLOB_RECURSE LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/CUDA-GL/src/vector/*.cu
    ${CMAKE_SOURCE_DIR}/CUDA-GL/src/matrix/*.cu
    ${CMAKE_SOURCE_DIR}/CUDA-GL/src/utility/*.cu
    ${CMAKE_SOURCE_DIR}/CUDA-GL/src/experiment/*.cu
)

add_library(CUDA-GL STATIC ${LIB_SOURCES})
target_include_directories(CUDA-GL PUBLIC ${CMAKE_SOURCE_DIR}/CUDA-GL/include)

# Test executable using a single .cpp file
add_executable(TestCPU ${CMAKE_SOURCE_DIR}/tests/testCPU.cpp)
add_executable(TestGPU ${CMAKE_SOURCE_DIR}/tests/testGPU.cu)

# Force CUDA linker for TestCPU (even though it's a .cpp file)
set_target_properties(TestCPU PROPERTIES LINKER_LANGUAGE CUDA)

# Link test executable with the CUDA library
target_link_libraries(TestCPU PRIVATE CUDA-GL)
target_link_libraries(TestGPU PRIVATE CUDA-GL)