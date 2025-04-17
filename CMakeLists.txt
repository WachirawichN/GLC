cmake_minimum_required(VERSION 4.0.0)
project(CUDA-GL VERSION 0.1.0 LANGUAGES CXX CUDA)

# Enable separable compilation and device linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

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

# Force TestCPU to be treated as a CUDA target
set_source_files_properties(${CMAKE_SOURCE_DIR}/tests/testCPU.cpp PROPERTIES LANGUAGE CUDA)

# Test executable using a single .cpp file
add_executable(TestCPU ${CMAKE_SOURCE_DIR}/tests/testCPU.cpp)
add_executable(TestGPU ${CMAKE_SOURCE_DIR}/tests/testGPU.cu)

# Required for device linking
set_target_properties(TestCPU PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(TestGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link test executable with the CUDA library
target_link_libraries(TestCPU PRIVATE CUDA-GL)
target_link_libraries(TestGPU PRIVATE CUDA-GL)