# Include path
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA-GL/include)

# Force TestCPU to be treated as a CUDA target
set_source_files_properties(testCPU.cpp PROPERTIES LANGUAGE CUDA)

# Test executable using a single .cpp file
add_executable(TestCPU testCPU.cpp)
add_executable(TestGPU testGPU.cu)

# Required for device linking
set_target_properties(TestCPU PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(TestGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link test executable with the CUDA library
target_link_libraries(TestCPU PRIVATE CUDA-GL)
target_link_libraries(TestGPU PRIVATE CUDA-GL)