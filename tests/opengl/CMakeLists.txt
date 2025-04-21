# Force TestCPU to be treated as a CUDA target
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

# Test executable using a single .cpp file
add_executable(TestOpenGL main.cpp)

# Required for device linking
set_target_properties(TestOpenGL PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(TestOpenGL
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/include/GL
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/include/GLFW
)

# Link test executable with the CUDA library and the OpenGL Libraries
# May needed Windows SDK to access opengl32.lib, or just swap to opengl32.dll which is already inside System32
target_link_libraries(TestOpenGL
    PRIVATE
    CUDA-GL
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/lib/glew32s.lib
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/lib/glfw3.lib
    opengl32.lib
)