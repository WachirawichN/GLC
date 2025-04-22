# Force TestCPU to be treated as a CUDA target
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(TestOpenGL
    main.cpp
    glad.c
)

target_include_directories(TestOpenGL
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/include
)

# Link test executable with the CUDA library and the OpenGL Libraries
# May needed Windows SDK to access opengl32.lib, or just swap to opengl32.dll which is already inside System32
target_link_libraries(TestOpenGL
    PRIVATE
    CUDA-GL
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/lib/glfw3.lib
    opengl32.lib
)