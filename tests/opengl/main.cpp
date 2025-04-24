#include <cuda_runtime.h>
#include <iostream>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "graphic/shader.h"
#include "graphic/VAO.h"
#include "graphic/VBO.h"
#include "graphic/EBO.h"

#include "matrix.cuh"
#include "vector.cuh"
#include "utility.cuh"


unsigned int wWidth = 800;
unsigned int wHeight = 800;

void resizeCallback(GLFWwindow* window, int width, int height)
{
    wWidth = width;
    wHeight = height;
    glViewport(0, 0, wWidth, wHeight);
}
void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main()
{
    if (!glfwInit())
    {
        glfwTerminate();
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(wWidth, wHeight, "CUDA-GL Test", NULL, NULL);
    if (!window)
    {
        std::cout << "Fail to create window." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, resizeCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }    

    GLfloat vertices[] =
	{ //    COORDINATES       /     COLORS           //
        -0.5f, 0.0f,  0.5f,     0.69f, 0.25f, 0.40f, // Bottom front left
        -0.5f, 0.0f, -0.5f,     0.94f, 0.65f, 0.34f, // Bottom back left
         0.5f, 0.0f, -0.5f,     0.37f, 0.30f, 0.26f, // Bottom back right
         0.5f, 0.0f,  0.5f,     0.94f, 0.65f, 0.34f, // Bottom front tight
         0.0f, 0.8f,  0.0f,     0.43f, 0.27f, 0.49f, // Top
	};
    GLuint indices[] =
    {
        0, 1, 2,
        0, 2, 3,
        0, 1, 4,
        1, 2, 4,
        2, 3, 4,
        3, 0, 4
    };

    shader shaderProgram("shader/default.vert", "shader/default.frag");

    VAO VAO1;
    VAO1.bind();

    VBO VBO1(vertices, sizeof(vertices));
    EBO EBO1(indices, sizeof(indices));

    VAO1.linkAttrib(VBO1, 0, 3, GL_FLOAT, 6 * sizeof(float), (void*)0);
    VAO1.linkAttrib(VBO1, 1, 3, GL_FLOAT, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    VAO1.unbind();
    VBO1.unbind();
    EBO1.unbind();

    float rotation = 0.0f;
	double previousTime = glfwGetTime();

    const float radius = 2.0f;
    CUDA_GL::vec3 camPos(std::sinf(previousTime) * radius, 0.0f, std::cosf(previousTime) * radius);
    CUDA_GL::vec3 objPos(0.0f, 0.0f, 0.0f);

    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT);
        shaderProgram.active();

		double currentTime = glfwGetTime();
		if (currentTime - previousTime >= 1 / 10)
		{
			rotation += 0.01f;
            camPos = CUDA_GL::vec3(std::sinf(currentTime) * radius, 0.0f, std::cosf(currentTime) * radius);
			previousTime = currentTime;
		}

        CUDA_GL::mat4 model = CUDA_GL::scale(std::sinf(rotation));
        model *= CUDA_GL::rotate(rotation, CUDA_GL::vec3(1.0f, 0.0f, 0.0f));
        CUDA_GL::mat4 view = CUDA_GL::lookAt(camPos, objPos, CUDA_GL::vec3(0.0f, 1.0f, 0.0f));
        CUDA_GL::mat4 projection = CUDA_GL::perspective(90.0f, (float)(wWidth / wHeight), 0.1f, 100.0f);

        int modelLoc = glGetUniformLocation(shaderProgram.ID, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, CUDA_GL::unpack(model));
        int viewLoc = glGetUniformLocation(shaderProgram.ID, "view");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, CUDA_GL::unpack(view));
        int projectionLoc = glGetUniformLocation(shaderProgram.ID, "projection");
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, CUDA_GL::unpack(projection));

        VAO1.bind();
        glDrawElements(GL_TRIANGLES, sizeof(indices)/sizeof(int), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    VAO1.del();
    VBO1.del();
    EBO1.del();
    shaderProgram.del();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}