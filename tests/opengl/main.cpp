#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "graphic/shader.h"
#include "graphic/VAO.h"
#include "graphic/VBO.h"
#include "graphic/EBO.h"

#include <iostream>

unsigned int wWidth = 640;
unsigned int wHeight = 480;

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
	{
		-0.5f, -0.5f * float(sqrt(3)) / 3, 0.0f, // Lower left corner
		0.5f, -0.5f * float(sqrt(3)) / 3, 0.0f, // Lower right corner
		0.0f, 0.5f * float(sqrt(3)) * 2 / 3, 0.0f, // Upper corner
		-0.5f / 2, 0.5f * float(sqrt(3)) / 6, 0.0f, // Inner left
		0.5f / 2, 0.5f * float(sqrt(3)) / 6, 0.0f, // Inner right
		0.0f, -0.5f * float(sqrt(3)) / 3, 0.0f // Inner down
	};
    GLuint indices[] =
    {
        0, 3, 5,
        3, 2, 4,
        5, 4, 1
    };

    shader shaderProgram("shader/default.frag", "shader/default.vert");

    VAO VAO1;
    VAO1.bind();

    VBO VBO1(vertices, sizeof(vertices));
    EBO EBO1(indices, sizeof(indices));

    VAO1.linkVBO(VBO1, 0);
    VAO1.unbind();
    VBO1.unbind();
    EBO1.unbind();

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        shaderProgram.active();
        VAO1.bind();
        glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_INT, 0);

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