#pragma once

#include "glad/glad.h"

class VBO
{
    public:
        GLuint ID;
        // Constructor that generates a Vertex Buffer Object and links it to vertices
        VBO(GLfloat* vertices, GLsizeiptr size);

        void bind();
        void unbind();
        void del();
};