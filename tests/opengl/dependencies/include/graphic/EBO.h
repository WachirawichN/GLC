#pragma once

#include "glad/glad.h"

class EBO
{
    public:
        GLuint ID;
	    // Constructor that generates a Elements Buffer Object and links it to indices
        EBO(GLuint* indices, GLsizeiptr size);

        void bind();
        void unbind();
        void del();
};