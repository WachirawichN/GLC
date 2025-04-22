#pragma once

#include "glad/glad.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cerrno>

std::string getFileContents(const char* fileName);

class shader
{
    public:
        GLuint ID;
        shader(const char* vertexFile, const char* fragmentFile);

        void active();
        void del();
};