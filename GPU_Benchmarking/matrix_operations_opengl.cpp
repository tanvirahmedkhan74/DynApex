#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <iostream>

// Shader code for matrix multiplication
const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
uniform sampler2D matA;
uniform sampler2D matB;
uniform int N;

out vec4 FragColor;

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);
    float result = 0.0;

    for (int k = 0; k < N; ++k) {
        float a = texelFetch(matA, ivec2(k, y), 0).r;
        float b = texelFetch(matB, ivec2(x, k), 0).r;
        result += a * b;
    }

    FragColor = vec4(result, 0.0, 0.0, 1.0);
}
)";

void openglMatrixMultiplication(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB, std::vector<std::vector<int>>& result) {
    int N = matA.size();
    result.resize(N, std::vector<int>(N, 0));

    // Initialize OpenGL context (with GLFW and GLEW)
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    GLFWwindow* window = glfwCreateWindow(1, 1, "Matrix Multiplication", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return;
    }

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Bind matrices to OpenGL textures
    GLuint texA, texB;
    glGenTextures(1, &texA);
    glBindTexture(GL_TEXTURE_2D, texA);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, N, N, 0, GL_RED, GL_FLOAT, &matA[0][0]);

    glGenTextures(1, &texB);
    glBindTexture(GL_TEXTURE_2D, texB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, N, N, 0, GL_RED, GL_FLOAT, &matB[0][0]);

    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "matA"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "matB"), 1);
    glUniform1i(glGetUniformLocation(shaderProgram, "N"), N);

    // Render and retrieve data from framebuffer
    glDrawArrays(GL_TRIANGLES, 0, 6);

    std::vector<float> flatResult(N * N);
    glReadPixels(0, 0, N, N, GL_RED, GL_FLOAT, flatResult.data());

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i][j] = static_cast<int>(flatResult[i * N + j]);
        }
    }

    glDeleteTextures(1, &texA);
    glDeleteTextures(1, &texB);
    glDeleteProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
}
