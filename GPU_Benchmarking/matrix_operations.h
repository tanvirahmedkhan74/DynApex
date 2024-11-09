#pragma once
#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <vector>

// GPU-based CUDA functions
void cudaMatrixMultiplication(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB, std::vector<std::vector<int>>& result);

// GPU-based OpenGL functions
void openglMatrixMultiplication(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB, std::vector<std::vector<int>>& result);

#endif // MATRIX_OPERATIONS_H
