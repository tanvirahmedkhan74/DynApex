#ifndef VECTOR_OPERATIONS_H
#define VECTOR_OPERATIONS_H

#include <vector>
// GPU-based CUDA functions
void cudaVectorScalarMultiplication(std::vector<int>& vec, int scalar);
void cudaMatrixVectorMultiplication(const std::vector<std::vector<int>>& mat, const std::vector<int>& vec, std::vector<int>& result);

// GPU-based OpenGL functions
void openglVectorScalarMultiplication(std::vector<int>& vec, int scalar);
void openglMatrixVectorMultiplication(const std::vector<std::vector<int>>& mat, const std::vector<int>& vec, std::vector<int>& result);

#endif // VECTOR_OPERATIONS_H