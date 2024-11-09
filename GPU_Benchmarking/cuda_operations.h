#ifndef CUDA_OPERATIONS_H
#define CUDA_OPERATIONS_H
#include<vector>

// Matrix Operations
void cudaMatrixMultiplication(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB, std::vector<std::vector<int>>& result);
void cudaMatrixDotProduct(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB, std::vector<std::vector<int>>& result);
// Nested Loops
void cuda3DMatrixInitialization(int dim1, int dim2, int dim3);

// Search Operations
int cudaLinearSearch(const std::vector<int>& arr, int target);

// Vector Operations
void cudaVectorScalarMultiplication(std::vector<int>& vec, int scalar);
void cudaMatrixVectorMultiplication(const std::vector<std::vector<int>>& mat, const std::vector<int>& vec, std::vector<int>& result);

#endif