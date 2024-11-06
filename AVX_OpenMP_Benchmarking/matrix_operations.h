#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <vector>

std::vector<std::vector<int>> normalMatrixMultiplication(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB);
std::vector<std::vector<int>> openmpMatrixMultiplication(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB);

int normalDotProduct(const std::vector<int>& vec1, const std::vector<int>& vec2);
int avxDotProduct(const std::vector<int>& vec1, const std::vector<int>& vec2);
int openmpDotProduct(const std::vector<int>& vec1, const std::vector<int>& vec2);

#endif // MATRIX_OPERATIONS_H
