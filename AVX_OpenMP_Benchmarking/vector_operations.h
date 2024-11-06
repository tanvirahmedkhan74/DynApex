#ifndef VECTOR_OPERATIONS_H
#define VECTOR_OPERATIONS_H

#include <vector>

void normalVectorScalarMultiplication(std::vector<int>& vec, int scalar);
void avxVectorScalarMultiplication(std::vector<int>& vec, int scalar);
void openmpVectorScalarMultiplication(std::vector<int>& vec, int scalar);

std::vector<int> normalMatrixVectorMultiplication(const std::vector<std::vector<int>>& mat, const std::vector<int>& vec);
std::vector<int> openmpMatrixVectorMultiplication(const std::vector<std::vector<int>>& mat, const std::vector<int>& vec);

std::vector<int> normalPrefixSum(const std::vector<int>& vec);
std::vector<int> openmpPrefixSum(const std::vector<int>& vec);

int normalMax(const std::vector<int>& vec);
int openmpMax(const std::vector<int>& vec);

#endif // VECTOR_OPERATIONS_H
