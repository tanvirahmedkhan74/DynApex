#ifndef UTILS_H
#define UTILS_H

#include <vector>

std::vector<int> generateRandomVector(int size);
std::vector<std::vector<int>> generateRandomMatrix(int size);

int normalSum(const std::vector<int>& arr);
int avxSum(const std::vector<int>& arr);
int openmpSum(const std::vector<int>& arr);

#endif // UTILS_H
