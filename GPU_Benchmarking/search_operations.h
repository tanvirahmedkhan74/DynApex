#pragma once
#ifndef SEARCH_OPERATIONS_H
#define SEARCH_OPERATIONS_H

#include <vector>

// GPU-based CUDA functions
int cudaLinearSearch(const std::vector<int>& arr, int target);

// GPU-based OpenGL functions
int openglLinearSearch(const std::vector<int>& arr, int target);

#endif // SEARCH_OPERATIONS_H