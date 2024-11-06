#include "vector_operations.h"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>

// Vector-Scalar Multiplication
void normalVectorScalarMultiplication(std::vector<int>& vec, int scalar) {
    for (auto& val : vec) {
        val *= scalar;
    }
}

void avxVectorScalarMultiplication(std::vector<int>& vec, int scalar) {
    __m256i scalar_vec = _mm256_set1_epi32(scalar);
    size_t i = 0;
    for (; i + 8 <= vec.size(); i += 8) {
        __m256i vals = _mm256_loadu_si256((__m256i*) & vec[i]);
        __m256i result = _mm256_mullo_epi32(vals, scalar_vec);
        _mm256_storeu_si256((__m256i*) & vec[i], result);
    }
    for (; i < vec.size(); ++i) {
        vec[i] *= scalar;
    }
}

void openmpVectorScalarMultiplication(std::vector<int>& vec, int scalar) {
#pragma omp parallel for
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] *= scalar;
    }
}

// Matrix-Vector Multiplication
std::vector<int> normalMatrixVectorMultiplication(const std::vector<std::vector<int>>& mat, const std::vector<int>& vec) {
    std::vector<int> result(mat.size(), 0);
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

std::vector<int> openmpMatrixVectorMultiplication(const std::vector<std::vector<int>>& mat, const std::vector<int>& vec) {
    std::vector<int> result(mat.size(), 0);
#pragma omp parallel for
    for (int i = 0; i < mat.size(); ++i) {
        for (int j = 0; j < vec.size(); ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

// Prefix Sum
std::vector<int> normalPrefixSum(const std::vector<int>& vec) {
    std::vector<int> prefix_sum(vec.size());
    std::partial_sum(vec.begin(), vec.end(), prefix_sum.begin());
    return prefix_sum;
}

std::vector<int> openmpPrefixSum(const std::vector<int>& vec) {
    int n = vec.size();
    std::vector<int> prefix_sum(n, 0);

    if (n == 0) return prefix_sum;  // Return early if the input is empty

    int num_threads;
    std::vector<int> thread_sums;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_elements = n / omp_get_num_threads();
        int start = thread_id * num_elements;
        int end = (thread_id == omp_get_num_threads() - 1) ? n : start + num_elements;

        // Each thread calculates a partial prefix sum
        prefix_sum[start] = vec[start];
        for (int i = start + 1; i < end; ++i) {
            prefix_sum[i] = prefix_sum[i - 1] + vec[i];
        }

        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            thread_sums.resize(num_threads);
        }

        // Store each thread’s total sum
        thread_sums[thread_id] = prefix_sum[end - 1];

        #pragma omp barrier

        // Each thread adds the sum of previous threads to its range
        int offset = 0;
        for (int i = 0; i < thread_id; ++i) {
            offset += thread_sums[i];
        }
        for (int i = start; i < end; ++i) {
            prefix_sum[i] += offset;
        }
    }

    return prefix_sum;
}

// Max Reduction
int normalMax(const std::vector<int>& vec) {
    return *std::max_element(vec.begin(), vec.end());
}



int openmpMax(const std::vector<int>& vec) {
    int max_val = vec[0];
#pragma omp parallel
    {
        int local_max = max_val;
#pragma omp for
        for (int i = 0; i < vec.size(); ++i) {
            local_max = std::max(local_max, vec[i]);
        }
#pragma omp critical
        {
            max_val = std::max(max_val, local_max);
        }
    }
    return max_val;
}

