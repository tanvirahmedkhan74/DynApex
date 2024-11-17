#include "matrix_operations.h"
#include <vector>
#include <immintrin.h>
#include <omp.h>
#include <numeric>

std::vector<std::vector<int>> normalMatrixMultiplication(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB) {
    int size = matA.size();
    std::vector<std::vector<int>> result(size, std::vector<int>(size, 0));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                result[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
    return result;
}

std::vector<std::vector<int>> openmpMatrixMultiplication(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB) {
    int size = matA.size();
    std::vector<std::vector<int>> result(size, std::vector<int>(size, 0));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                result[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
    return result;
}

int normalDotProduct(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    int dot_product = 0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dot_product += vec1[i] * vec2[i];
    }
    return dot_product;
}

int avxDotProduct(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    __m256i dot_vec = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= vec1.size(); i += 8) {
        __m256i vals1 = _mm256_loadu_si256((__m256i*) & vec1[i]);
        __m256i vals2 = _mm256_loadu_si256((__m256i*) & vec2[i]);
        __m256i mul_vals = _mm256_mullo_epi32(vals1, vals2);
        dot_vec = _mm256_add_epi32(dot_vec, mul_vals);
    }
    int dot_array[8];
    _mm256_storeu_si256((__m256i*) dot_array, dot_vec);
    int dot_product = std::accumulate(dot_array, dot_array + 8, 0);
    for (; i < vec1.size(); ++i) {
        dot_product += vec1[i] * vec2[i];
    }
    return dot_product;
}

int openmpDotProduct(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    int dot_product = 0;
#pragma omp parallel for reduction(+:dot_product)
    for (int i = 0; i < vec1.size(); ++i) {
        dot_product += vec1[i] * vec2[i];
    }
    return dot_product;
}
