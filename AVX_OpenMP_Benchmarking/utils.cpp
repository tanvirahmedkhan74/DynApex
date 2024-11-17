#include "utils.h"
#include <vector>
#include <numeric>
#include <random>
#include <immintrin.h>
#include <omp.h>

std::vector<int> generateRandomVector(int size) {
    std::vector<int> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100000);
    for (auto& val : vec) {
        val = dis(gen);
    }
    return vec;
}

std::vector<std::vector<int>> generateRandomMatrix(int size) {
    std::vector<std::vector<int>> matrix(size, std::vector<int>(size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    for (auto& row : matrix) {
        for (auto& val : row) {
            val = dis(gen);
        }
    }
    return matrix;
}

int normalSum(const std::vector<int>& arr) {
    return std::accumulate(arr.begin(), arr.end(), 0);
}

int avxSum(const std::vector<int>& arr) {
    __m256i sum_vec = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= arr.size(); i += 8) {
        __m256i vals = _mm256_loadu_si256((__m256i*) & arr[i]);
        sum_vec = _mm256_add_epi32(sum_vec, vals);
    }
    int sum_array[8];
    _mm256_storeu_si256((__m256i*) sum_array, sum_vec);
    int sum = std::accumulate(sum_array, sum_array + 8, 0);
    for (; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

int openmpSum(const std::vector<int>& arr) {
    int sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}
