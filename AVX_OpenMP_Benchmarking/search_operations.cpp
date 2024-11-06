#include "search_operations.h"
#include <vector>
#include <immintrin.h>
#include <omp.h>

int normalLinearSearch(const std::vector<int>& arr, int target) {
    for (size_t i = 0; i < arr.size(); ++i) {
        if (arr[i] == target)
            return i;
    }
    return -1;
}

int avxLinearSearch(const std::vector<int>& arr, int target) {
    __m256i target_vec = _mm256_set1_epi32(target);
    size_t i = 0;
    for (; i + 8 <= arr.size(); i += 8) {
        __m256i vals = _mm256_loadu_si256((__m256i*) & arr[i]);
        __m256i cmp_result = _mm256_cmpeq_epi32(vals, target_vec);
        int mask = _mm256_movemask_epi8(cmp_result);
        if (mask) {
            for (int j = 0; j < 8; ++j) {
                if (arr[i + j] == target)
                    return i + j;
            }
        }
    }
    for (; i < arr.size(); ++i) {
        if (arr[i] == target)
            return i;
    }
    return -1;
}

int openmpLinearSearch(const std::vector<int>& arr, int target) {
    int found_index = -1;
#pragma omp parallel for
    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] == target && found_index == -1) {
#pragma omp critical
            found_index = i;
        }
    }
    return found_index;
}
