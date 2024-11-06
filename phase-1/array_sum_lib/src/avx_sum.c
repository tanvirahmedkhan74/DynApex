#include <immintrin.h>
#include "array_sum.h"

void sumArrayWithAVX(float* A, float* B, float* C, int N) {
    int i = 0;
    for (i = 0; i < N - 7; i += 8) {
        __m256 a = _mm256_loadu_ps(&A[i]);   // Load 8 floats from A
        __m256 b = _mm256_loadu_ps(&B[i]);   // Load 8 floats from B
        __m256 result = _mm256_add_ps(a, b); // Perform 8 simultaneous additions
        _mm256_storeu_ps(&C[i], result);     // Store result into C
    }

    // Handle any remaining elements (if N is not divisible by 8)
    for (; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}
