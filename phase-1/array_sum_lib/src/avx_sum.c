#include <immintrin.h>
#include "array_sum.h"

#ifdef __AVX2__  // Only compile this if AVX2 is available
void sumArrayWithAVX(float* A, float* B, float* C, int N) {
    int i = 0;
    for (i = 0; i < N - 7; i += 8) {
        __m256 a = _mm256_loadu_ps(&A[i]);
        __m256 b = _mm256_loadu_ps(&B[i]);
        __m256 result = _mm256_add_ps(a, b);
        _mm256_storeu_ps(&C[i], result);
    }

    // Handle remaining elements if N is not divisible by 8
    for (; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}
#else
void sumArrayWithAVX(float* A, float* B, float* C, int N) {
    // Fallback to standard summing if AVX is not available
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}
#endif



