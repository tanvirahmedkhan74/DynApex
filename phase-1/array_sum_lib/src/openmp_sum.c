#include <omp.h>
#include "array_sum.h"

void sumArrayWithOpenMP(float* A, float* B, float* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}
