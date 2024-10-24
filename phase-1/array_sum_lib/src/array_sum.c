#include "array_sum.h"
#include <string.h>

void array_sum(float* A, float* B, float* C, int n, const char* method) {
    if (strcmp(method, "cuda") == 0) {
        sumArrayOnGPU(A, B, C, n);
    } else if (strcmp(method, "avx") == 0) {
        sumArrayWithAVX(A, B, C, n);
    } else if (strcmp(method, "openmp") == 0) {
        sumArrayWithOpenMP(A, B, C, n);
    } else {
        for (int i = 0; i < n; i++) {
            C[i] = A[i] + B[i];
        }
    }
}
