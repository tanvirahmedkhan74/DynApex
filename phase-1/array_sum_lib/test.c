#include <stdio.h>
#include <stdlib.h>
#include "array_sum.h"

#define N 100000000  // Size of the arrays

int main() {
    // Allocate arrays
    float *A = (float*)malloc(N * sizeof(float));
    float *B = (float*)malloc(N * sizeof(float));
    float *C = (float*)malloc(N * sizeof(float));

    // Initialize arrays with some values
    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    // Run and benchmark each method
    array_sum(A, B, C, N, "default");
    array_sum(A, B, C, N, "avx");
    array_sum(A, B, C, N, "openmp");

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}
