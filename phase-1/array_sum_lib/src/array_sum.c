#include "array_sum.h"
#include <string.h>
#include <stdio.h>
#include <time.h>

void array_sum(float* A, float* B, float* C, int n, const char* method) {
    clock_t start_time, end_time;
    double time_taken;

    start_time = clock();  // Start the timer

    if (strcmp(method, "avx") == 0) {
        sumArrayWithAVX(A, B, C, n);
    } else if (strcmp(method, "openmp") == 0) {
        sumArrayWithOpenMP(A, B, C, n);
    } else {
        for (int i = 0; i < n; i++) {
            C[i] = A[i] + B[i];
        }
    }

    end_time = clock();  // End the timer
    time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;  // Calculate time taken
    printf("Method: %s, Time taken: %.6f seconds\n", method, time_taken);
}
