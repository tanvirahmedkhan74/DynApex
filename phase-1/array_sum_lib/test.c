#include <stdio.h>
#include <stdlib.h> // for malloc and free
#include "include/array_sum.h" // This is correct as it points to the header file

int main() {
    // Example array sizes
    int n = 5;

    // Allocate memory for arrays
    float A[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float B[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    float *C = (float *)malloc(n * sizeof(float));

    if (C == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Call the library function (example using array_sum)
    array_sum(A, B, C, n, "openmp");

    // Print the result
    printf("Result of array sum:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", C[i]);
    }
    printf("\n");

    // Free allocated memory
    free(C);

    return 0;
}
