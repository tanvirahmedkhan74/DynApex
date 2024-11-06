#include "nested_loops.h"
#include <vector>

void normal3DMatrixInitialization(int dim1, int dim2, int dim3) {
    std::vector<std::vector<std::vector<int>>> matrix(dim1, std::vector<std::vector<int>>(dim2, std::vector<int>(dim3, 0)));
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                matrix[i][j][k] = i + j + k;
            }
        }
    }
}

void openmp3DMatrixInitialization(int dim1, int dim2, int dim3) {
    std::vector<std::vector<std::vector<int>>> matrix(dim1, std::vector<std::vector<int>>(dim2, std::vector<int>(dim3, 0)));
#pragma omp parallel for collapse(3)
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                matrix[i][j][k] = i + j + k;
            }
        }
    }
}
