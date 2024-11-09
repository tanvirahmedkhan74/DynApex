#include "cuda_operations.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// CUDA Kernel to initialize a 3D matrix using 1D indexing
__global__ void matrix3DInitializationKernel(int* matrix, int dim1, int dim2, int dim3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int totalSize = dim1 * dim2 * dim3;
    if (idx < totalSize) {
        int x = idx / (dim2 * dim3);
        int y = (idx / dim3) % dim2;
        int z = idx % dim3;
        matrix[idx] = x + y + z; // Example initialization formula (can be customized)
    }
}

void cuda3DMatrixInitialization(int dim1, int dim2, int dim3) {
    // Total number of elements in the 3D matrix
    int totalSize = dim1 * dim2 * dim3;
    std::vector<int> hostMatrix(totalSize, 0);

    // Allocate device memory
    int* d_matrix;
    cudaMalloc(&d_matrix, totalSize * sizeof(int));

    // Define total number of threads and blocks
    int threadsPerBlock = 256;
    int numBlocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel with 1D configuration
    matrix3DInitializationKernel << <numBlocks, threadsPerBlock >> > (d_matrix, dim1, dim2, dim3);

    // Copy result back to host
    cudaMemcpy(hostMatrix.data(), d_matrix, totalSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
}