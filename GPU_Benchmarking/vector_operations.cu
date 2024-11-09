#include "cuda_operations.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// CUDA Kernel for Vector-Scalar Multiplication
__global__ void vectorScalarMultiplicationKernel(int* vec, int scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        vec[idx] *= scalar;
    }
}

void cudaVectorScalarMultiplication(std::vector<int>& vec, int scalar) {
    int size = vec.size();
    int* d_vec;

    // Allocate device memory
    cudaMalloc(&d_vec, size * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_vec, vec.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256; // Adjust based on GPU capabilities
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorScalarMultiplicationKernel << <numBlocks, threadsPerBlock >> > (d_vec, scalar, size);

    // Copy result back to host
    cudaMemcpy(vec.data(), d_vec, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_vec);
}

// CUDA Kernel for Matrix-Vector Multiplication
__global__ void matrixVectorMultiplicationKernel(const int* mat, const int* vec, int* result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        int sum = 0;
        for (int col = 0; col < cols; ++col) {
            sum += mat[row * cols + col] * vec[col];
        }
        result[row] = sum;
    }
}

void cudaMatrixVectorMultiplication(const std::vector<std::vector<int>>& mat,
    const std::vector<int>& vec,
    std::vector<int>& result) {
    int rows = mat.size();
    int cols = mat[0].size();
    result.resize(rows, 0);

    // Flatten input matrix
    std::vector<int> flatMat(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flatMat[i * cols + j] = mat[i][j];
        }
    }

    // Allocate device memory
    int* d_mat, * d_vec, * d_result;
    cudaMalloc(&d_mat, rows * cols * sizeof(int));
    cudaMalloc(&d_vec, cols * sizeof(int));
    cudaMalloc(&d_result, rows * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_mat, flatMat.data(), rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec.data(), cols * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256; // Adjust as needed
    int numBlocks = (rows + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    matrixVectorMultiplicationKernel << <numBlocks, threadsPerBlock >> > (d_mat, d_vec, d_result, rows, cols);

    // Copy result back to host
    cudaMemcpy(result.data(), d_result, rows * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_result);
}