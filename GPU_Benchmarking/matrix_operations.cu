#include "cuda_operations.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 16 // Block size for thread blocks

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMultiplyKernel(int* matA, int* matB, int* result, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        int value = 0;
        for (int i = 0; i < M; ++i) {
            value += matA[row * M + i] * matB[i * K + col];
        }
        result[row * K + col] = value;
    }
}

void cudaMatrixMultiplication(const std::vector<std::vector<int>>& matA,
    const std::vector<std::vector<int>>& matB,
    std::vector<std::vector<int>>& result) {
    int N = matA.size();     // Rows in matA
    int M = matA[0].size();  // Columns in matA
    int K = matB[0].size();  // Columns in matB

    // Flatten input matrices
    std::vector<int> flatA(N * M);
    std::vector<int> flatB(M * K);
    std::vector<int> flatResult(N * K, 0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            flatA[i * M + j] = matA[i][j];
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            flatB[i * K + j] = matB[i][j];
        }
    }

    // Device memory allocation
    int* d_matA, * d_matB, * d_result;
    size_t sizeA = N * M * sizeof(int);
    size_t sizeB = M * K * sizeof(int);
    size_t sizeResult = N * K * sizeof(int);

    cudaMalloc(&d_matA, sizeA);
    cudaMalloc(&d_matB, sizeB);
    cudaMalloc(&d_result, sizeResult);

    // Copy data to device
    cudaMemcpy(d_matA, flatA.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, flatB.data(), sizeB, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiplyKernel << <numBlocks, threadsPerBlock >> > (d_matA, d_matB, d_result, N, M, K);

    // Copy result back to host
    cudaMemcpy(flatResult.data(), d_result, sizeResult, cudaMemcpyDeviceToHost);

    // Convert result to 2D matrix form
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            result[i][j] = flatResult[i * K + j];
        }
    }

    // Free device memory
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_result);
}

// CUDA Kernel for Matrix Dot Product(element - wise multiplication)
__global__ void matrixDotProductKernel(int* matA, int* matB, int* result, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        int index = row * M + col;
        result[index] = matA[index] * matB[index];
    }
}

void cudaMatrixDotProduct(const std::vector<std::vector<int>>&matA,
    const std::vector<std::vector<int>>&matB,
    std::vector<std::vector<int>>& result) {
    int N = matA.size();     // Rows in matA
    int M = matA[0].size();  // Columns in matA

    // Flatten input matrices
    std::vector<int> flatA(N * M);
    std::vector<int> flatB(N * M);
    std::vector<int> flatResult(N * M, 0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            flatA[i * M + j] = matA[i][j];
            flatB[i * M + j] = matB[i][j];
        }
    }

    // Device memory allocation
    int* d_matA, * d_matB, * d_result;
    size_t size = N * M * sizeof(int);

    cudaMalloc(&d_matA, size);
    cudaMalloc(&d_matB, size);
    cudaMalloc(&d_result, size);

    // Copy data to device
    cudaMemcpy(d_matA, flatA.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, flatB.data(), size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixDotProductKernel << <numBlocks, threadsPerBlock >> > (d_matA, d_matB, d_result, N, M);

    // Copy result back to host
    cudaMemcpy(flatResult.data(), d_result, size, cudaMemcpyDeviceToHost);

    // Convert result to 2D matrix form
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            result[i][j] = flatResult[i * M + j];
        }
    }

    // Free device memory
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_result);
}