#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void cudaMatrixMulKernel(const int* matA, const int* matB, int* matC, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += matA[row * N + k] * matB[k * N + col];
        }
        matC[row * N + col] = sum;
    }
}

void cudaMatrixMultiplication(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB, std::vector<std::vector<int>>& result) {
    int N = matA.size();
    int* d_matA;
    int* d_matB;
    int* d_matC;

    size_t bytes = N * N * sizeof(int);
    cudaMalloc(&d_matA, bytes);
    cudaMalloc(&d_matB, bytes);
    cudaMalloc(&d_matC, bytes);

    std::vector<int> flatA(N * N);
    std::vector<int> flatB(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            flatA[i * N + j] = matA[i][j];
            flatB[i * N + j] = matB[i][j];
        }
    }

    cudaMemcpy(d_matA, flatA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, flatB.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    cudaMatrixMulKernel << <blocksPerGrid, threadsPerBlock >> > (d_matA, d_matB, d_matC, N);

    std::vector<int> flatC(N * N);
    cudaMemcpy(flatC.data(), d_matC, bytes, cudaMemcpyDeviceToHost);

    result.resize(N, std::vector<int>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i][j] = flatC[i * N + j];
        }
    }

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
}