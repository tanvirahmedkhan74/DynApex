#include "cuda_operations.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// CUDA Kernel for Linear Search
__global__ void linearSearchKernel(const int* arr, int size, int target, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (arr[idx] == target) {
            *result = idx; // Assign index to result when target is found
        }
    }
}

int cudaLinearSearch(const std::vector<int>& arr, int target) {
    int size = arr.size();
    int* d_arr, * d_result;
    int hostResult = -1; // Result initialized to -1 (not found)

    // Allocate device memory
    cudaMalloc(&d_arr, size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_arr, arr.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &hostResult, sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256; // Set based on GPU capabilities
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    linearSearchKernel << <numBlocks, threadsPerBlock >> > (d_arr, size, target, d_result);

    // Copy result back to host
    cudaMemcpy(&hostResult, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_result);

    // Return the index of the target element or -1 if not found
    return hostResult;
}