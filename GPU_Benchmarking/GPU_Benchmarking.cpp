#include "cuda_operations.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include "utils.h"

void benchmarkCudaMatrixMultiplication(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB) {
    std::cout << "Benchmark for CUDA Matrix Multiplication\n";

    std::vector<std::vector<int>> result(matA.size(), std::vector<int>(matB[0].size(), 0));

    auto start = std::chrono::high_resolution_clock::now();
    cudaMatrixMultiplication(matA, matB, result);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "CUDA Matrix Multiplication Time:       "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

void benchmarkCudaLinearSearch(const std::vector<int>& vec, int target) {
    std::cout << "Benchmark for CUDA Linear Search\n";

    auto start = std::chrono::high_resolution_clock::now();
    int result = cudaLinearSearch(vec, target);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "CUDA Linear Search Time:               "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

void benchmarkCuda3DMatrixInitialization(int dim1, int dim2, int dim3) {
    std::cout << "Benchmark for CUDA 3D Matrix Initialization\n";

    auto start = std::chrono::high_resolution_clock::now();
    cuda3DMatrixInitialization(dim1, dim2, dim3);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "CUDA 3D Matrix Initialization Time:    "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

void benchmarkCudaVectorScalarMultiplication(std::vector<int>& vec, int scalar) {
    std::cout << "Benchmark for CUDA Vector-Scalar Multiplication\n";

    auto start = std::chrono::high_resolution_clock::now();
    cudaVectorScalarMultiplication(vec, scalar);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "CUDA Vector-Scalar Multiplication Time: "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

void benchmarkCudaMatrixVectorMultiplication(const std::vector<std::vector<int>>& mat, const std::vector<int>& vec) {
    std::cout << "Benchmark for CUDA Matrix-Vector Multiplication\n";

    std::vector<int> result(mat.size(), 0);

    auto start = std::chrono::high_resolution_clock::now();
    cudaMatrixVectorMultiplication(mat, vec, result);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "CUDA Matrix-Vector Multiplication Time: "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(10);

    const int VECTOR_SIZE = 1 << 20;  // Large vector size (e.g., 2^20 elements)
    const int MATRIX_SIZE = 512;      // Large matrix size for matrix operations

    auto vec = generateRandomVector(VECTOR_SIZE);
    auto matA = generateRandomMatrix(MATRIX_SIZE);
    auto matB = generateRandomMatrix(MATRIX_SIZE);
    int target = vec[VECTOR_SIZE / 2]; 
    int scalar = 5;

    std::cout << "Benchmarking CUDA Operations\n\n";
    std::cout << "Vector Size: " << VECTOR_SIZE << "\n";
    std::cout << "Matrix Size: " << MATRIX_SIZE << "\n\n";
    std::cout << "Operation Benchmark Results\n\n";

    benchmarkCudaMatrixMultiplication(matA, matB);
    benchmarkCudaLinearSearch(vec, target);
    benchmarkCuda3DMatrixInitialization(100, 100, 100);
    benchmarkCudaVectorScalarMultiplication(vec, scalar);
    benchmarkCudaMatrixVectorMultiplication(matA, vec);

    return 0;
}

/**
Benchmarking CUDA Operations

Vector Size: 1048576
Matrix Size: 512

Operation Benchmark Results

Benchmark for CUDA Matrix Multiplication
CUDA Matrix Multiplication Time:          0.1193513000 seconds

Benchmark for CUDA Linear Search
CUDA Linear Search Time:                  0.0015102000 seconds

Benchmark for CUDA 3D Matrix Initialization
CUDA 3D Matrix Initialization Time:       0.0166959000 seconds

Benchmark for CUDA Vector-Scalar Multiplication
CUDA Vector-Scalar Multiplication Time:    0.0117496000 seconds

Benchmark for CUDA Matrix-Vector Multiplication
CUDA Matrix-Vector Multiplication Time:    0.0026575000 seconds
**/