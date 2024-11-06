#include "matrix_operations.h"
#include "search_operations.h"
#include "nested_loops.h"
#include "vector_operations.h"
#include "utils.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

void benchmarkLinearSearch(const std::vector<int>& vec, int target) {
    std::cout << "Benchmark for Linear Search\n";

    auto start = std::chrono::high_resolution_clock::now();
    normalLinearSearch(vec, target);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Normal Linear Search Time:           "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    avxLinearSearch(vec, target);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "AVX Linear Search Time:              "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    openmpLinearSearch(vec, target);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "OpenMP Linear Search Time:           "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

void benchmarkSumOfElements(const std::vector<int>& vec) {
    std::cout << "Benchmark for Sum of Elements\n";

    auto start = std::chrono::high_resolution_clock::now();
    normalSum(vec);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Normal Sum Time:                     "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    avxSum(vec);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "AVX Sum Time:                        "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    openmpSum(vec);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "OpenMP Sum Time:                     "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

void benchmarkDotProduct(const std::vector<int>& vec) {
    std::cout << "Benchmark for Dot Product\n";

    auto start = std::chrono::high_resolution_clock::now();
    normalDotProduct(vec, vec);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Normal Dot Product Time:             "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    avxDotProduct(vec, vec);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "AVX Dot Product Time:                "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    openmpDotProduct(vec, vec);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "OpenMP Dot Product Time:             "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

void benchmarkMatrixMultiplication(const std::vector<std::vector<int>>& matA, const std::vector<std::vector<int>>& matB) {
    std::cout << "Benchmark for Matrix Multiplication\n";

    auto start = std::chrono::high_resolution_clock::now();
    normalMatrixMultiplication(matA, matB);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Normal Matrix Multiplication Time:   "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    openmpMatrixMultiplication(matA, matB);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "OpenMP Matrix Multiplication Time:   "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

void benchmarkNestedLoops() {
    std::cout << "Benchmark for Nested Loops (Triple Loop Initialization)\n";

    const int DIM1 = 100, DIM2 = 100, DIM3 = 100;

    auto start = std::chrono::high_resolution_clock::now();
    normal3DMatrixInitialization(DIM1, DIM2, DIM3);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Normal 3D Matrix Initialization Time: "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    openmp3DMatrixInitialization(DIM1, DIM2, DIM3);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "OpenMP 3D Matrix Initialization Time: "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

void benchmarkVectorScalarMultiplication(std::vector<int>& vec, int scalar) {
    std::cout << "Benchmark for Vector-Scalar Multiplication\n";

    auto start = std::chrono::high_resolution_clock::now();
    normalVectorScalarMultiplication(vec, scalar);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Normal Vector-Scalar Multiplication Time: "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    avxVectorScalarMultiplication(vec, scalar);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "AVX Vector-Scalar Multiplication Time:    "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    openmpVectorScalarMultiplication(vec, scalar);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "OpenMP Vector-Scalar Multiplication Time: "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

void benchmarkPrefixSum(const std::vector<int>& vec) {
    std::cout << "Benchmark for Prefix Sum\n";

    auto start = std::chrono::high_resolution_clock::now();
    normalPrefixSum(vec);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Normal Prefix Sum Time:                 "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    openmpPrefixSum(vec);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "OpenMP Prefix Sum Time:                 "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

void benchmarkMaxReduction(const std::vector<int>& vec) {
    std::cout << "Benchmark for Max Reduction\n";

    auto start = std::chrono::high_resolution_clock::now();
    normalMax(vec);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Normal Max Reduction Time:              "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    openmpMax(vec);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "OpenMP Max Reduction Time:              "
        << std::setw(15) << std::chrono::duration<double>(end - start).count() << " seconds\n\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(10);

    const int VECTOR_SIZE = 1 << 20;  // Large vector size (e.g., 2^20 elements)
    const int MATRIX_SIZE = 512;      // Large matrix size for matrix operations

    auto vec = generateRandomVector(VECTOR_SIZE);
    auto matA = generateRandomMatrix(MATRIX_SIZE);
    auto matB = generateRandomMatrix(MATRIX_SIZE);
    int target = vec[VECTOR_SIZE / 2];  // Choose an element in the middle as the target
    int scalar = 5;

    std::cout << "Benchmarking Different Operations\n\n";
    std::cout << "Vector Size: " << VECTOR_SIZE << "\n";
    std::cout << "Matrix Size: " << MATRIX_SIZE << "\n\n";
    std::cout << "Operation Benchmark Results\n\n";

    benchmarkLinearSearch(vec, target);
    benchmarkSumOfElements(vec);
    benchmarkDotProduct(vec);
    benchmarkMatrixMultiplication(matA, matB);
    benchmarkNestedLoops();
    benchmarkVectorScalarMultiplication(vec, scalar);
    benchmarkPrefixSum(vec);
    benchmarkMaxReduction(vec);

    return 0;
}

/*
Benchmarking Different Operations

Vector Size: 1048576
Matrix Size: 512

Operation Benchmark Results

Benchmark for Linear Search
Normal Linear Search Time:              0.0020025000 seconds
AVX Linear Search Time:                 0.0003413000 seconds
OpenMP Linear Search Time:              0.0016182000 seconds

Benchmark for Sum of Elements
Normal Sum Time:                        0.0054466000 seconds
AVX Sum Time:                           0.0019338000 seconds
OpenMP Sum Time:                        0.0008676000 seconds

Benchmark for Dot Product
Normal Dot Product Time:                0.0169785000 seconds
AVX Dot Product Time:                   0.0037373000 seconds
OpenMP Dot Product Time:                0.0015183000 seconds

Benchmark for Matrix Multiplication
Normal Matrix Multiplication Time:      3.6579025000 seconds
OpenMP Matrix Multiplication Time:      0.6347074000 seconds

Benchmark for Nested Loops (Triple Loop Initialization)
Normal 3D Matrix Initialization Time:    0.0353353000 seconds
OpenMP 3D Matrix Initialization Time:    0.0198506000 seconds

Benchmark for Vector-Scalar Multiplication
Normal Vector-Scalar Multiplication Time:    0.0025681000 seconds
AVX Vector-Scalar Multiplication Time:       0.0030861000 seconds
OpenMP Vector-Scalar Multiplication Time:    0.0017295000 seconds

Benchmark for Prefix Sum
Normal Prefix Sum Time:                    0.0085202000 seconds
OpenMP Prefix Sum Time:                    0.0066057000 seconds

Benchmark for Max Reduction
Normal Max Reduction Time:                 0.0071747000 seconds
*/