#include <iostream>
#include <vector>
#include <cmath>

int main() {
    const int N = 1000;
    const int M = 500;
    const int K = 200;

    // Create a 3D array
    std::vector<std::vector<std::vector<double>>> array(N, 
        std::vector<std::vector<double>>(M, std::vector<double>(K, 0)));

    // Complex nested loop without data dependencies
#pragma omp target teams distribute parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < K; k++) {
                array[i][j][k] = sin(i) + cos(j) * tan(k);
            }
        }
    }

    // Output a sample value to confirm calculations
    std::cout << "Computation complete. Sample value: " << array[N/2][M/2][K/2] << std::endl;

    return 0;
}
