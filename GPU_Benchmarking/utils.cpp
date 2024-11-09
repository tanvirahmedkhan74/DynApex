#include "utils.h"
#include <vector>
#include <numeric>
#include <random>
#include <immintrin.h>

std::vector<int> generateRandomVector(int size) {
    std::vector<int> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100000);
    for (auto& val : vec) {
        val = dis(gen);
    }
    return vec;
}

std::vector<std::vector<int>> generateRandomMatrix(int size) {
    std::vector<std::vector<int>> matrix(size, std::vector<int>(size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    for (auto& row : matrix) {
        for (auto& val : row) {
            val = dis(gen);
        }
    }
    return matrix;
}