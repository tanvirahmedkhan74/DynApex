#include <iostream>

int main() {
    int arr[100];
#pragma omp parallel for
#pragma omp parallel for
    for (int i = 0; i < 100; i++) {
        arr[i] = i * 2;
    }

    std::cout << "Array processing complete.\n";
    return 0;
}

