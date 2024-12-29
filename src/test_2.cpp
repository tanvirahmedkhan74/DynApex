#include <iostream>
#include <vector>

int main() {
    const int N = 100;

    // Example 1: Loop with no dependencies (parallelizable) - from previous example
    std::vector<int> a(N);
    for (int i = 0; i < N; ++i) {
        a[i] = i * 2;
    }
    std::cout << "Example 1: Loop with no dependency. Final value: " << a[N-1] << "\n";


    // Example 2: Loop with circular dependency due to array[i+1] on LHS
    std::vector<int> b(N);   
    b[0] = 0; // Initialize the first element
    for (int i = 0; i < N - 1; ++i) { 
        b[i + 1] = b[i] + 1; // array[i+1] on LHS creating a circular dependency
    }
    std::cout << "Example 2: Loop with circular dependency from array[i+1]. Final value: " << b[N-1] << "\n";
    
    // Example 3: Loop with a dependency (not parallelizable) - from previous example
      std::vector<int> c(N);
    c[0] = 1;
    for (int i = 1; i < N; ++i) {
        c[i] = c[i - 1] + 1;
    }

     std::cout << "Example 3: Loop with dependency. Final value: " << c[N-1] << "\n";
    
    return 0;
}
