#include <iostream>
#include <vector>

int main() {
    const int N = 100;
    const int M = 50;
    const int K = 20;

    // Example 1: Loop with no dependencies (parallelizable)
    std::vector<int> a(N);
    for (int i = 0; i < N; ++i) {
        a[i] = i * 2;
    }
    
    std::cout << "Example 1: Loop with no dependency. Final value: " << a[N-1] << "\n";

    // Example 2: Loop with a dependency (not parallelizable) - simple example
    std::vector<int> b(N);
    b[0] = 1;
    for (int i = 1; i < N; ++i) {
        b[i] = b[i - 1] + 1;
    }

    std::cout << "Example 2: Loop with simple dependency. Final value: " << b[N-1] << "\n";


    // Example 3: Nested loops with no outer loop dependency
    std::vector<std::vector<int>> c(M, std::vector<int>(K, 0));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
           c[i][j] = i + j;
        }
    }

   std::cout << "Example 3: Nested loops with no dependency. Final value: " << c[M-1][K-1] << "\n";

    // Example 4: Nested loop with dependency in inner loop
     std::vector<std::vector<int>> d(M, std::vector<int>(K, 0));
    for (int i = 0; i < M; ++i) {
         d[i][0] = i;
        for (int j = 1; j < K; ++j) {
          d[i][j] = d[i][j-1] + 1;
        }
    }

    std::cout << "Example 4: Nested loop with inner dependency. Final value: " << d[M-1][K-1] << "\n";
    
     // Example 5: Nested loops with dependency in outer loop
     std::vector<std::vector<int>> e(M, std::vector<int>(K, 0));
     e[0][0] = 1;
    for (int i = 1; i < M; ++i) {
        e[i][0] = e[i-1][0] + 1;
        for (int j = 0; j < K; ++j) {
           e[i][j] = e[i][0] + j;
        }
    }
   std::cout << "Example 5: Nested loops with outer dependency. Final value: " << e[M-1][K-1] << "\n";
   

    // Example 6: Loop with dependency and also outside dependency
    int f = 100;
    std::vector<int> g(N);
    g[0] = f;
    for (int i = 1; i < N; ++i) {
        g[i] = g[i-1] + f;
        f = f+1; // f also modified
    }

    std::cout << "Example 6: Loop with outer variable dependency. Final value: " << g[N-1] << "\n";

    return 0;
}