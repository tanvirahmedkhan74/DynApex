# ~ DynApex ~

#### **Table of Contents**

1.  [DynApex: Dynamic Dependency Analysis and Execution Profiling for Parallelization](#dynapex-dynamic-dependency-analysis-and-execution-profiling-for-parallelization)
2.  [Key Features](#key-features)
3.  [System Architecture](#system-architecture)
    *   [Directory Hierarchy](#directory-hierarchy)
4.  [Package Installation on Ubuntu](#package-installation-on-ubuntu)
5.  [Build and Run](#build-and-run)
6.  [Example Execution Results](#example-execution-results)
7.  [Limitations](#limitations)
8.  [Future Improvements](#future-improvements)

## DynApex: Dynamic Dependency Analysis and Execution Profiling for Parallelization

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](BUILD_STATUS)

**DynApex** is a sophisticated tool designed for automated source code parallelization. By employing dynamic analysis and dependency checking, DynApex identifies opportunities for parallel execution and intelligently dispatches tasks to either the CPU or GPU. This system emphasizes peak performance by making runtime decisions informed by profiling data, thereby optimizing execution based on the specific characteristics of the code and the available hardware.

## Key Features

*   **Advanced Data Dependency Analysis**: Utilizes a robust algorithm to meticulously analyze C++ code, ensuring that only safe and parallelizable sections are targeted. This analysis prevents race conditions and maintains data integrity during parallel execution.
*   **Intelligent Dynamic Dispatch**: Leverages profiling data to dynamically select the most suitable execution platform—either the CPU or GPU—maximizing performance based on the real-time behavior of the program.
*   **Comprehensive Profiling and Visualization**: Provides detailed performance profiles and generates dependency graphs, which are crucial for understanding execution bottlenecks and dependency structures.
*   **OpenMP Parallelization**: Implements parallel processing on the CPU through seamless integration with OpenMP, a widely adopted standard for shared-memory parallel programming. *(Note: GPU support currently uses OpenMP target offloading but is not fully optimized for full GPU frameworks like CUDA or OpenCL.)*
*   **Flexible Execution Modes**: Supports multiple execution modes (Normal, CPU, and GPU) to allow for comparison and performance analysis of the impact of parallelization strategies.

## System Architecture

The DynApex project is structured into the following directory layout:
### Directory Hierarchy

```
dynamic_dispatcher/
├── include/       // Header files
│   └── ...
├── src/          // Source code
│   ├── main.cpp    // Main program
│   ├── dependency_checker.cpp // Dependency analysis
│   ├── pragma_injector.cpp // OpenMP pragma injection
│   └── dispatcher.cpp       // Dynamic dispatcher
├── build/        // Compiled binaries
├── logs/         // Profiling logs and graphs
└── Makefile      // Build instructions
```

## **Package Installation on Ubuntu**

1.  **Update Package Lists:**

    ```bash
    sudo apt update
    ```

2.  **Install Essential Build Tools:**
    ```bash
    sudo apt install build-essential
    ```

3.  **Install g++ Compiler (with OpenMP support):**
    ```bash
    sudo apt install g++
    ```

4.  **Install GNU Make:**
    ```bash
    sudo apt install make
    ```

5.  **Install Valgrind (for profiling):**

    ```bash
    sudo apt install valgrind
    ```

6.  **Install Graphviz (for dependency graph visualization):**

    ```bash
    sudo apt install graphviz
    ```

7.  **Install `gprof2dot` (for call graph generation):**

    ```bash
    sudo pip install gprof2dot
    ```
8.  **Install NVIDIA Drivers and CUDA Toolkit (if you intend to utilize GPU):**
    *   First, verify that you have a compatible NVIDIA GPU and that the appropriate drivers are not already installed:
        ```bash
         lspci | grep -i nvidia
        ```
    *   If no output from the command above, it is highly probable that NVIDIA drivers are not installed.
    *   Install the NVIDIA driver with the recommended version.
        ```bash
        sudo apt install nvidia-driver-<version>
        ```
         (replace `<version>` with the recommended driver version of your NVIDIA GPU)
        *You can also use `ubuntu-drivers devices` to list your GPU's available driver options*
    *   Install the CUDA Toolkit. Download the installer from NVIDIA's website ([https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads))
    *  Follow NVIDIA's installation steps for your specific toolkit version
    *   Alternatively, install using:
      ```bash
        sudo apt install nvidia-cuda-toolkit
      ```

9.  **Verify NVIDIA Installation:**
     *  Verify that you have the installation of `nvidia-smi` working properly
      ```bash
         nvidia-smi
      ```
     * If this does not work, reboot the system.

10. **Check for OpenMP support (after installing g++):**

    *   Create a test file (e.g., `test_omp.cpp`) with the following content:
        ```cpp
        #include <iostream>
        #include <omp.h>

        int main() {
            #pragma omp parallel
            {
                std::cout << "Hello from thread " << omp_get_thread_num() << std::endl;
            }
            return 0;
        }
        ```
    *   Compile and run the test:
        ```bash
        g++ -fopenmp test_omp.cpp -o test_omp
        ./test_omp
        ```
        *   If you see output from multiple threads (e.g., "Hello from thread 0," "Hello from thread 1," etc.), OpenMP is working correctly. If you only see "Hello from thread 0", OpenMP is not enabled, try re-installing `g++`.

**Important Notes:**

*   **CUDA Version**: Ensure that the CUDA toolkit version you install is compatible with your NVIDIA driver.
*   **NVIDIA Drivers:** Ensure the NVIDIA drivers are correctly installed and working; otherwise, GPU usage will not be possible. You may need to install them separately from the CUDA toolkit.
*   **`nvidia-smi`:** Verify that you can run `nvidia-smi` to check that your drivers and GPU are working properly. It will also show the CUDA version in the output.
*  **C++ with OpenMP:** If needed, you can include this flag to check if OpenMP flags are working, using `g++ -fopenmp -v`.
*   **Alternatives**: If `apt` does not have the latest CUDA version, you will have to follow NVIDIA's instructions which typically involves adding their repository and then installing via `apt`.
*   **Test Environment:** Consider testing the installation with NVIDIA's samples.
*   **Permissions:** Depending on the NVIDIA installation, you may have to reboot the system.
* **Troubleshooting**: If there are issues with the installation, check for driver and CUDA errors; check the installation documentation for the drivers, and ensure that you have installed the correct versions.
*  **OpenMP**: Make sure the test program provided to test OpenMP compiles and runs correctly. If not, there may be a problem with `g++` and OpenMP, make sure you have installed `g++` with OpenMP support, and if there are still problems, consult online forums for possible solutions.


## Build and Run

### Compile Normally:

Run the program without profiling:

```bash
make
make run-dispatcher 
```

### Profile Dispatcher:

Recompile with profiling (`-pg`), run, and generate a dependency graph:

```bash
make profile-dispatcher MODE=n 
```
(Replace `n` with `c` for CPU-optimized or `g` for GPU-optimized mode).


### Compare All Modes:

Automates profiling and graph generation for all modes:

```bash
make compare
```

### Clean Project:

Remove binaries, logs, and temporary files:

```bash
make clean
```

Okay, here's the concise and smaller version of the example log output and limitations, suitable for a README file:

## **Example Execution Results**

```
------------------------------------------------------
|           DynApex Results                         |
------------------------------------------------------
| Mode: Normal, Time: 6.40s                         |
------------------------------------------------------
| Mode: CPU:                                        |
| Time: 4.68s                                       |
| Actual Speedup: 1.37                              |
| Calculated Speedup (Amdahl's): 2.11               |
------------------------------------------------------
| Mode: GPU                                         |
| Time: 2.75s                                       |
| Actual Speedup: 2.33                              |          
| Calculated Speedup (Amdahl's): 3.26               |
------------------------------------------------------
```

**Key:**

*   `Mode`: Execution mode (Normal, CPU, GPU).
*   `Time`: Execution time in seconds.
*   `Actual Speedup`:  Speedup vs. Normal mode.
*   `Calculated Speedup (Amdahl's)`: Theoretical speedup.

## **Limitations**

*   **Dependency Analysis**: Limited handling of complex dependencies (inter-procedural, advanced array indexing).
*  **GPU Support**: Not fully optimized for GPU utilization, memory management, or full CUDA/OpenCL usage.
*   **Auto-Tuning**: No dynamic tuning of parallelization parameters (e.g. threads), or auto detection of parallelizable portion of the code.
*   **Loop Resolution**: Cannot automatically resolve loop-carried dependencies (e.g. loop transformations).
*   **Static Analysis**: Incomplete static analysis might miss runtime dependencies.
*   **Complexity**: May struggle with very large, complex code.
*   **General Purpose**: System is designed for a specific program and not a generic one.

## **Future Improvements**

-   [x] **Enhanced GPU Support**: Full integration with CUDA or OpenCL for optimal GPU performance.
-   [ ] **Automatic Dependency Resolution**: Ability to automatically remove or resolve detected dependencies via loop transformations and data restructuring (`--rd` flag).
-   [ ] **Advanced Profiling Tools**: Implementation of more sophisticated techniques for analyzing and visualizing performance data.
-   [ ] **Improved Auto-Tuning**: Incorporating auto-tuning mechanisms to dynamically adjust parallelization parameters based on system characteristics.
-   [ ] **Inter-Procedural Dependency Analysis**: Extending the dependency analysis to recognize inter-procedural dependencies for more effective parallelization.
-   [ ] **Complex Array Access Handling**: Handling of complex array access patterns in dependency checker.
-   [ ] **General Purpose**: Enhance the tool so that it works for any c++ program.