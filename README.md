# DynApex

### Dynamic Analysis and Execution Profiling for Extreme Parallelization.

##### Emphasizes dynamic analysis and peak performance.


## Overview

**DynApex** is a tool designed to analyze C++ code for parallelization opportunities and dynamically dispatch execution to either the CPU or GPU based on profiling data.  It performs data dependency analysis to determine compatibility with parallel execution and then uses profiling to make informed runtime decisions.

## Features

* **Data Dependency Analysis:**  Analyzes code for data dependencies to ensure safe parallelization.
* **Dynamic Dispatch:**  Selects the optimal execution platform (CPU or GPU) based on profiling results.
* **Profiling and Visualization:** Generates performance profiles and dependency graphs for analysis.
* **OpenMP Support:** Leverages OpenMP for parallel execution on the CPU.  (GPU support requires adapting the code for a suitable framework like CUDA or OpenCL - this is not yet fully implemented).

## Directory Hierarchy

```
dynamic_dispatcher/
├── include/       // Header files
│   └── ...
├── src/          // Source code
│   ├── main.cpp    // Main program
│   ├── dependency_checker.cpp // Dependency analysis
│   ├── pragma_injector.cpp // OpenMP pragma injection (if applicable)
│   └── dispatcher.cpp       // Dynamic dispatcher
├── build/        // Compiled binaries
├── logs/         // Profiling logs and graphs
└── Makefile      // Build instructions
```

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

## Future Improvements

* Full GPU support.
* Automatic dependency removal (`--rd` flag).
* More sophisticated profiling and analysis capabilities.
