# Hybrid CPU-GPU Accelerated Query Processing for Dynamic Data Structures

**October 19, 2024**

---

## Project Overview

This project focuses on developing a **Generic Framework for Energy-Efficient Optimization** in query processing using a hybrid approach that leverages both CPU and GPU capabilities. The framework aims to enhance performance and energy efficiency when dealing with dynamic data structures.

---

## Authors

- **Tanvir Ahmed Khan**  
  Email: [khan.tanvir01@northsouth.edu](mailto:khan.tanvir01@northsouth.edu)

- **Mahir Shahriar Tamim**  
  Email: [mahir.tamim@northsouth.edu](mailto:mahir.tamim@northsouth.edu)

- **Mahiyat Nawar Mantaqa**  
  Email: [mahiyat.mantaqa@northsouth.edu](mailto:mahiyat.mantaqa@northsouth.edu)

- **Maharun Afroz**  
  Email: [maharun.afroz@northsouth.edu](mailto:maharun.afroz@northsouth.edu)

---

## Course Instructor

**Dr. Md Shahriar Karim**  
Associate Professor, Department of Computer Science

---
---

## Requirements

To run this project, you need to have the following tools and libraries installed:

- **CUDA Toolkit**
- **OpenMP**
- **GCC**
- **NVCC (NVIDIA CUDA Compiler)**
- **Windows SDK**
- **Make**
- **AVX Specification**

---

## Build Instructions for Phase-1

## Necessary Changes in Makefile

To ensure proper compilation and linking of the project, you may need to make the following changes to the Makefile.

### Update CFLAGS

Modify the `CFLAGS` variable to include the appropriate compiler flags and include paths:

```makefile
CFLAGS = /std:c11 /W3 /showIncludes /Iinclude \
         /IC:"<path_to_your_include_directory>"
```

### Important Notes

- Make sure to replace `<path_to_your_include_directory>` with the correct paths for your system.

- **Update CUDAFLAGS (if needed)**  
  If you are using CUDA, ensure that the `CUDAFLAGS` variable is set correctly:

## Necessary Changes in build.bat

To configure the build environment properly, you may need to make the following updates to the `build.bat` script.

### Set INCLUDE Path

Add the following line to set the `INCLUDE` environment variable:

```batch
set INCLUDE=<path_to_your_include_directory>;%INCLUDE%
```

To build the project, follow these steps:

1. **Clone the Repository**:

   First, clone the repository to your local machine using the following command:

   ```bash
   git clone <repository-url>

2. **Set Up Environment Variables**:

   Make sure to add the following path to your system environment variables:

   - **Include Path**: `C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt`

   This will ensure that the compiler can locate the necessary headers.

---
3. **Run the Build Script**:

   Navigate to the project directory and execute the build script:

   ```bash
   ./build.bat
