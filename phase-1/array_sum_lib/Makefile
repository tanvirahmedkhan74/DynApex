# Compiler and flags
CC = cl
NVCC = nvcc

# Explicitly add the path to the Windows Kits UCRT where string.h is located
CFLAGS = /std:c11 /W3 /showIncludes /Iinclude \
         /IC:"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\include" \
         /IC:"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt" \
         /IC:"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared" \
         /IC:"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um" \
         /IC:"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\winrt"

CUDAFLAGS = -Xcompiler /std:c11 -Iinclude
LDFLAGS = /openmp /link
AVXFLAGS = /arch:AVX2
LIBFLAGS = /DLL

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

# Files
C_SOURCES = $(SRCDIR)/array_sum.c $(SRCDIR)/avx_sum.c $(SRCDIR)/openmp_sum.c
CUDA_SOURCES = $(SRCDIR)/cuda_sum.cu
HEADERS = $(INCDIR)/array_sum.h

C_OBJS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.obj, $(C_SOURCES))
CUDA_OBJS = $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.obj, $(CUDA_SOURCES))

# CUDA Library Path and Libraries
CUDA_LIB_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64"
CUDA_LIBS = $(CUDA_LIB_PATH)\cudart.lib

# Target library
LIBRARY = $(BINDIR)/libarray_sum.dll

# Rules
all: directories $(LIBRARY)

directories:
	@echo "Creating necessary directories if they do not exist..."
	@if not exist $(OBJDIR) mkdir $(OBJDIR)
	@if not exist $(BINDIR) mkdir $(BINDIR)

# Compile C files, adding AVX flag specifically for avx_sum.c
$(OBJDIR)/avx_sum.obj: $(SRCDIR)/avx_sum.c $(HEADERS)
	@echo "Compiling AVX-enabled file: avx_sum.c"
	$(CC) $(CFLAGS) $(AVXFLAGS) /c $< /Fo$@

# Compile OpenMP file
$(OBJDIR)/openmp_sum.obj: $(SRCDIR)/openmp_sum.c $(HEADERS)
	@echo "Compiling OpenMP file: openmp_sum.c"
	$(CC) $(CFLAGS) /c $< /Fo$@

# Compile general C files
$(OBJDIR)/array_sum.obj: $(SRCDIR)/array_sum.c $(HEADERS)
	@echo "Compiling general C file: array_sum.c"
	$(CC) $(CFLAGS) /c $< /Fo$@

# Compile CUDA files
$(OBJDIR)/%.obj: $(SRCDIR)/%.cu $(HEADERS)
	@echo "Compiling CUDA file: $<"
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

# Link object files together to create a shared library (DLL) and its import library (.lib)
$(LIBRARY): $(C_OBJS) $(CUDA_OBJS)
	@echo "Linking object files to create DLL and LIB files..."
	link $(LIBFLAGS) $(C_OBJS) $(CUDA_OBJS) $(CUDA_LIBS) /out:"$(LIBRARY)" /implib:"$(BINDIR)/libarray_sum.lib" $(LDFLAGS)

# Test the library with a test.c file
test: $(LIBRARY) test.c
	@echo "Compiling and linking test program: test.c"
	$(CC) test.c /Iinclude /link /LIBPATH:"$(BINDIR)" libarray_sum.lib /out:"$(BINDIR)/test.exe" $(LDFLAGS)

# Clean up
clean:
	@echo "Cleaning up generated files and directories..."
	@if exist $(OBJDIR) rmdir /S /Q $(OBJDIR)
	@if exist $(BINDIR) rmdir /S /Q $(BINDIR)