# Compiler and flags
CC = gcc

# Compiler flags
CFLAGS = -std=c11 -Wall -Iinclude
LDFLAGS = -fopenmp
AVXFLAGS = -mavx2
LIBFLAGS = -shared

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

# Files
C_SOURCES = $(SRCDIR)/array_sum.c $(SRCDIR)/avx_sum.c $(SRCDIR)/openmp_sum.c
HEADERS = $(INCDIR)/array_sum.h

C_OBJS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(C_SOURCES))

# Target library
LIBRARY = $(BINDIR)/libarray_sum.so

# Rules
all: directories $(LIBRARY)

directories:
	@echo "Creating necessary directories if they do not exist..."
	@mkdir -p $(OBJDIR) $(BINDIR)

# Compile C files, adding AVX flag specifically for avx_sum.c
$(OBJDIR)/avx_sum.o: $(SRCDIR)/avx_sum.c $(HEADERS)
	@echo "Compiling AVX-enabled file: avx_sum.c"
	$(CC) $(CFLAGS) $(AVXFLAGS) -c $< -o $@

# Compile OpenMP file
$(OBJDIR)/openmp_sum.o: $(SRCDIR)/openmp_sum.c $(HEADERS)
	@echo "Compiling OpenMP file: openmp_sum.c"
	$(CC) $(CFLAGS) -c $< -o $@

# Compile general C files
$(OBJDIR)/array_sum.o: $(SRCDIR)/array_sum.c $(HEADERS)
	@echo "Compiling general C file: array_sum.c"
	$(CC) $(CFLAGS) -c $< -o $@

# Link object files together to create a shared library (.so)
$(LIBRARY): $(C_OBJS)
	@echo "Linking object files to create shared library..."
	$(CC) $(LIBFLAGS) $(C_OBJS) -o $(LIBRARY)

# Test the library with a test.c file
test: $(LIBRARY) test.c
	@echo "Compiling and linking test program: test.c"
	$(CC) test.c -Iinclude -L$(BINDIR) -larray_sum -o $(BINDIR)/test $(LDFLAGS)

# Clean up
clean:
	@echo "Cleaning up generated files and directories..."
	@rm -rf $(OBJDIR) $(BINDIR)
