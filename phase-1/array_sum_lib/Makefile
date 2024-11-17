# Compiler and flags
CC = gcc

# Compiler flags
CFLAGS = -std=c11 -Wall -I$(INCDIR)
LDFLAGS = -fopenmp
LIBFLAGS = -shared

# AVX flags
AVXFLAGS = -mavx2  # Explicitly enable AVX for AVX-specific files

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
	@mkdir -p $(OBJDIR) $(BINDIR)

# Compile AVX-specific file with AVX flags
$(OBJDIR)/avx_sum.o: $(SRCDIR)/avx_sum.c $(HEADERS)
	@echo "Compiling AVX-enabled file: avx_sum.c"
	$(CC) $(CFLAGS) $(AVXFLAGS) -c $< -o $@

# Compile OpenMP file
$(OBJDIR)/openmp_sum.o: $(SRCDIR)/openmp_sum.c $(HEADERS)
	@echo "Compiling OpenMP file: openmp_sum.c"
	$(CC) $(CFLAGS) -c $< -o $@

# Compile general C file
$(OBJDIR)/array_sum.o: $(SRCDIR)/array_sum.c $(HEADERS)
	@echo "Compiling general C file: array_sum.c"
	$(CC) $(CFLAGS) -c $< -o $@

# Link object files to create a shared library
$(LIBRARY): $(C_OBJS)
	@echo "Linking object files to create shared library..."
	$(CC) $(LIBFLAGS) $(C_OBJS) -o $(LIBRARY)

# Test the library with test.c file
test: $(LIBRARY) test.c
	@echo "Compiling and linking test program: test.c"
	$(CC) test.c $(CFLAGS) -L$(BINDIR) -larray_sum -o $(BINDIR)/test $(LDFLAGS)

# Clean up
clean:
	@echo "Cleaning up generated files and directories..."
	@rm -rf $(OBJDIR) $(BINDIR)



