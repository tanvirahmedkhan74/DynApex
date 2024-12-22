# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -fopenmp -O2
PROFILING_FLAGS = -g  # -g is used to generate debug information for Valgrind

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
LOGS_DIR = logs

# Source files
SRC = $(SRC_DIR)/main.cpp
DEPENDENCY_CHECKER = $(SRC_DIR)/dependency_checker.cpp
PRAGMA_INJECTOR = $(SRC_DIR)/pragma_injector.cpp
DISPATCHER = $(SRC_DIR)/dispatcher.cpp

# Outputs
OUTPUT = $(BUILD_DIR)/dispatcher
PROFILING_OUTPUT = $(BUILD_DIR)/dispatcher_prof

# Rules
all: prepare $(OUTPUT)

prepare:
	mkdir -p $(BUILD_DIR) $(LOGS_DIR)

# Normal compilation (no profiling)
$(OUTPUT): $(DEPENDENCY_CHECKER) $(PRAGMA_INJECTOR) $(DISPATCHER)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $(OUTPUT) $(DEPENDENCY_CHECKER) $(PRAGMA_INJECTOR) $(DISPATCHER)

# Profiling-enabled compilation (using Valgrind)
$(PROFILING_OUTPUT): prepare $(DEPENDENCY_CHECKER) $(PRAGMA_INJECTOR) $(DISPATCHER)
	$(CXX) $(CXXFLAGS) $(PROFILING_FLAGS) -I$(INCLUDE_DIR) -o $(PROFILING_OUTPUT) $(DEPENDENCY_CHECKER) $(PRAGMA_INJECTOR) $(DISPATCHER)

# Run dispatcher normally
run-dispatcher: prepare
	$(OUTPUT) $(SRC) c # Change 'r' to 'n', 'c', or 'g' for specific modes

# Profile and generate dependency graph using Valgrind
profile-dispatcher: $(PROFILING_OUTPUT)
	@echo "Running dispatcher with Valgrind profiling in mode $(MODE)..."
	valgrind --tool=callgrind ./$(PROFILING_OUTPUT) $(SRC) $(MODE)
	gprof2dot -w -f callgrind -n10 -s callgrind.out.* -o $(LOGS_DIR)/valgrind.dot
	dot -Tpng $(LOGS_DIR)/valgrind.dot -o $(LOGS_DIR)/dependency_graph_$(OUTPUT_NAME).png
	rm -f callgrind.out.*

# Comparison modes
compare: compare-normal compare-cpu compare-gpu

compare-normal:
	$(MAKE) profile-dispatcher MODE=n OUTPUT_NAME=normal

compare-cpu:
	$(MAKE) profile-dispatcher MODE=c OUTPUT_NAME=cpu

compare-gpu:
	@echo "Removing OpenMP pragmas for GPU mode..."
	sed -i '/#pragma omp/d' $(SRC)
	$(MAKE) profile-dispatcher MODE=g OUTPUT_NAME=gpu

# Clean files
clean:
	rm -rf $(BUILD_DIR) $(LOGS_DIR) callgrind.out.*
	$(MAKE) reset-src

reset-src:
	sed -i '/#pragma omp/d' $(SRC)