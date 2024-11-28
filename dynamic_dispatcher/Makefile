# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -fopenmp -O2

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

# Output
OUTPUT = $(BUILD_DIR)/dispatcher

# Rules
all: prepare $(OUTPUT)

prepare:
	mkdir -p $(BUILD_DIR) $(LOGS_DIR)

$(OUTPUT): $(DEPENDENCY_CHECKER) $(PRAGMA_INJECTOR) $(DISPATCHER)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $(OUTPUT) $(DEPENDENCY_CHECKER) $(PRAGMA_INJECTOR) $(DISPATCHER)

run-dispatcher:
	$(OUTPUT) $(SRC) r # Change 'r' to 'c' or 'g' for specific modes

clean:
	rm -rf $(BUILD_DIR) $(LOGS_DIR)

