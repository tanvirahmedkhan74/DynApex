#include "dependency_checker.h"
#include "pragma_injector.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <chrono>

void log_execution(const std::string &mode, double executionTime, const std::string &modifiedFileContent) {
    std::ofstream logFile("logs/execution_log.txt", std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Unable to write to log file.\n";
        return;
    }

    logFile << "======================== Execution Log ========================\n";
    logFile << "Run Mode       : " << (mode == "c" ? "CPU" : "GPU") << "\n";
    logFile << "Execution Time : " << executionTime << " seconds\n";
    logFile << "Modified Source Code:\n";
    logFile << "************************************************************\n";
    logFile << modifiedFileContent;
    logFile << "************************************************************\n";
    logFile.close();
}

std::string read_file(const std::string &filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to read the source file.\n";
        return "";
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

void compile_and_execute(const std::string &filePath, char executionMode) {
    std::string outputBinary = "build/main_parallel";
    std::string compileCommand = "g++ -fopenmp -o " + outputBinary + " " + filePath;

    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    if (std::system(compileCommand.c_str()) != 0) {
        std::cerr << "Error: Compilation failed.\n";
        return;
    }

    // Execute the binary
    std::cout << "Running the program...\n";
    if (std::system(("./" + outputBinary).c_str()) != 0) {
        std::cerr << "Error: Execution failed.\n";
        return;
    }

    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> executionTime = endTime - startTime;

    // Read modified file for logging
    std::string modifiedFileContent = read_file(filePath);

    // Log execution details
    log_execution(std::string(1, executionMode), executionTime.count(), modifiedFileContent);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./dispatcher <source_file> <mode: c|g|r>\n";
        return 1;
    }

    std::string sourceFilePath = argv[1];
    char executionMode = argv[2][0];

    if (executionMode == 'r') {
        // Randomly decide between CPU ('c') and GPU ('g')
        std::srand(std::time(nullptr));
        executionMode = (std::rand() % 2 == 0) ? 'c' : 'g';
    }

    std::cout << "Checking dependencies...\n";
    if (!dependency_checker(sourceFilePath)) {
        std::cerr << "Error: Loop dependencies detected. Cannot parallelize.\n";
        return 1;
    }

    std::cout << "Modifying source file for execution mode: "
              << (executionMode == 'c' ? "CPU" : "GPU") << "\n";
    if (!add_pragma_to_loops(sourceFilePath, executionMode)) {
        std::cerr << "Error: Failed to add pragmas to the source file.\n";
        return 1;
    }

    compile_and_execute(sourceFilePath, executionMode);
    return 0;
}

