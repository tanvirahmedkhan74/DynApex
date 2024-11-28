#include "pragma_injector.h"
#include <iostream>
#include <fstream>
#include <regex>

bool add_pragma_to_loops(const std::string &filePath, char executionMode) {
    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open source file.\n";
        return false;
    }

    std::string modifiedCode;
    std::string line;
    std::regex loopRegex(R"((for|while)\s*\(.*\))");
    bool pragmaAdded = false;

    while (std::getline(inputFile, line)) {
        if (std::regex_search(line, loopRegex) && !pragmaAdded) {
            if (executionMode == 'c') {
                modifiedCode += "#pragma omp parallel for\n";
            } else if (executionMode == 'g') {
                modifiedCode += "#pragma omp target teams distribute parallel for\n";
            }
            pragmaAdded = true; // Add pragma only to the first loop
        }
        modifiedCode += line + "\n";
    }
    inputFile.close();

    // Write back the modified code
    std::ofstream outputFile(filePath);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to write to source file.\n";
        return false;
    }
    outputFile << modifiedCode;
    outputFile.close();
    return true;
}

