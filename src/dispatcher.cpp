#include "dependency_checker.h"
#include "pragma_injector.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <array>
#include <memory>
#include <iomanip>

void log_execution(const std::string &mode, double executionTime) {
    std::ofstream logFile("logs/execution_log.txt", std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Unable to write to log file.\n";
        return;
    }

    // logFile << "======================== Execution Log ========================\n";
   
    logFile << "Run Mode       : " << (mode == "n" ? "Normal" : (mode == "c" ? "CPU" : "GPU")) << "\n";
    logFile << "Execution Time : " << executionTime << " seconds\n";
    logFile.close();
}

void log_gpu_stats() {
    const std::string logFolder = "logs/";
    const std::string gpuLogFilePath = logFolder + "gpu_stat.csv";

    // Ensure the logs directory exists
    system(("mkdir -p " + logFolder).c_str());

    std::ofstream gpuLogFile(gpuLogFilePath, std::ios::app);
    if (!gpuLogFile.is_open()) {
        std::cerr << "Error: Unable to write to GPU log file.\n";
        return;
    }

    // Log headers if the file is empty
    gpuLogFile.seekp(0, std::ios::end);
    if (gpuLogFile.tellp() == 0) {
        gpuLogFile << "Timestamp,Temperature (C),Power (W),Memory Used (MB),Utilization (%),Clock Speed (MHz)\n";
    }

    // Run nvidia-smi command and capture the output
    std::array<char, 256> buffer;
    std::string command = "nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits -i 0"; // GPU 0 stats
    std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
    
    if (!pipe) {
        std::cerr << "Error: Unable to run nvidia-smi command.\n";
        return;
    }

    std::string result;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    // Write the stats into the CSV file
    gpuLogFile << result;
    gpuLogFile.close();
}

void compile_and_execute(const std::string &filePath, char executionMode) {
    const std::string outputBinary = "build/main_parallel";
    const std::string compileCommand = (executionMode == 'n') 
                                        ? "g++ -o " + outputBinary + " " + filePath 
                                        : "g++ -fopenmp -o " + outputBinary + " " + filePath;

    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    // log the start time
    std::ofstream logFile("logs/execution_log.txt", std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Unable to write to log file.\n";
        return;
    }
    logFile << "======================== Execution Log ========================\n";
    // logFile << "Start Time     : " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
    auto startTimeFormatted = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logFile << "Start Time     : " << std::put_time(std::localtime(&startTimeFormatted), "%c %Z") << "\n";
    if (std::system(compileCommand.c_str()) != 0) {
        std::cerr << "Compilation failed!\n";
        return;
    }
    logFile.close();

    // Run the program
    std::string executeCommand = outputBinary;
    if (executionMode == 'g') {
        // Run GPU-specific actions
        std::thread gpuLogThread([&]() {
            while (true) {
                log_gpu_stats();
                std::this_thread::sleep_for(std::chrono::seconds(1)); // Log every second
            }
        });

        gpuLogThread.detach();  // Detach thread to continue logging GPU stats while running
    }

    auto executionStart = std::chrono::high_resolution_clock::now();
    if (std::system(executeCommand.c_str()) != 0) {
        std::cerr << "Execution failed!\n";
        return;
    }
    auto executionEnd = std::chrono::high_resolution_clock::now();
    // Calculate execution time
    std::chrono::duration<double> executionTime = executionEnd - executionStart;
    log_execution(executionMode == 'g' ? "GPU" : (executionMode == 'c' ? "CPU" : "Normal"), executionTime.count());

    // Stop logging GPU stats
    if (executionMode == 'g') {
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Wait for the last log
        log_gpu_stats();  // Log one last time
    }

    // reopne the log file to log the end time
    logFile.open("logs/execution_log.txt", std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Unable to write to log file.\n";
        return;
    }
      
    // log the end time
    // logFile << "End Time       : " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
    // format the end time
    auto endTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logFile << "End Time       : " << std::put_time(std::localtime(&endTime), "%c %Z") << "\n";

    logFile.close();
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./dispatcher <source_file> <mode: n|c|g|r>\n";
        return 1;
    }

    const std::string sourceFilePath = argv[1];
    char executionMode = argv[2][0];

    if (executionMode == 'r') {
        // Randomly decide between CPU ('c') and GPU ('g')
        std::srand(std::time(nullptr));
        executionMode = (std::rand() % 2 == 0) ? 'c' : 'g';
    }

    if (executionMode == 'n') {
        std::cout << "Running program in Normal mode (no pragmas).\n";
        compile_and_execute(sourceFilePath, executionMode);
        return 0;
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
