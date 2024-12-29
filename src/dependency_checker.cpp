#include "dependency_checker.h"
#include <iostream>
#include <fstream>
#include <regex>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stack>

// Node for dependency graph
struct Node {
    std::string name;
    // Flags used for cycle detection
    bool visiting = false;
    bool visited = false;
    std::vector<Node*> dependencies;    //  node pointers

    Node(const std::string& name) : name(name) {}
};

// trim whitespace from the beginning and end of a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

// Function to extract variable names from expressions
std::vector<std::string> extractVariables(const std::string& expression) {
    std::regex varRegex(R"(\b[a-zA-Z_][a-zA-Z0-9_]*\b)"); // Regex with word boundaries
    std::vector<std::string> variables;
    std::smatch matches;
    std::string s = expression;
    std::unordered_set<std::string> seenVariables;  // Track unique variables

    while (std::regex_search(s, matches, varRegex)) {
        std::string varName = matches.str();
        if (seenVariables.find(varName) == seenVariables.end()) {
            variables.push_back(varName);
            seenVariables.insert(varName);
        }
        s = matches.suffix();
    }
    return variables;
}

// For circular dependency
// Helper function to detect array access with index as i+constant
std::string extractArrayIndex(const std::string& assignment, int & offset) {
    std::regex arrayRegex(R"([a-zA-Z_][a-zA-Z0-9_]*\[i([+-]\d+)?\])");
    std::smatch matches;
    if (std::regex_search(assignment, matches, arrayRegex)) {
        std::string indexString = matches[1].str(); // Extract "i+1"
        if (indexString.empty()) {
            offset = 0;
        } else {
            try {
                offset = std::stoi(indexString);
            } catch (const std::invalid_argument& e) {
                offset = 0; // Default to 0 if not convertible
                return ""; //If not integer return empty string.
            }
        }

        return matches[0].str().substr(0, matches[0].str().find("[")); // return only array name
    }
    return "";
}

// Helper function for DFS cycle detection (Iterative)
bool dfs_iterative(Node* startNode, std::unordered_set<Node*>& visited) {
    std::stack<Node*> stack;
    std::unordered_set<Node*> visiting;

    stack.push(startNode);
    visiting.insert(startNode);

    while (!stack.empty()) {
        Node* node = stack.top();
        
        if (node->visiting == false) {
             node->visiting = true;
             for (Node* dep : node->dependencies) {
                 if (visiting.count(dep)) {
                    return true; // Cycle detected
                 }
                 if (!visited.count(dep)) {
                    stack.push(dep);
                    visiting.insert(dep);
                  }
             }
          } else {
             stack.pop();
             visiting.erase(node);
             visited.insert(node);
           }

     }
     return false; // No cycle found
}


// Main dependency checking function
bool dependency_checker(const std::string &filePath) {
    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open source file.\n";
        return false;
    }

    std::string line;
    std::regex loopRegex(R"(for\s*\(.*\))");
    std::regex assignmentRegex(R"((([a-zA-Z_][a-zA-Z0-9_]*)(\[[^\]]*\])?)\s*=\s*(.*);)");

    // Keep track of current loops
    std::vector<std::vector<std::string>> loopScopes;

    // Track variables and dependencies
    std::unordered_map<std::string, Node*> nodes;

    // For checking dependencies within loop
    bool inLoop = false;

    // Parse and analyze line by line for dependencies
    while (std::getline(inputFile, line)) {
        if (std::regex_search(line, loopRegex)) {
            loopScopes.push_back({});
            inLoop = true;
            continue;
        }
        if (line.find("}") != std::string::npos && inLoop) {
            if (!loopScopes.empty()) {
                loopScopes.pop_back();
                inLoop = false;
            }
            continue;
        }

        std::smatch matches;
        if (std::regex_search(line, matches, assignmentRegex) && inLoop) {
            std::string fullAssignment = matches[0].str();
            std::string lhs = matches[1].str(); // LHS of the expression
            std::string variableName = trim(matches[2].str()); // var name before array index
            std::string expression = trim(matches[4].str()); // RHS


            // Check if it's an array on LHS (e.g., b[i+1])
            int offset;
            std::string arrayName = extractArrayIndex(lhs, offset);

            if (!arrayName.empty()) {
                if (nodes.find(arrayName) == nodes.end()) {
                    nodes[arrayName] = new Node(arrayName);
                }
                if (offset != 0) {
                    nodes[arrayName]->dependencies.push_back(nodes[arrayName]);
                }
            } else {
                // Handle normal variable assignment
                if (nodes.find(variableName) == nodes.end()) {
                    nodes[variableName] = new Node(variableName);
                }
            }

            // Get variable dependencies from RHS
            std::vector<std::string> usedVariables = extractVariables(expression);

            // Add used variables as dependency to the assigned variable node
            for (const auto& usedVar : usedVariables) {
                // Create the dependency if it does not exist
                if (nodes.find(usedVar) == nodes.end()) {
                    nodes[usedVar] = new Node(usedVar);
                }
                if (!arrayName.empty()) {
                    nodes[arrayName]->dependencies.push_back(nodes[usedVar]);
                } else {
                    nodes[variableName]->dependencies.push_back(nodes[usedVar]);
                }
            }

            if (!arrayName.empty()) {
                loopScopes.back().push_back(arrayName); // Add to the scope
            } else {
                loopScopes.back().push_back(variableName); // Add to the scope
            }

        }

    }

    inputFile.close();

    // Cycle Detection
    std::unordered_set<Node*> visitedNodes;

    for (auto const& [key, val] : nodes) {
        if (!visitedNodes.count(val)) {
            if (dfs_iterative(val, visitedNodes)) {
                // Clean up
                for (auto const& [key, val] : nodes) {
                    delete val;
                }
                return false; // Cycle found
            }
        }
    }
    // Clean up memory
    for (auto const& [key, val] : nodes) {
        delete val;
    }
    return true; // No loop-carried dependencies
}