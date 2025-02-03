#include "cli.h"

#include "equation.h"

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <format>
#include <iostream>
#include <string>

#include "util.h"

constexpr double PI = std::numbers::pi;

double f(const double x, const double y) {
    return -4.0 * PI * PI * std::sin(2 * PI * y) * (2 * std::cos(2 * PI * x) - 1) + x * x;
}

double g(const double x, const double y) {
    return 4.0 * PI * PI * std::sin(2 * PI * x) * (2 * std::cos(2 * PI * y) - 1);
}

[[noreturn]] void showError(const std::string &message) {
    std::cerr << message << std::endl;
    exit(1);
}

Config::Config(const int argc, char **argv) {
    int i = 1;
    maxThreads = -1;
    bool methodSet = false;
    while (i < argc) {
        if (std::string(argv[i]) == "-t") {
            if (i + 1 >= argc) {
                showError("Missing argument for -t");
            }
            maxThreads = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "-m") {
            if (i + 1 >= argc) {
                showError("Missing argument for -m");
            }
            auto method = std::string(argv[++i]);
            std::ranges::transform(method, method.begin(), tolower);
            if (method == "uzawaldl") {
                this->method = Method::UzawaLDL;
            } else if (method == "uzawacg") {
                this->method = Method::UzawaCG;
            } else if (method == "uzawapcg") {
                this->method = Method::UzawaPCG;
            } else if (method == "multigrid") {
                this->method = Method::MultiGrid;
            } else {
                showError("Invalid method");
            }
            methodSet = true;
        } else if (std::string(argv[i]) == "-h") {
            constexpr auto help = R"(
Usage: {} [options]

Options:
    -t <threads>    Set the maximum number of threads
    -m <method>     Set the method to use. Available methods are:
                        - UzawaLDL
                        - UzawaCG
                        - UzawaPCG
                        - MultiGrid
    -h              Display this help message
)";
            std::cout << std::format(help, argv[0]);
            exit(0);
        } else {
            showError("Invalid argument, use -h for help");
        }
        i++;
    }
    if (!methodSet) {
        showError("Method not set");
    }
}

void applyMultiGrid() {
    for (constexpr std::array ns = {64, 128, 256, 512, 1024, 2048, 4096}; const auto n: ns) {
        Equation eq(n, f, g);
        const auto start = std::chrono::high_resolution_clock::now();
        const auto [resU, resV, resP, iterNum] = eq.solveMultiGrid(2, 2, 1e-8);
        const auto end = std::chrono::high_resolution_clock::now();
        std::cout << std::format("n = {}, Iteration count: {}, Error: {:.6e}, Time: {:.6f}s", n,
                                 iterNum, error(resU, resV, n),
                                 std::chrono::duration<double>(end - start).count())
                  << std::endl;
    }
}

void applyUzawaLDL() {
    for (constexpr std::array ns = {64, 128, 256, 512}; const auto n: ns) {
        Equation eq(n, f, g);
        const auto start = std::chrono::high_resolution_clock::now();
        const auto [resU, resV, resP, iterNum] = eq.solveUzawaLDL(1e-8);
        const auto end = std::chrono::high_resolution_clock::now();
        std::cout << std::format("n = {}, Iteration count: {}, Error: {:.6e}, Time: {:.6f}s", n,
                                 iterNum, error(resU, resV, n),
                                 std::chrono::duration<double>(end - start).count())
                  << std::endl;
    }
}

void applyUzawaCG() {
    for (constexpr std::array ns = {64, 128, 256, 512}; const auto n: ns) {
        Equation eq(n, f, g);
        const auto start = std::chrono::high_resolution_clock::now();
        const auto [resU, resV, resP, iterNum] = eq.solveUzawaCG(1e-8);
        const auto end = std::chrono::high_resolution_clock::now();
        std::cout << std::format("n = {}, Iteration count: {}, Error: {:.6e}, Time: {:.6f}s", n,
                                 iterNum, error(resU, resV, n),
                                 std::chrono::duration<double>(end - start).count())
                  << std::endl;
    }
}

void applyUzawaPCG() {
    for (constexpr std::array ns = {64, 128, 256, 512, 1024, 2048}; const auto n: ns) {
        Equation eq(n, f, g);
        const auto start = std::chrono::high_resolution_clock::now();
        const auto [resU, resV, resP, iterNum] = eq.solveUzawaPCG(2, 2, 1e-8);
        const auto end = std::chrono::high_resolution_clock::now();
        std::cout << std::format("n = {}, Iteration count: {}, Error: {:.6e}, Time: {:.6f}s", n,
                                 iterNum, error(resU, resV, n),
                                 std::chrono::duration<double>(end - start).count())
                  << std::endl;
    }
}

void Config::apply() const {
    Eigen::initParallel();
    Eigen::setNbThreads(maxThreads);

    if (maxThreads == -1) {
        std::cout << "Using default number of threads" << std::endl;
    } else {
        std::cout << "Using " << maxThreads << " threads" << std::endl;
        omp_set_num_threads(maxThreads);
    }

    switch (method) {
        case Method::MultiGrid:
            std::cout << "Using MultiGrid method" << std::endl;
            applyMultiGrid();
            break;
        case Method::UzawaCG:
            std::cout << "Using UzawaCG method" << std::endl;
            applyUzawaLDL();
            break;
        case Method::UzawaLDL:
            std::cout << "Using UzawaLDL method" << std::endl;
            applyUzawaCG();
            break;
        case Method::UzawaPCG:
            std::cout << "Using UzawaPCG method" << std::endl;
            applyUzawaPCG();
            break;
    }
}
