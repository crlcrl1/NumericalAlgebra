#include <chrono>
#include <format>
#include <iostream>

#include "equation.h"
#include "util.h"

constexpr double PI = std::numbers::pi;

double f(const double x, const double y) {
    return -4.0 * PI * PI * std::sin(2 * PI * y) * (2 * std::cos(2 * PI * x) - 1) + x * x;
}

double g(const double x, const double y) {
    return 4.0 * PI * PI * std::sin(2 * PI * x) * (2 * std::cos(2 * PI * y) - 1);
}

int main() {
    Eigen::initParallel();
    Eigen::setNbThreads(-1);

    constexpr int n = 2048;

    Equation eq(n, f, g);

    const auto start = std::chrono::high_resolution_clock::now();
    const auto &[resU, resV, resP, k] = eq.solveMultiGrid(2, 2, 1e-8);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diff = end - start;
    std::cout << std::format("Time: {}s, Iteration count: {}, Error: {}", diff.count(), k,
                             error(resU, resV, n))
              << std::endl;
    return 0;
}
