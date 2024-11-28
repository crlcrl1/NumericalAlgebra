#include <chrono>
#include <format>
#include <functional>
#include <iostream>

#include <Eigen/Sparse>

#include "conjugate_gradient.h"
#include "gauss_seidel.h"
#include "ldl.h"
#include "multi_grid.h"
#include "util.h"

int main(const int argc, char *argv[]) {
    using Solver = std::function<std::pair<Eigen::VectorXd, int>(int)>;
    using Milliseconds = std::chrono::duration<double, std::milli>;

    const Method method = parse_argument(argc, argv);
    Solver solver;

    switch (method) {
        case Method::LDL:
            solver = [](const int N) { return std::make_pair(ldl_solver(N), -1); };
            std::cout << "LDL solver" << std::endl;
            break;
        case Method::GAUSS_SEIDEL:
            solver = std::bind(gauss_seidel_solver, std::placeholders::_1, 1e-6);
            std::cout << "Gauss-Seidel solver" << std::endl;
            break;
        case Method::CONJUGATE_GRADIENT:
            solver = std::bind(conjugate_gradient_solver, std::placeholders::_1, 1e-6);
            std::cout << "Conjugate Gradient solver" << std::endl;
            break;
        case Method::MULTI_GRID:
            solver = std::bind(multi_grid_solver, std::placeholders::_1, 3, 2, 1e-6);
            std::cout << "Multi-Grid solver" << std::endl;
            break;
    }

    const auto timed_solver =
            [solver](const int N) -> std::tuple<Eigen::VectorXd, int, Milliseconds> {
        const auto start = std::chrono::high_resolution_clock::now();
        const auto [result, k] = solver(N);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapsed = std::chrono::duration<double, std::milli>(end - start);
        return {result, k, elapsed};
    };

    for (const std::array numbers = {64, 128, 256, 512, 1024}; int N: numbers) {
        const auto [result, k, elapsed] = timed_solver(N);
        const auto u = real_solution(N);
        const double error = (u - result).norm() / N;
        const auto message =
                std::format("N: {}, error: {:.10f}, time: {}, iter:{}", N, error, elapsed, k);
        std::cout << message << std::endl;
    }
}
