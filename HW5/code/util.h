/**
 * Some utility functions.
 */

#pragma once

#include <Eigen/Sparse>
#include <chrono>

constexpr double PI = std::numbers::pi;

/**
 * @brief Generate the left-hand side matrix of the Poisson equation.
 *
 * @param N The number of grid points in each dimension.
 * @return The left-hand side matrix in sparse format.
 */
Eigen::SparseMatrix<double> make_matrix(int N);

/**
 * @brief Generate the right-hand side vector of the Poisson equation.
 *
 * @param N The number of grid points in each dimension.
 * @return The right-hand side vector.
 */
Eigen::VectorXd make_vector(int N);

/**
 * @brief Generate the real solution of the Poisson equation.
 *
 * @param N The number of grid points in each dimension.
 * @return The real solution of the Poisson equation.
 */
Eigen::VectorXd real_solution(int N);


enum class Method { LDL, GAUSS_SEIDEL, CONJUGATE_GRADIENT, MULTI_GRID };

/**
 * @brief Parse the command line arguments.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 * @return The method to solve the Poisson equation.
 */
Method parse_argument(int argc, char *argv[]);
