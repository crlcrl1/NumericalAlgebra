#pragma once

#include <Eigen/Sparse>

/**
 * Solve Poisson equation using Gauss-Seidel method.
 *
 * @param N the number of grid points in one dimension.
 * @param tol the tolerance for the residual.
 * @return
 */
std::pair<Eigen::VectorXd, int> gauss_seidel_solver(int N, double tol = 1e-6);
