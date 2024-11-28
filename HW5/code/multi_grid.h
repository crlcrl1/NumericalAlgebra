/**
 * Solve the Poisson equation using the multi-grid method.
 */

#pragma once

#include <Eigen/Sparse>

/**
 * @brief Solve the Poisson equation using the multi-grid method.
 *
 * @param N the number of grid points in one dimension.
 * @param v1 the number of Gauss-Seidel iterations before restriction and prolongation.
 * @param v2 the number of Gauss-Seidel iterations after restriction and prolongation.
 * @param tol the tolerance for the residual.
 * @return a pair of the solution u and the number of iterations.
 */
std::pair<Eigen::VectorXd, int> multi_grid_solver(int N, int v1, int v2, double tol = 1e-6);
