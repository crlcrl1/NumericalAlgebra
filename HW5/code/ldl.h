/**
 * Solve the Poisson equation using the LDL^T decomposition.
 */

#pragma once

#include <Eigen/Sparse>

Eigen::VectorXd ldl_solver(int N);
