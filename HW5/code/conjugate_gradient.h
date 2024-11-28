#pragma once

#include <Eigen/Sparse>

std::pair<Eigen::VectorXd, int> conjugate_gradient_solver(int N, double tol = 1e-6);
