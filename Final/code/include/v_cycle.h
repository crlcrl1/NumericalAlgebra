#ifndef V_CYCLE_H
#define V_CYCLE_H

#include <Eigen/Sparse>

using CRSMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

int multiGridSolver(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                    const Eigen::VectorXd &f, const Eigen::VectorXd &d, int n, int v1, int v2,
                    double tol);

std::pair<CRSMatrix, CRSMatrix> restrictOperator(int n);

#endif // V_CYCLE_H
