#ifndef UZAWA_H
#define UZAWA_H

#include <Eigen/Sparse>

using CRSMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

int uzawaSolverLDL(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                   const Eigen::VectorXd &f, double tol);

int uzawaSolverCG(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                  const Eigen::VectorXd &f, double tol);

int uzawaSolverPCG(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                   const Eigen::VectorXd &f, int n, int v1, int v2, double tol);

#endif // UZAWA_H
