#ifndef UTIL_H
#define UTIL_H

#include <Eigen/Sparse>

using CRSMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

CRSMatrix initA(int n);

CRSMatrix initB(int n);

double error(const Eigen::MatrixXd &u, const Eigen::MatrixXd &v, int n);

void logError(double error, int iter);

void resetLog();

CRSMatrix restrictU(int n);

CRSMatrix restrictF(int n);

void gaussSeidel(const CRSMatrix &A, Eigen::VectorXd &u, const Eigen::VectorXd &b);

void parallelGaussSeidel(const CRSMatrix &A, Eigen::VectorXd &u, const Eigen::VectorXd &b);

#endif // UTIL_H
