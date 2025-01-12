#ifndef EQUATION_H
#define EQUATION_H

#include <Eigen/Sparse>
#include <functional>

using CRSMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

class Equation {
public:
    explicit Equation(int n, const std::function<double(double, double)> &f,
                      const std::function<double(double, double)> &g);

    [[nodiscard]] const CRSMatrix &getA() const { return A; }
    [[nodiscard]] const CRSMatrix &getB() const { return B; }
    [[nodiscard]] const Eigen::VectorXd &getF() const { return f; }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, int>
    solveMultiGrid(int v1, int v2, double tol);

private:
    const int n;
    const double h;

    // Coefficients Matrix
    const CRSMatrix A;
    const CRSMatrix B;

    // Vectors to solve
    Eigen::VectorXd u;
    Eigen::VectorXd p;

    // Right-hand side
    Eigen::VectorXd f;
};

#endif // EQUATION_H
