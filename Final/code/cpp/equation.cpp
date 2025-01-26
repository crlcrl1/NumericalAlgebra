#include "equation.h"

#include <tuple>

#include "util.h"
#include "uzawa.h"
#include "v_cycle.h"

constexpr double PI = std::numbers::pi;


double deriveU(const double x) { return 2 * PI * (1 - std::cos(2 * PI * x)); }

double deriveV(const double y) { return -2 * PI * (1 - std::cos(2 * PI * y)); }

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
getResult(const Eigen::VectorXd &u, const Eigen::VectorXd &p, const int n) {
    auto resultU = Eigen::MatrixXd(n, n - 1);
    auto resultV = Eigen::MatrixXd(n - 1, n);
    auto resultP = Eigen::MatrixXd(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - 1; ++j) {
            resultU(i, j) = u(i * (n - 1) + j);
        }
    }
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n; ++j) {
            resultV(i, j) = u(n * (n - 1) + i * n + j);
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            resultP(i, j) = p(i * n + j);
        }
    }

    return {resultU, resultV, resultP};
}


Equation::Equation(const int n, const std::function<double(double, double)> &f,
                   const std::function<double(double, double)> &g) :
    n(n), h(1.0 / n), A(initA(n)), B(initB(n)), u(2 * n * (n - 1)), p(n * n), f(2 * n * (n - 1)) {
    for (int i = 0; i < 2 * n * (n - 1); ++i) {
        this->u(i) = 0;
    }
    for (int i = 0; i < n * n; ++i) {
        this->p(i) = 0;
    }

    // Initialize f
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - 1; ++j) {
            this->f(i * (n - 1) + j) = f(h * (j + 1), h * (i + 0.5));
        }
    }
    for (int j = 0; j < n - 1; ++j) {
        this->f(j) -= n * deriveU(h * (j + 1));
        this->f((n - 1) * (n - 1) + j) += n * deriveU(h * (j + 1));
    }
    const int offset = n * (n - 1);
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n; ++j) {
            this->f(offset + i * n + j) = g(h * (j + 0.5), h * (i + 1));
        }
    }
    for (int i = 0; i < n - 1; ++i) {
        this->f(offset + i * n) -= n * deriveV(h * (i + 1));
        this->f(offset + i * n + n - 1) += n * deriveV(h * (i + 1));
    }
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, int>
Equation::solveMultiGrid(const int v1, const int v2, const double tol) {
    const Eigen::VectorXd d = Eigen::VectorXd::Zero(n * n);
    const int iterNum = multiGridSolver(A, B, u, p, f, d, n, v1, v2, tol);

    auto [resultU, resultV, resultP] = getResult(u, p, n);
    return {resultU, resultV, resultP, iterNum};
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, int>
Equation::solveUzawaLDL(const double tol) {
    const Eigen::VectorXd d = Eigen::VectorXd::Zero(n * n);
    const int iterNum = uzawaSolverLDL(A, B, u, p, f, tol);

    auto [resultU, resultV, resultP] = getResult(u, p, n);
    return {resultU, resultV, resultP, iterNum};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, int>
Equation::solveUzawaCG(const double tol) {
    const Eigen::VectorXd d = Eigen::VectorXd::Zero(n * n);
    const int iterNum = uzawaSolverCG(A, B, u, p, f, tol);

    auto [resultU, resultV, resultP] = getResult(u, p, n);
    return {resultU, resultV, resultP, iterNum};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, int>
Equation::solveUzawaPCG(const int v1, const int v2, const double tol) {
    const Eigen::VectorXd d = Eigen::VectorXd::Zero(n * n);
    const int iterNum = uzawaSolverPCG(A, B, u, p, f, n, v1, v2, tol);

    auto [resultU, resultV, resultP] = getResult(u, p, n);
    return {resultU, resultV, resultP, iterNum};
}
