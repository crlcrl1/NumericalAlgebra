#include "uzawa.h"

#include <format>
#include <iostream>

#include "util.h"

void lowerGaussSeidel(const CRSMatrix &A, Eigen::VectorXd &u, const Eigen::VectorXd &b) {
    const int rows = static_cast<int>(u.size());
    const double *values = A.valuePtr();
    const int *outerIndices = A.outerIndexPtr();
    const int *innerIndices = A.innerIndexPtr();

    for (int j = rows - 1; j >= 0; j--) {
        double sum = 0;
        const int rowStart = outerIndices[j];
        const int rowEnd = outerIndices[j + 1];

        for (int i = rowStart; i < rowEnd; i++) {
            if (const int col = innerIndices[i]; col != j) {
                sum += values[i] * u[col];
            }
        }

        u[j] = (b[j] - sum) / A.coeff(j, j);
    }
}

void parallelLowerGaussSeidel(const CRSMatrix &A, Eigen::VectorXd &u, const Eigen::VectorXd &b) {
    const int rows = static_cast<int>(u.size());
    const double *values = A.valuePtr();
    const int *outerIndices = A.outerIndexPtr();
    const int *innerIndices = A.innerIndexPtr();

    const int block = rows / 2;
#pragma omp parallel for
    for (int k = 0; k < 1; ++k) {
        const int start = k * block;
        const int end = (k + 1) * block - 1;
        for (int j = end; j >= start; j--) {
            double sum = 0;
            const int rowStart = outerIndices[j];
            const int rowEnd = outerIndices[j + 1];

            for (int i = rowStart; i < rowEnd; i++) {
                if (const int col = innerIndices[i]; col != j) {
                    sum += values[i] * u[col];
                }
            }
            u[j] = (b[j] - sum) / A.coeff(j, j);
        }
    }
}

void iterOnce(const CRSMatrix &A, Eigen::VectorXd &x, const Eigen::VectorXd &b, const int v1,
              const int v2, const int n) {
    static std::unordered_map<int, CRSMatrix> restrictOps;
    static std::unordered_map<int, CRSMatrix> prolongOps;
    static std::unordered_map<int, CRSMatrix> coeffs;

    if (n == 1) {
        return;
    }

    assert(n % 2 == 0);

    for (int i = 0; i < v1; ++i) {
        if (n > 8) {
            parallelGaussSeidel(A, x, b);
            parallelLowerGaussSeidel(A, x, b);
        } else {
            gaussSeidel(A, x, b);
            lowerGaussSeidel(A, x, b);
        }
    }

    if (!restrictOps.contains(n)) {
        const auto restrictU = ::restrictU(n);
        restrictOps.insert({n, restrictU});
        prolongOps.insert({n, restrictU.transpose() * 4});
    }

    const auto &restrict = restrictOps.at(n);
    const auto &lift = prolongOps.at(n);

    const Eigen::VectorXd error = restrict * (b - A * x);
    Eigen::VectorXd rectif = Eigen::VectorXd::Zero(error.size());

    if (!coeffs.contains(n)) {
        const auto restrictA = initA(n / 2);
        coeffs.insert({n, restrictA});
    }

    const auto &restrictA = coeffs.at(n);
    iterOnce(restrictA, rectif, error, v1, v2, n / 2);

    x += lift * rectif;

    for (int i = 0; i < v2; ++i) {
        if (n > 8) {
            parallelGaussSeidel(A, x, b);
            parallelLowerGaussSeidel(A, x, b);
        } else {
            gaussSeidel(A, x, b);
            lowerGaussSeidel(A, x, b);
        }
    }
}


Eigen::VectorXd multiGrid(const CRSMatrix &A, const Eigen::VectorXd &b, const int maxIter,
                          const int v1, const int v2, const int n) {

    Eigen::VectorXd res = Eigen::VectorXd::Zero(b.size());
    Eigen::VectorXd error = b;
    int k = 0;
    while (k < maxIter && error.norm() > 1e-8) {
        iterOnce(A, res, error, v1, v2, n);
        error = b - A * res;
        k++;
    }
    return res;
}

Eigen::VectorXd PCG(const CRSMatrix &A, const Eigen::VectorXd &b, const int innerMaxIter,
                    const int n, const int v1, const int v2, const double tol,
                    const bool verbose = false) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    Eigen::VectorXd r = b - A * x;
    Eigen::VectorXd p;
    double rho = 0;
    const double bNorm = b.norm();
    int k = 0;
    while (r.norm() > tol * bNorm) {
        const auto z = multiGrid(A, r, innerMaxIter, v1, v2, n);
        if (k > 0) {
            const double rhoPrev = rho;
            rho = r.dot(z);
            const double bata = rho / rhoPrev;
            p = z + bata * p;
        } else {
            rho = r.dot(z);
            p = z;
        }
        const Eigen::VectorXd w = A * p;
        const double alpha = rho / p.dot(w);
        x += alpha * p;
        r -= alpha * w;
        k++;
    }
    if (verbose) {
        std::cout << std::format("PCG Iteration: {}, error: {:.8e}", k, r.norm()) << std::endl;
    }
    return x;
}


int uzawaSolverLDL(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                   const Eigen::VectorXd &f, const double tol) {
    assert(tol < 1);

    const CRSMatrix BTrans = B.transpose();
    Eigen::VectorXd errorU = f;
    Eigen::VectorXd errorP = BTrans * u;
    Eigen::VectorXd rhs = f - B * p;
    const double initialError = std::sqrt(errorU.squaredNorm() + errorP.squaredNorm());
    double error = initialError;
    int k = 0;
    logError(error, k);

    while (error > initialError * tol) {
        Eigen::SimplicialLDLT<CRSMatrix> solver;
        solver.compute(A);
        u = solver.solve(rhs);
        errorP = BTrans * u;
        p += errorP;

        rhs = f - B * p;
        errorU = rhs - A * u;
        error = std::sqrt(errorU.squaredNorm() + errorP.squaredNorm());
        logError(error, ++k);
    }
    return k;
}

int uzawaSolverCG(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                  const Eigen::VectorXd &f, const double tol) {
    assert(tol < 1);

    const CRSMatrix BTrans = B.transpose();
    Eigen::VectorXd errorU = f;
    Eigen::VectorXd errorP = BTrans * u;
    Eigen::VectorXd rhs = f - B * p;
    const double initialError = std::sqrt(errorU.squaredNorm() + errorP.squaredNorm());
    double error = initialError;
    int k = 0;
    logError(error, k);

    while (error > initialError * tol) {
        Eigen::ConjugateGradient<CRSMatrix, Eigen::Lower | Eigen::Upper> solver;
        solver.setTolerance(tol);
        solver.compute(A);
        u = solver.solve(rhs);
        errorP = BTrans * u;
        p += errorP;

        rhs = f - B * p;
        errorU = rhs - A * u;
        error = std::sqrt(errorU.squaredNorm() + errorP.squaredNorm());
        logError(error, ++k);
    }
    return k;
}

int uzawaSolverPCG(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                   const Eigen::VectorXd &f, const int n, const int v1, const int v2,
                   const double tol) {
    assert(tol < 1);

    const CRSMatrix BTrans = B.transpose();
    Eigen::VectorXd errorU = f;
    Eigen::VectorXd errorP = BTrans * u;
    Eigen::VectorXd rhs = f - B * p;
    const double initialError = std::sqrt(errorU.squaredNorm() + errorP.squaredNorm());
    double error = initialError;
    int k = 0;
    logError(error, k);

    while (error > initialError * tol) {
        u = PCG(A, rhs, 1, n, v1, v2, tol);
        errorP = BTrans * u;
        p += errorP;

        rhs = f - B * p;
        errorU = rhs - A * u;
        error = std::sqrt(errorU.squaredNorm() + errorP.squaredNorm());
        logError(error, ++k);
    }
    return k;
}
