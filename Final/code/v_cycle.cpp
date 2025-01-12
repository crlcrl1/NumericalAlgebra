#include "v_cycle.h"
#include "util.h"

#include <iostream>

std::unordered_map<int, std::pair<CRSMatrix, CRSMatrix>> restrictOps;
std::unordered_map<int, std::pair<CRSMatrix, CRSMatrix>> liftOps;
std::unordered_map<int, std::pair<CRSMatrix, CRSMatrix>> coeff;


std::pair<CRSMatrix, CRSMatrix> restrictOperator(const int n) {
    assert(n % 2 == 0);

    const int colsU = 2 * n * (n - 1);
    const int colsF = n * n;
    const int nNew = n / 2;
    const int rowsU = 2 * nNew * (nNew - 1);
    const int rowsF = nNew * nNew;
    CRSMatrix restrictU(rowsU, colsU);
    CRSMatrix restrictF(rowsF, colsF);
    restrictU.reserve(Eigen::VectorXi::Constant(rowsU, 6));
    restrictF.reserve(Eigen::VectorXi::Constant(rowsF, 4));

    // restrict u
    for (int i = 0; i < nNew; ++i) {
        for (int j = 0; j < nNew - 1; ++j) {
            const int row = i * (nNew - 1) + j;
            const int base1 = 2 * i * (n - 1) + 2 * j + 1;
            const int base2 = (2 * i + 1) * (n - 1) + 2 * j + 1;
            restrictU.insert(row, base1) = 0.25;
            restrictU.insert(row, base2) = 0.25;
            restrictU.insert(row, base1 - 1) = 0.125;
            restrictU.insert(row, base1 + 1) = 0.125;
            restrictU.insert(row, base2 - 1) = 0.125;
            restrictU.insert(row, base2 + 1) = 0.125;
        }
    }

    const int offsetRestrict = nNew * (nNew - 1);
    const int offsetOriginal = n * (n - 1);
    // restrict v
    for (int i = 0; i < nNew - 1; ++i) {
        for (int j = 0; j < nNew; ++j) {
            const int row = offsetRestrict + i * nNew + j;
            const int base1 = (2 * i + 1) * n + 2 * j + offsetOriginal;
            const int base2 = base1 + 1;
            restrictU.insert(row, base1) = 0.25;
            restrictU.insert(row, base2) = 0.25;
            restrictU.insert(row, base1 - n) = 0.125;
            restrictU.insert(row, base1 + n) = 0.125;
            restrictU.insert(row, base2 - n) = 0.125;
            restrictU.insert(row, base2 + n) = 0.125;
        }
    }
    restrictU.makeCompressed();

    // restrict f
    for (int i = 0; i < nNew; ++i) {
        for (int j = 0; j < nNew; ++j) {
            const int row = i * nNew + j;
            const int base = 2 * i * n + 2 * j;
            restrictF.insert(row, base) = 0.25;
            restrictF.insert(row, base + 1) = 0.25;
            restrictF.insert(row, base + n) = 0.25;
            restrictF.insert(row, base + n + 1) = 0.25;
        }
    }
    restrictF.makeCompressed();

    return {restrictU, restrictF};
}

void ompGaussSeidel(const CRSMatrix &A, Eigen::VectorXd &u, const Eigen::VectorXd &b) {
    const int rows = static_cast<int>(u.size());
    const int block_size = rows / 2;
#pragma omp parallel for
    for (int row = 0; row < 2; ++row) {
        const int start = row * block_size;
        const int end = std::min(start + block_size, rows);
        for (int j = start; j < end; j++) {
            double sum = 0;
            for (CRSMatrix::InnerIterator it(A, j); it; ++it) {
                if (const long col = it.col(); col != j) {
                    sum += it.value() * u[col];
                }
            }
            u[j] = (b[j] - sum) / A.coeff(j, j);
        }
    }
}

void gaussSeidel(const CRSMatrix &A, Eigen::VectorXd &u, const Eigen::VectorXd &b) {
    const int rows = static_cast<int>(u.size());
    for (int j = 0; j < rows; j++) {
        double sum = 0;
        for (CRSMatrix::InnerIterator it(A, j); it; ++it) {
            if (const long col = it.col(); col != j) {
                sum += it.value() * u[col];
            }
        }
        u[j] = (b[j] - sum) / A.coeff(j, j);
    }
}

// Perform one iteration
void DGSIter(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
             const Eigen::VectorXd &f, const Eigen::VectorXd &d, const int n) {
    const Eigen::VectorXd rhs = f - B * p;

    ompGaussSeidel(A, u, rhs);

    const int offset = n * (n - 1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double r = 0;
            int noneZeroCnt = 0;
            // top
            if (i != 0) {
                r -= u[offset + (i - 1) * n + j];
                noneZeroCnt++;
            }
            // bottom
            if (i != n - 1) {
                r += u[offset + i * n + j];
                noneZeroCnt++;
            }
            // left
            if (j != 0) {
                r -= u[i * (n - 1) + j - 1];
                noneZeroCnt++;
            }
            // right
            if (j != n - 1) {
                r += u[i * (n - 1) + j];
                noneZeroCnt++;
            }
            r *= -n;
            r -= d[i * n + j];

            const double delta = r / (noneZeroCnt * n);
            const double temp = r / noneZeroCnt;
            p[i * n + j] += r;
            // top
            if (i != 0) {
                u[offset + (i - 1) * n + j] -= delta;
                p[(i - 1) * n + j] -= temp;
            }
            // bottom
            if (i != n - 1) {
                u[offset + i * n + j] += delta;
                p[(i + 1) * n + j] -= temp;
            }
            // left
            if (j != 0) {
                u[i * (n - 1) + j - 1] -= delta;
                p[i * n + j - 1] -= temp;
            }
            // right
            if (j != n - 1) {
                u[i * (n - 1) + j] += delta;
                p[i * n + j + 1] -= temp;
            }
        }
    }
}

void multiGridIter(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                   const Eigen::VectorXd &f, const Eigen::VectorXd &d, const int n, const int v1,
                   const int v2) {
    if (n == 1) {
        return;
    }

    for (int i = 0; i < v1; ++i) {
        DGSIter(A, B, u, p, f, d, n);
    }

    const Eigen::VectorXd errorU = f - A * u - B * p;
    const Eigen::VectorXd errorP = d - B.transpose() * u;

    if (!restrictOps.contains(n)) {
        const auto [restrictU, restrictP] = restrictOperator(n);
        restrictOps.insert({n, {restrictU, restrictP}});
        liftOps.insert({n, {restrictU.transpose() * 4, restrictP.transpose() * 4}});
    }

    const auto &[restrictU, restrictP] = restrictOps.at(n);
    const auto &[liftU, liftP] = liftOps.at(n);

    const auto restrictErrorU = restrictU * errorU;
    const auto restrictErrorP = restrictP * errorP;

    if (!coeff.contains(n)) {
        const auto restrictA = initA(n / 2);
        const auto restrictB = initB(n / 2);
        coeff.insert({n, {restrictA, restrictB}});
    }
    const auto &[restrictA, restrictB] = coeff.at(n);

    Eigen::VectorXd rectifU = Eigen::VectorXd::Zero(n * (n / 2 - 1));
    Eigen::VectorXd rectifP = Eigen::VectorXd::Zero(n * n / 4);

    multiGridIter(restrictA, restrictB, rectifU, rectifP, restrictErrorU, restrictErrorP, n / 2, v1,
                  v2);

    u += liftU * rectifU;
    p += liftP * rectifP;

    for (int i = 0; i < v2; ++i) {
        DGSIter(A, B, u, p, f, d, n);
    }
}

int multiGridSolver(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                    const Eigen::VectorXd &f, const Eigen::VectorXd &d, const int n, const int v1,
                    const int v2, const double tol) {
    assert(tol < 1);

    const CRSMatrix BTranspose = B.transpose();
    Eigen::VectorXd errorU = f;
    Eigen::VectorXd errorP = d;
    const double initialError = errorU.norm() + errorP.norm();
    double error = initialError;
    int k = 0;
    logError(error, k);

    while (error > initialError * tol) {
        Eigen::VectorXd rectifU = Eigen::VectorXd::Zero(2 * n * (n - 1));
        Eigen::VectorXd rectifP = Eigen::VectorXd::Zero(n * n);
        multiGridIter(A, B, rectifU, rectifP, errorU, errorP, n, v1, v2);
        u += rectifU;
        p += rectifP;
        errorU = f - A * u - B * p;
        errorP = d - BTranspose * u;
        error = errorU.norm() + errorP.norm();
        logError(error, ++k);
    }
    return k;
}
