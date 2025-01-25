#include "v_cycle.h"
#include "util.h"


void updatePressure(Eigen::VectorXd &u, Eigen::VectorXd &p, const Eigen::VectorXd &d, const int i,
                    const int j, const int n) {
    const int offset = n * (n - 1);
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

void parallelDGSIter(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                     const Eigen::VectorXd &f, const Eigen::VectorXd &d, const int n) {
    const Eigen::VectorXd rhs = f - B * p;

    parallelGaussSeidel(A, u, rhs);

#pragma omp parallel for collapse(2)
    for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 2; ++l) {
            // these units will update in parallel
            const int start1 = k * (n / 2 + 1);
            const int end1 = k * (n / 2 + 1) + n / 2 - 1;
            const int start2 = l * (n / 2 + 1);
            const int end2 = l * (n / 2 + 1) + n / 2 - 1;
            for (int i = start1; i < end1; ++i) {
                for (int j = start2; j < end2; ++j) {
                    updatePressure(u, p, d, i, j, n);
                }
            }
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            const int start = j * (n / 2 + 1);
            const int end = j * (n / 2 + 1) + n / 2 - 1;
            if (i == 0) {
                for (int k = start; k < end; ++k) {
                    updatePressure(u, p, d, n / 2 - 1, k, n);
                    updatePressure(u, p, d, n / 2, k, n);
                }
            } else {
                for (int k = start; k < end; ++k) {
                    updatePressure(u, p, d, k, n / 2 - 1, n);
                    updatePressure(u, p, d, k, n / 2, n);
                }
            }
        }
    }

    updatePressure(u, p, d, n / 2 - 1, n / 2 - 1, n);
    updatePressure(u, p, d, n / 2 - 1, n / 2, n);
    updatePressure(u, p, d, n / 2, n / 2 - 1, n);
    updatePressure(u, p, d, n / 2, n / 2, n);
}

// Perform one iteration
void DGSIter(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
             const Eigen::VectorXd &f, const Eigen::VectorXd &d, const int n) {
    const Eigen::VectorXd rhs = f - B * p;
    gaussSeidel(A, u, rhs);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            updatePressure(u, p, d, i, j, n);
        }
    }
}

void multiGridIter(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                   const Eigen::VectorXd &f, const Eigen::VectorXd &d, const int n, const int v1,
                   const int v2) {
    static std::unordered_map<int, std::pair<CRSMatrix, CRSMatrix>> restrictOps;
    static std::unordered_map<int, std::pair<CRSMatrix, CRSMatrix>> prolongOps;
    static std::unordered_map<int, std::pair<CRSMatrix, CRSMatrix>> coeffs;

    if (n == 1) {
        return;
    }

    for (int i = 0; i < v1; ++i) {
        if (n > 4) {
            parallelDGSIter(A, B, u, p, f, d, n);
        } else {
            DGSIter(A, B, u, p, f, d, n);
        }
    }

    const Eigen::VectorXd errorU = f - A * u - B * p;
    const Eigen::VectorXd errorP = d - B.transpose() * u;

    if (!restrictOps.contains(n)) {
        const auto restrictU = ::restrictU(n);
        const auto restrictP = restrictF(n);
        restrictOps.insert({n, {restrictU, restrictP}});
        prolongOps.insert({n, {restrictU.transpose() * 4, restrictP.transpose() * 4}});
    }

    const auto &[restrictU, restrictP] = restrictOps.at(n);
    const auto &[liftU, liftP] = prolongOps.at(n);

    const auto restrictErrorU = restrictU * errorU;
    const auto restrictErrorP = restrictP * errorP;

    if (!coeffs.contains(n)) {
        const auto restrictA = initA(n / 2);
        const auto restrictB = initB(n / 2);
        coeffs.insert({n, {restrictA, restrictB}});
    }
    const auto &[restrictA, restrictB] = coeffs.at(n);

    Eigen::VectorXd rectifU = Eigen::VectorXd::Zero(n * (n / 2 - 1));
    Eigen::VectorXd rectifP = Eigen::VectorXd::Zero(n * n / 4);

    multiGridIter(restrictA, restrictB, rectifU, rectifP, restrictErrorU, restrictErrorP, n / 2, v1,
                  v2);

    u += liftU * rectifU;
    p += liftP * rectifP;

    for (int i = 0; i < v2; ++i) {
        if (n > 4) {
            parallelDGSIter(A, B, u, p, f, d, n);
        } else {
            DGSIter(A, B, u, p, f, d, n);
        }
    }
}

int multiGridSolver(const CRSMatrix &A, const CRSMatrix &B, Eigen::VectorXd &u, Eigen::VectorXd &p,
                    const Eigen::VectorXd &f, const Eigen::VectorXd &d, const int n, const int v1,
                    const int v2, const double tol) {
    assert(tol < 1);

    const CRSMatrix BTrans = B.transpose();
    Eigen::VectorXd errorU = f;
    Eigen::VectorXd errorP = d;
    const double initialError = std::sqrt(errorU.squaredNorm() + errorP.squaredNorm());
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
        errorP = d - BTrans * u;
        error = std::sqrt(errorU.squaredNorm() + errorP.squaredNorm());
        logError(error, ++k);
    }
    return k;
}
