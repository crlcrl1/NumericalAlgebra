#include "util.h"

#include <format>
#include <iostream>

#include "v_cycle.h"


constexpr double PI = std::numbers::pi;

void setBlock(CRSMatrix &A, const int i, const int j, const CRSMatrix &B) {
    for (int k = 0; k < B.outerSize(); k++) {
        const int row = i + k;
        for (CRSMatrix::InnerIterator it(B, k); it; ++it) {
            A.insert(row, j + it.col()) = it.value();
        }
    }
}

void setIdentityBlock(CRSMatrix &A, const int i, const int j, const int n, const double value) {
    for (int k = 0; k < n; k++) {
        A.insert(i + k, j + k) = value;
    }
}

CRSMatrix initAu(const int n) {
    auto A = CRSMatrix(n * (n - 1), n * (n - 1));
    A.reserve(Eigen::VectorXi::Constant(n * (n - 1), 5));

    // A1 is the matrix for the first and last row of blocks
    auto A1 = CRSMatrix(n - 1, n - 1);
    A1.reserve(Eigen::VectorXi::Constant(n - 1, 3));
    for (int i = 0; i < n - 1; i++) {
        A1.insert(i, i) = 3.0 * n * n;
        if (i < n - 2) {
            A1.insert(i, i + 1) = -n * n;
            A1.insert(i + 1, i) = -n * n;
        }
    }

    // A2 is the matrix for the rest of the rows of blocks
    auto A2 = CRSMatrix(n - 1, n - 1);
    A2.reserve(Eigen::VectorXi::Constant(n - 1, 3));
    for (int i = 0; i < n - 1; i++) {
        A2.insert(i, i) = 4.0 * n * n;
        if (i < n - 2) {
            A2.insert(i, i + 1) = -n * n;
            A2.insert(i + 1, i) = -n * n;
        }
    }

    setBlock(A, 0, 0, A1);
    setBlock(A, (n - 1) * (n - 1), (n - 1) * (n - 1), A1);

#pragma omp parallel for
    for (int i = 1; i < n - 1; i++) {
        setBlock(A, i * (n - 1), i * (n - 1), A2);
    }
#pragma omp parallel for
    for (int i = 0; i < n - 1; i++) {
        setIdentityBlock(A, i * (n - 1), i * (n - 1) + n - 1, n - 1, -n * n);
    }
#pragma omp parallel for
    for (int i = 1; i < n; ++i) {
        setIdentityBlock(A, i * (n - 1), (i - 1) * (n - 1), n - 1, -n * n);
    }
    return A;
}

CRSMatrix initAv(const int n) {
    auto A = CRSMatrix(n * (n - 1), n * (n - 1));
    A.reserve(Eigen::VectorXi::Constant(n * (n - 1), 5));

    auto A3 = CRSMatrix(n, n);
    A3.reserve(Eigen::VectorXi::Constant(n, 3));
    for (int i = 0; i < n; i++) {
        if (i == 0 || i == n - 1) {
            A3.insert(i, i) = 3.0 * n * n;
        } else {
            A3.insert(i, i) = 4.0 * n * n;
        }
        if (i < n - 1) {
            A3.insert(i, i + 1) = -n * n;
            A3.insert(i + 1, i) = -n * n;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < n - 1; i++) {
        setBlock(A, i * n, i * n, A3);
        if (i != n - 2) {
            setIdentityBlock(A, i * n, i * n + n, n, -n * n);
        }
    }
#pragma omp parallel for
    for (int i = 0; i < n - 2; ++i) {
        setIdentityBlock(A, i * n + n, i * n, n, -n * n);
    }

    return A;
}

CRSMatrix initA(const int n) {
    auto A = CRSMatrix(2 * n * (n - 1), 2 * n * (n - 1));
    A.reserve(Eigen::VectorXd::Constant(2 * n * (n - 1), 5));

    const auto aU = initAu(n);
    const auto aV = initAv(n);

#pragma omp parallel
#pragma omp sections
    {
#pragma omp section
        {
            setBlock(A, 0, 0, aU);
        }
#pragma omp section
        {
            setBlock(A, n * (n - 1), n * (n - 1), aV);
        }
    }

    A.makeCompressed();
    return A;
}

CRSMatrix initBu(const int n) {
    auto B = CRSMatrix(n * (n - 1), n * n);
    B.reserve(Eigen::VectorXi::Constant(n * (n - 1), 2));

    auto H = CRSMatrix(n - 1, n);
    H.reserve(Eigen::VectorXi::Constant(n, 2));
    for (int i = 0; i < n - 1; i++) {
        H.insert(i, i) = -n;
        H.insert(i, i + 1) = n;
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        setBlock(B, i * n - i, i * n, H);
    }

    return B;
}

CRSMatrix initBv(const int n) {
    auto B = CRSMatrix(n * (n - 1), n * n);
    B.reserve(Eigen::VectorXi::Constant(2 * n * (n - 1), 2));

#pragma omp parallel for
    for (int i = 0; i < n - 1; i++) {
        setIdentityBlock(B, i * n, i * n, n, -n);
        setIdentityBlock(B, i * n, i * n + n, n, n);
    }

    return B;
}

CRSMatrix initB(const int n) {
    auto B = CRSMatrix(2 * n * (n - 1), n * n);
    B.reserve(Eigen::VectorXd::Constant(2 * n * (n - 1), 2));

    const auto bU = initBu(n);
    const auto bV = initBv(n);

#pragma omp parallel
#pragma omp sections
    {
#pragma omp section
        {
            setBlock(B, 0, 0, bU);
        }
#pragma omp section
        {
            setBlock(B, n * (n - 1), 0, bV);
        }
    }

    B.makeCompressed();
    return B;
}


double realU(const double x, const double y) {
    return std::sin(2 * PI * y) * (1 - std::cos(2 * PI * x));
}

double realV(const double x, const double y) {
    return -std::sin(2 * PI * x) * (1 - std::cos(2 * PI * y));
}

double error(const Eigen::MatrixXd &u, const Eigen::MatrixXd &v, const int n) {
    const double h = 1.0 / n;

    auto realU = Eigen::MatrixXd(n, n - 1);
    auto realV = Eigen::MatrixXd(n - 1, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - 1; ++j) {
            realU(i, j) = ::realU(h * (j + 1), h * (i + 0.5));
        }
    }
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n; ++j) {
            realV(i, j) = ::realV(h * (j + 0.5), h * (i + 1));
        }
    }

    return std::sqrt((realU - u).squaredNorm() + (realV - v).squaredNorm()) * h;
}

void logError(const double error, const int iter) {
    const std::string message = std::format("[Iter {}] Error: {}", iter, error);
    std::cout << message << std::endl;
    // put the cursor to line start
    std::cout << "\033[1A";
}

CRSMatrix restrictU(const int n) {
    assert(n % 2 == 0);

    const int cols = 2 * n * (n - 1);
    const int nNew = n / 2;
    const int rows = 2 * nNew * (nNew - 1);
    CRSMatrix restrictU(rows, cols);
    restrictU.reserve(Eigen::VectorXi::Constant(rows, 6));

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
    return restrictU;
}

CRSMatrix restrictF(const int n) {
    assert(n % 2 == 0);

    const int cols = n * n;
    const int nNew = n / 2;
    const int rows = nNew * nNew;
    CRSMatrix restrictF(rows, cols);
    restrictF.reserve(Eigen::VectorXi::Constant(rows, 4));

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

    return restrictF;
}

void gaussSeidel(const CRSMatrix &A, Eigen::VectorXd &u, const Eigen::VectorXd &b) {
    const int rows = static_cast<int>(u.size());
    const double *values = A.valuePtr();
    const int *outerIndices = A.outerIndexPtr();
    const int *innerIndices = A.innerIndexPtr();

    for (int j = 0; j < rows; j++) {
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

void parallelGaussSeidel(const CRSMatrix &A, Eigen::VectorXd &u, const Eigen::VectorXd &b) {
    const int rows = static_cast<int>(u.size());
    const int block_size = rows / 2;
    const double *values = A.valuePtr();
    const int *outerIndices = A.outerIndexPtr();
    const int *innerIndices = A.innerIndexPtr();

#pragma omp parallel for
    for (int row = 0; row < 2; ++row) {
        const int start = row * block_size;
        const int end = std::min(start + block_size, rows);
        for (int j = start; j < end; j++) {
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
