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
    for (int i = 1; i < n - 1; i++) {
        setBlock(A, i * (n - 1), i * (n - 1), A2);
    }
    for (int i = 0; i < n - 1; i++) {
        setIdentityBlock(A, i * (n - 1), i * (n - 1) + n - 1, n - 1, -n * n);
        setIdentityBlock(A, i * (n - 1) + n - 1, i * (n - 1), n - 1, -n * n);
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

    for (int i = 0; i < n - 1; i++) {
        setBlock(A, i * n, i * n, A3);
        if (i != n - 2) {
            setIdentityBlock(A, i * n, i * n + n, n, -n * n);
            setIdentityBlock(A, i * n + n, i * n, n, -n * n);
        }
    }

    return A;
}

CRSMatrix initA(const int n) {
    auto A = CRSMatrix(2 * n * (n - 1), 2 * n * (n - 1));
    A.reserve(10 * n * (n - 1));

    setBlock(A, 0, 0, initAu(n));
    setBlock(A, n * (n - 1), n * (n - 1), initAv(n));
    A.makeCompressed();
    return A;
}

CRSMatrix initBu(const int n) {
    auto B = CRSMatrix(n * (n - 1), n * n);
    B.reserve(2 * n * (n - 1));

    auto H = CRSMatrix(n - 1, n);
    H.reserve(Eigen::VectorXi::Constant(n, 2));
    for (int i = 0; i < n - 1; i++) {
        H.insert(i, i) = -n;
        H.insert(i, i + 1) = n;
    }

    for (int i = 0; i < n; i++) {
        setBlock(B, i * n - i, i * n, H);
    }

    return B;
}

CRSMatrix initBv(const int n) {
    auto B = CRSMatrix(n * (n - 1), n * n);
    B.reserve(2 * n * (n - 1));

    for (int i = 0; i < n - 1; i++) {
        setIdentityBlock(B, i * n, i * n, n, -n);
        setIdentityBlock(B, i * n, i * n + n, n, n);
    }

    return B;
}

CRSMatrix initB(const int n) {
    auto B = CRSMatrix(2 * n * (n - 1), n * n);
    B.reserve(4 * n * (n - 1));

    setBlock(B, 0, 0, initBu(n));
    setBlock(B, n * (n - 1), 0, initBv(n));
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

void resetLog() { std::cout << "\033[2K"; }
