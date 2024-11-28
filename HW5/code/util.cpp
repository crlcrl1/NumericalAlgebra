#include "util.h"

#include <iostream>

Eigen::SparseMatrix<double> make_matrix(const int N) {
    const int rows = (N - 1) * (N - 1);
    Eigen::SparseMatrix<double> A(rows, rows);
    A.reserve(Eigen::VectorXi::Constant(rows, 5));
    for (int x = 0; x < N - 1; x++) {
        for (int y = 0; y < N - 1; y++) {
            const int i = x * (N - 1) + y;
            A.insert(i, i) = 4;
            if (x > 0) {
                A.insert(i, i - (N - 1)) = -1;
            }
            if (x < N - 2) {
                A.insert(i, i + (N - 1)) = -1;
            }
            if (y > 0) {
                A.insert(i, i - 1) = -1;
            }
            if (y < N - 2) {
                A.insert(i, i + 1) = -1;
            }
        }
    }
    A.makeCompressed();
    return A;
}

Eigen::VectorXd make_vector(const int N) {
    const int n = (N - 1) * (N - 1);
    const double h = 1.0 / N;
    Eigen::VectorXd b(n);
    for (int i = 1; i < N; i++) {
        for (int j = 1; j < N; j++) {
            const int idx = (i - 1) * (N - 1) + j - 1;
            b[idx] = 2 * PI * PI * std::sin(PI * i * h) * std::sin(PI * j * h) * h * h;
        }
    }
    return b;
}


Eigen::VectorXd real_solution(const int N) {
    Eigen::VectorXd u = Eigen::VectorXd::Zero((N - 1) * (N - 1));
    for (int i = 1; i < N; i++) {
        for (int j = 1; j < N; j++) {
            u[(i - 1) * (N - 1) + j - 1] = std::sin(PI * i / N) * std::sin(PI * j / N);
        }
    }
    return u;
}

Method parse_argument(const int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <method>" << std::endl;
        std::cerr << "method: 0 for LDL, 1 for Gauss-Seidel, 2 for Conjugate Gradient, 3 for "
                     "Multi-grid"
                  << std::endl;
        exit(1);
    }
    return static_cast<Method>(std::stoi(argv[1]));
}
