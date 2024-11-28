#include "ldl.h"

#include <iostream>

#include "util.h"

/**
 * A class to view the memory-optimized matrix as a normal matrix
 */
class MatrixView {
    std::vector<std::vector<double>> *original;
    size_t n;

public:
    inline explicit MatrixView(std::vector<std::vector<double>> &original);

    inline double &operator()(int i, int j) const;
};

std::vector<std::vector<double>> generate_matrix(const int n) {
    const int temp = (n - 1) * (n - 1);
    std::vector res(temp, std::vector<double>(2 * n - 1, 0));
    for (int i = 0; i < temp; ++i) {
        res[i][n - 1] = 4;
        res[i][n] = -1;
        res[i][n - 2] = -1;
        res[i][0] = -1;
        res[i][2 * n - 2] = -1;
    }
    for (int i = 0; i < n - 1; ++i) {
        res[(n - 1) * i][n] = 0;
        res[(n - 1) * i + n - 2][n - 2] = 0;
    }
    return res;
}

double original_function(const double x, const double y) {
    return 2 * PI * PI * sin(PI * x) * sin(PI * y);
}

std::vector<double> generate_vector(const int n) {
    std::vector<double> res((n - 1) * (n - 1), 0);
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - 1; ++j) {
            res[i * (n - 1) + j] = original_function(static_cast<double>(i + 1) / n,
                                                     static_cast<double>(j + 1) / n) /
                                   (n * n);
        }
    }
    return res;
}

/**
 * Banded Gauss elimination
 */
std::vector<double> strip_gauss_elimination(std::vector<std::vector<double>> &a,
                                            const std::vector<double> &b) {
    std::vector<double> res = b;
    const int n = static_cast<int>(sqrt(b.size())) + 1;
    const MatrixView view(a);
    const int temp1 = (n - 1) * (n - 1);
    for (int i = 0; i < temp1; ++i) {
        const int temp2 = std::min(n + i, temp1);
        for (int j = i + 1; j < temp2; ++j) {
            const double ratio = view(j, i) / view(i, i);
            for (int k = i; k < temp2; ++k) {
                view(j, k) -= ratio * view(i, k);
            }
            res[j] -= ratio * res[i];
        }
    }
    for (int i = (n - 1) * (n - 1) - 1; i >= 0; --i) {
        const int temp = std::max(-1, i - n);
        for (int j = i - 1; j > temp; --j) {
            const double ratio = view(j, i) / view(i, i);
            res[j] -= ratio * res[i];
            view(j, i) = 0;
        }
        res[i] /= view(i, i);
    }
    return res;
}

/**
 * LDL^T decomposition
 */
void ldl_decomposition(std::vector<std::vector<double>> &a, std::vector<double> &v) {
    const MatrixView view(a);
    const int d = static_cast<int>(v.size());
    const int n = static_cast<int>(sqrt(d)) + 1;
    for (int j = 0; j < d; ++j) {
        const int temp2 = std::max(j - n + 1, 0);
        for (int i = temp2; i <= j - 1; ++i) {
            v[i] = view(j, i) * view(i, i);
        }
        double temp = 0;
        for (int i = temp2; i <= j - 1; ++i) {
            temp += view(j, i) * v[i];
        }
        view(j, j) -= temp;
        const int temp1 = std::min(d, j + n);
        for (int i = j + 1; i < temp1; ++i) {
            temp = 0;
            const int start = std::max(i - n + 1, temp2);
            const int end = std::min(j - 1, i + n - 1);
            for (int k = start; k <= end; ++k) {
                temp += view(i, k) * v[k];
            }
            view(i, j) -= temp;
            view(i, j) /= view(j, j);
        }
    }

    for (int i = 0; i < d; ++i) {
        v[i] = view(i, i);
        view(i, i) = 1;
    }
    for (int i = 0; i < d; ++i) {
        const int temp = std::min(d, i + n);
        for (int j = i + 1; j < temp; ++j) {
            view(i, j) = view(j, i);
        }
    }
}

void forward_elimination(std::vector<std::vector<double>> &a, std::vector<double> &b) {
    const int d = static_cast<int>(b.size());
    const int n = static_cast<int>(sqrt(d)) + 1;
    const MatrixView view(a);
    for (int i = 0; i < d - 1; ++i) {
        b[i] /= view(i, i);
        const int temp = std::min(d, i + n);
        for (int j = i + 1; j < temp; ++j) {
            b[j] -= view(j, i) * b[i];
        }
    }
    b[d - 1] /= view(d - 1, d - 1);
}

void backward_elimination(std::vector<std::vector<double>> &a, std::vector<double> &b) {
    const int d = static_cast<int>(b.size());
    const int n = static_cast<int>(sqrt(d)) + 1;
    const MatrixView view(a);
    for (int i = d - 1; i > 0; --i) {
        b[i] /= view(i, i);
        const int temp = std::max(-1, i - n);
        for (int j = i - 1; j > temp; --j) {
            b[j] -= view(j, i) * b[i];
        }
    }
    b[0] /= view(0, 0);
}

inline MatrixView::MatrixView(std::vector<std::vector<double>> &original) : original(&original) {
    n = static_cast<size_t>(sqrt(static_cast<double>(original.size()))) + 1;
}

inline double &MatrixView::operator()(const int i, const int j) const {
    return (*original)[i][n - 1 + i - j];
}
Eigen::VectorXd ldl_solver(int N) {
    auto A = generate_matrix(N);
    auto b = generate_vector(N);
    const size_t d = b.size();
    std::vector<double> v(d);
    ldl_decomposition(A, v);
    forward_elimination(A, b);
    for (int i = 0; i < d; ++i) {
        b[i] /= v[i];
    }
    backward_elimination(A, b);
    return Eigen::Map<Eigen::VectorXd>(b.data(), b.size());
}
