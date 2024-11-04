#include <cassert>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

/**
 * A matrix that stores only a strip of its element.
 * It is optimized for large sparse matrices.
 *
 * @tparam T Data type of the matrix
 */
template<typename T>
class StripMatrix {
    const int strip_size;
    const int n;
    const int data_col_size;
    T *data; // n * (2 * strip_size + 1) matrix

public:
    StripMatrix(const int n, const int strip_size) :
        strip_size(strip_size), n(n), data_col_size(2 * strip_size + 1) {
        data = new T[n * (2 * strip_size + 1)]{};
    }
    ~StripMatrix() { delete[] data; }

    T &operator()(const int i, const int j) {
        const int temp = i - j + strip_size;
        assert(0 <= temp && temp < data_col_size);
        assert(0 <= i && i < n);
        return data[i * data_col_size + temp];
    }

    const T &operator()(const int i, const int j) const {
        const int temp = i - j + strip_size;
        assert(0 <= temp && temp < data_col_size);
        assert(0 <= i && i < n);
        return data[i * data_col_size + temp];
    }

    [[nodiscard]] int size() const { return n; }
    [[nodiscard]] int stripSize() const { return strip_size; }
};

/**
 * Perform Gauss-Seidel method to solve Ax = b.
 *
 * @param A Coefficient matrix
 * @param b Right-hand side vector
 * @param eps Error tolerance
 * @param max_iter Maximum number of iterations, -1 for no limit
 * @return A tuple of the solution vector and the number of iterations
 */
std::tuple<std::vector<double>, int> gauss_seidel(const StripMatrix<double> &A,
                                                  const std::vector<double> &b, const double eps,
                                                  const int max_iter = -1) {
    assert(A.size() == b.size());
    const int n = A.size();
    const int strip_size = A.stripSize();
    std::vector<double> x(n, 0);
    std::vector<double> g(n, 0);
    for (int i = 0; i < n; ++i) {
        g[i] = b[i] / A(i, i);
    }
    double error = eps + 1;
    int iter = 0;
    while (error > eps && (max_iter <= -1 || iter < max_iter)) {
        error = 0;
        for (int i = 0; i < n; ++i) {
            double sum = 0;
            const int start = std::max(0, i - strip_size);
            const int end = std::min(n, i + strip_size + 1);
            for (int j = start; j < i; ++j) {
                sum -= A(i, j) * x[j];
            }
            for (int j = i + 1; j < end; ++j) {
                sum -= A(i, j) * x[j];
            }
            sum /= A(i, i);
            const double x_old = x[i];
            const double x_new = g[i] + sum;
            x[i] = x_new;
            const double diff = x_new - x_old;
            error += diff * diff;
        }
        error = std::sqrt(error);
        ++iter;
    }
    return {x, iter};
}

/**
 * Solve the Poisson equation -Δu + g(x, y)u = f(x, y) in the unit square [0, 1] × [0, 1].
 * The boundary conditions are u = 1 on the boundary.
 *
 * @param f Right-hand side function
 * @param g Coefficient function
 * @param N Number of grid points in each direction
 * @param eps Error tolerance
 * @param max_iter Maximum number of iterations for gauss seidel method, -1 for no limit
 * @return A tuple of the solution vector and the number of iterations
 */
std::tuple<std::vector<double>, int> solve_pde(const std::function<double(double, double)> &f,
                                               const std::function<double(double, double)> &g,
                                               const int N, const double eps,
                                               const int max_iter = -1) {
    const double h = 1.0 / (N + 1);
    StripMatrix<double> A(N * N, N);
    std::vector<double> b(N * N, 0);
    // initialize the matrix A and vector b
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            b[i * N + j] = f((i + 1) * h, (j + 1) * h) * h * h;
            A(i * N + j, i * N + j) = 4 + g((i + 1) * h, (j + 1) * h) * h * h;
            if (i > 0) {
                A(i * N + j, (i - 1) * N + j) = -1;
            } else {
                b[i * N + j] += 1;
            }
            if (i < N - 1) {
                A(i * N + j, (i + 1) * N + j) = -1;
            } else {
                b[i * N + j] += 1;
            }
            if (j > 0) {
                A(i * N + j, i * N + j - 1) = -1;
            } else {
                b[i * N + j] += 1;
            }
            if (j < N - 1) {
                A(i * N + j, i * N + j + 1) = -1;
            } else {
                b[i * N + j] += 1;
            }
        }
    }

    return gauss_seidel(A, b, eps, max_iter);
}

double f(const double x, const double y) { return x + y; }

double g(const double x, const double y) { return std::exp(x * y); }

/**
 * Save a vector to a numpy array file.
 *
 * @param v the vector to be saved
 * @param file_path the file path
 */
void dump_numpy_array(const std::vector<double> &v, const std::string &file_path) {
    std::ofstream file(file_path);
    file << "[";
    for (int i = 0; i < v.size(); ++i) {
        file << std::format("{:.9f}", v[i]);
        if (i != v.size() - 1) {
            file << ", ";
        }
    }
    file << "]";
    file.close();
}

int main() {
    constexpr int n_list[] = {20, 40, 80, 160};
    for (const int n: n_list) {
        constexpr double eps = 1e-7;
        const auto start = std::chrono::high_resolution_clock::now();
        const auto [u, iter] = solve_pde(f, g, n, eps, 10000);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto diff = end - start;
        dump_numpy_array(u, std::format("u_{}.txt", n));
        std::cout << std::format("N = {}, CPU time: {}, Number of iterations: {}\n", n,
                                 std::chrono::duration<double, std::milli>(diff), iter);
    }
}
