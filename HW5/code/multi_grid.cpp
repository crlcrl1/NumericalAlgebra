#include "multi_grid.h"

#include "util.h"


// Buffers for R, P, and A
std::vector<Eigen::SparseMatrix<double>> A_list;
std::vector<Eigen::SparseMatrix<double>> restriction_list;
std::vector<Eigen::SparseMatrix<double>> lifting_list;

/**
 * Perform Gauss-Seidel iteration for @code v@endcode times.
 * It updates the solution @code u@endcode in place.
 */
inline void gauss_seidel_iter(const Eigen::SparseMatrix<double> &A, Eigen::VectorXd &u,
                              const Eigen::VectorXd &b, const int v) {
    const int rows = static_cast<int>(u.size());
    for (int i = 0; i < v; i++) {
        for (int j = 0; j < rows; j++) {
            double sum = 0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
                if (it.row() != j) {
                    sum += it.value() * u[it.row()];
                }
            }
            u[j] = (b[j] - sum) / A.coeff(j, j);
        }
    }
}

Eigen::SparseMatrix<double> restriction_operator(const int N) {
    const int rows = (N - 1) * (N - 1);
    const int N_new = N / 2;
    const int rows_new = (N_new - 1) * (N_new - 1);
    Eigen::SparseMatrix<double> restriction_matrix(rows_new, rows);
    restriction_matrix.reserve(Eigen::VectorXi::Constant(rows, 5));
    for (int x = 0; x < N_new - 1; x++) {
        for (int y = 0; y < N_new - 1; y++) {
            const int idx = (2 * x + 1) * (N - 1) + (2 * y + 1);
            const int idx_new = x * (N_new - 1) + y;
            restriction_matrix.insert(idx_new, idx) = 0.25;
            restriction_matrix.insert(idx_new, idx - (N - 1)) = 0.125;
            restriction_matrix.insert(idx_new, idx + (N - 1)) = 0.125;
            restriction_matrix.insert(idx_new, idx - 1) = 0.125;
            restriction_matrix.insert(idx_new, idx + 1) = 0.125;
            restriction_matrix.insert(idx_new, idx - (N - 1) - 1) = 0.0625;
            restriction_matrix.insert(idx_new, idx - (N - 1) + 1) = 0.0625;
            restriction_matrix.insert(idx_new, idx + (N - 1) - 1) = 0.0625;
            restriction_matrix.insert(idx_new, idx + (N - 1) + 1) = 0.0625;
        }
    }
    restriction_matrix.makeCompressed();
    return restriction_matrix;
}

/**
 * @brief Perform a V-cycle iteration
 */
Eigen::VectorXd V_cycle_iter(const Eigen::VectorXd &rhs, const int N, const int depth, const int v1,
                             const int v2, const Eigen::SparseMatrix<double> &A,
                             const int buffer_id) {
    const int rows = (N - 1) * (N - 1);
    if (depth == 0) {
        return Eigen::VectorXd::Zero(rows);
    }
    Eigen::VectorXd result = Eigen::VectorXd::Zero(rows);

    gauss_seidel_iter(A, result, rhs, v1);

    // calculate the error
    const Eigen::VectorXd error = rhs - A * result;

    // if the restriction matrix is not in the buffer, calculate it
    if (restriction_list.size() <= buffer_id) {
        const auto restriction_matrix = restriction_operator(N);
        restriction_list.push_back(restriction_matrix);
        lifting_list.emplace_back(restriction_matrix.transpose() * 4);
    }

    const Eigen::SparseMatrix<double> restriction_matrix = restriction_list[buffer_id];
    const Eigen::SparseMatrix<double> lifting_matrix = lifting_list[buffer_id];

    // restrict the error on the coarser grid
    const Eigen::VectorXd error_restricted = restriction_matrix * error;

    // if the A matrix is not in the buffer, calculate it
    if (A_list.size() <= buffer_id) {
        A_list.push_back((restriction_matrix * A * lifting_matrix).eval());
        A_list[buffer_id].makeCompressed();
    }
    const Eigen::SparseMatrix<double> A_new = A_list[buffer_id];

    // perform the V-cycle iteration on the coarser grid
    const auto rectification =
            V_cycle_iter(error_restricted, N / 2, depth - 1, v1, v2, A_new, buffer_id + 1);
    result += lifting_matrix * rectification;

    gauss_seidel_iter(A, result, rhs, v2);

    return result;
}


std::pair<Eigen::VectorXd, int> multi_grid_solver(const int N, const int v1, const int v2,
                                                  const double tol) {
    A_list.clear();
    restriction_list.clear();
    lifting_list.clear();
    const auto rhs = make_vector(N);
    const double b_norm = rhs.norm();
    const auto A = make_matrix(N);

    const int depth = static_cast<int>(std::log2(N) - 1);

    Eigen::VectorXd result = V_cycle_iter(rhs, N, depth, v1, v2, A, 0);
    Eigen::VectorXd error = rhs - A * result;
    double error_norm = error.norm();

    int k = 1;
    while (error_norm > tol * b_norm) {
        const Eigen::VectorXd rectification = V_cycle_iter(error, N, depth, v1, v2, A, 0);
        result += rectification;
        error = rhs - A * result;
        error_norm = error.norm();
        k += 1;
    }
    return {result, k};
}
