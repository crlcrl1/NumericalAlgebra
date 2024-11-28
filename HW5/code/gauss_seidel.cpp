#include "gauss_seidel.h"
#include "util.h"


std::pair<Eigen::VectorXd, int> gauss_seidel_solver(const int N, const double tol) {
    const auto A = make_matrix(N);
    const auto b = make_vector(N);

    const int rows = (N - 1) * (N - 1);

    const double b_norm = b.norm();
    double error_norm = tol * b_norm + 1;
    Eigen::VectorXd u = Eigen::VectorXd::Zero(rows);

    int iter_count = 0;
    while (error_norm > tol * b_norm) {
        for (int j = 0; j < rows; j++) {
            double sum = 0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
                if (it.row() != j) {
                    sum += it.value() * u[it.row()];
                }
            }
            u[j] = (b[j] - sum) / A.coeff(j, j);
        }
        Eigen::VectorXd r = b - A * u;
        error_norm = r.norm();
        iter_count++;
    }

    return {u, iter_count};
}
