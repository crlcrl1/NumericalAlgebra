#include "conjugate_gradient.h"

#include "util.h"

std::pair<Eigen::VectorXd, int> conjugate_gradient_solver(const int N, const double tol) {
    const auto A = make_matrix(N);
    const auto b = make_vector(N);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(A.rows());
    Eigen::VectorXd r = b - A * x;
    Eigen::VectorXd r_old = r;
    int iter_num = 0;
    const double b_norm = b.norm();
    double error = b_norm * tol + 1;
    Eigen::VectorXd p;
    while (error > tol * b_norm) {
        iter_num += 1;
        if (iter_num == 1) {
            p = r;
        } else {
            const double beta = r.squaredNorm() / r_old.squaredNorm();
            p = r + beta * p;
        }
        const double alpha = r.squaredNorm() / (p.transpose() * A * p)(0);
        x += alpha * p;
        r_old = r;
        r -= alpha * A * p;
        error = r.norm();
    }
    return {x, iter_num};
}
