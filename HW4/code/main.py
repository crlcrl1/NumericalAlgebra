from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def gauss_seidel(a: NDArray, b: NDArray, tol: float = 1e-7) -> Tuple[NDArray, int]:
    """
    Use Gauss-Seidel method to solve the linear system.

    Parameters
    ----------
    a : NDArray
        Coefficient matrix.
    b : NDArray
        Right-hand side vector.
    tol : float
        Tolerance for stopping criterion.

    Returns
    -------
    Tuple[NDArray, int]
        Solution vector and the number of iterations.
    """
    x = np.zeros(a.shape[0])
    d = np.diag(np.diag(a))
    l = -np.tril(a, -1)
    u = -np.triu(a, 1)
    # This is just a demo program, so we just use the inverse function provided by numpy.
    inv_d_l = np.linalg.inv(d - l)
    # initialize the iteration matrix and vector
    iter_matrix = inv_d_l @ u
    iter_vector = inv_d_l @ b
    cnt = 0
    b_norm = np.linalg.norm(b)

    while np.linalg.norm(b - a @ x) / b_norm > tol:
        x = iter_matrix @ x + iter_vector
        cnt += 1

    return x, cnt


def optimized_gauss_seidel(a: NDArray, b: NDArray, tol: float = 1e-7) -> Tuple[NDArray, int]:
    """
    Optimize the Gauss-Seidel method for linear system Ax=b, where
    ``A = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=np.float64) + eps * np.eye(3)``

    Parameters
    ----------
    a : NDArray
        Coefficient matrix.
    b : NDArray
        Right-hand side vector.
    tol : float
        Tolerance for stopping criterion.

    Returns
    -------
    Tuple[NDArray, int]
        Solution vector and the number of iterations.
    """
    x = np.zeros(a.shape[0])
    esp = a[0, 0] - 1
    b_norm = np.linalg.norm(b)
    ones = np.ones(a.shape[0])
    d = np.diag(np.diag(a))
    l = -np.tril(a, -1)
    inv_d_l = np.linalg.inv(d - l)
    cnt = 0
    while np.linalg.norm(b - a @ x) / b_norm > tol:
        r = b - a @ x
        # transpose r to k[1, 1, 1]^T and the direction vertical.
        r1 = ones * r.dot(ones) / ones.dot(ones)
        r2 = r - r1
        x += (inv_d_l @ r2 + r1 / esp)
        cnt += 1
    return x, cnt


def main():
    a = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=np.float64)
    b = np.array([1, 10, 23], dtype=np.float64)
    esp_list = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    print("Gauss-Seidel:")
    for esp in esp_list:
        x, cnt = gauss_seidel(a + esp * np.eye(3), b)
        print(f"esp: {esp}, x: {x}, cnt: {cnt}")

    print("Optimized Gauss-Seidel:")
    for esp in esp_list:
        x, cnt = optimized_gauss_seidel(a + esp * np.eye(3), b)
        print(f"esp: {esp}, x: {x}, cnt: {cnt}")


if __name__ == '__main__':
    main()
