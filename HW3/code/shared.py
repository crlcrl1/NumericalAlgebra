import numpy as np

from typing import Tuple

from numpy.typing import NDArray


def householder_reflection(x: NDArray) -> Tuple[NDArray, float]:
    """
    Compute the Householder reflection vector for a given vector.
    
    :return: A tuple containing the Householder reflection vector and the scalar beta.
    """
    n = x.shape[0]
    v = np.zeros(n)
    x_norm = np.linalg.norm(x, ord=np.inf)
    x /= x_norm
    sigma = x[1:].dot(x[1:])
    v[1:] = x[1:]
    
    if sigma == 0:
        beta = 0
    else:
        alpha = np.sqrt(x[0] ** 2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - alpha
        else:
            v[0] = -sigma / (x[0] + alpha)
        beta = 2 * v[0] ** 2 / (sigma + v[0] ** 2)
        v /= v[0]
    
    return v, beta


def qr(matrix: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Compute the QR decomposition of a matrix using Householder reflections.
    """
    m, n = matrix.shape
    d = np.zeros(n)
    
    for j in range(n):
        if j < m:
            v, beta = householder_reflection(matrix[j:, j].copy())
            matrix[j:, j:] = (np.eye(m - j) - beta * np.outer(v, v)) @ matrix[j:, j:]
            d[j] = beta
            matrix[j + 1:, j] = v[1:m - j]
    
    return matrix, d


def backward_substitution(a: NDArray, b: NDArray) -> NDArray:
    """
    Use backward substitution to solve the linear system.
    The input matrix should be in upper triangular form.
    """
    n = a.shape[0]
    for i in range(n - 1, 0, -1):
        b[i] /= a[i, i]
        b[:i] = b[:i] - b[i] * a[:i, i]
    b[0] /= a[0, 0]
    return b


def solve_linear_system(matrix: NDArray, b: NDArray) -> NDArray:
    """
    Solve a linear system of equations using the QR decomposition.
    """
    n = matrix.shape[1]
    r, d = qr(matrix)
    for i in range(n - 1):
        v = np.concat([np.array([1.]), r[i + 1:, i]]).reshape((-1, 1))
        b[i:] = (np.eye(n - i) - d[i] * (v @ v.T)) @ b[i:]
    
    return backward_substitution(r, b)


def solve_least_squares(matrix: NDArray, b: NDArray) -> NDArray:
    """
    Solve the least squares problem using the QR decomposition.
    """
    r, d = qr(matrix)
    m, n = matrix.shape
    for i in range(n):
        v = np.concatenate([np.array([1.]), r[i + 1:, i]]).reshape((-1, 1))
        b[i:] = (np.eye(m - i) - d[i] * (v @ v.T)) @ b[i:]
    
    return backward_substitution(r[:n, :], b[:n])
