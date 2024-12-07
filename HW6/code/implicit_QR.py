from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class EigResult:
    eigenvalues: NDArray
    eigenvectors: NDArray
    num_iter: int

    def __str__(self):
        return f"Eigenvalues:\n{self.eigenvalues}\nEigenvectors:\n{self.eigenvectors}\nNumber of iterations: {self.num_iter}"


def householder(x: NDArray) -> Tuple[NDArray, float]:
    """
    Compute the Householder reflection vector for a given vector.

    Parameters
    ----------
    x : NDArray
        The vector to compute the Householder reflection vector for.

    Returns
    -------
    Tuple[NDArray, float]
        The Householder reflection vector and beta.
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


def QR_iteration(H: NDArray) -> NDArray:
    """
    Perform a 2 shift QR iteration on a matrix.
    """
    n = H.shape[0]
    P = np.eye(n)
    m = n - 2
    s = H[m, m] + H[n - 1, n - 1]
    t = H[m, m] * H[n - 1, n - 1] - H[m, n - 1] * H[n - 1, m]
    x = H[0, 0] * H[0, 0] + H[0, 1] * H[1, 0] - s * H[0, 0] + t
    y = H[1, 0] * (H[0, 0] + H[1, 1] - s)
    z = H[1, 0] * H[2, 1]

    for k in range(-1, n - 3):
        v, beta = householder(np.array([x, y, z], copy=True))
        v = v.reshape(-1, 1)
        q = max(0, k)
        H[k + 1:k + 4, q:] -= beta * v @ (v.T @ H[k + 1:k + 4, q:])
        r = min(k + 5, n)
        H[:r, k + 1:k + 4] -= beta * H[:r, k + 1:k + 4] @ v @ v.T
        x = H[k + 2, k + 1]
        y = H[k + 3, k + 1]
        if k < n - 4:
            z = H[k + 4, k + 1]
        P[:, k + 1:k + 4] -= beta * P[:, k + 1:k + 4] @ v @ v.T

    v, beta = householder(np.array([x, y], copy=True))
    v = v.reshape(-1, 1)
    H[n - 2:, n - 3:] -= beta * v @ (v.T @ H[n - 2:, n - 3:])
    H[:, n - 2:] -= beta * H[:, n - 2:] @ v @ v.T
    P[:, n - 2:] -= beta * P[:, n - 2:] @ v @ v.T
    return P


def extract_eigenvalues(A: NDArray, tol: float) -> NDArray:
    # split A into numbers or 2x2 matrices
    m_list = []
    n = A.shape[0]
    last = 0
    for i in range(1, n):
        if np.abs(A[i, i - 1]) <= tol:
            m_list.append(A[last:i, last:i])
            last = i
    m_list.append(A[last:, last:])

    e_list = []
    for m in m_list:
        if m.shape[0] == 1:
            e_list.append(m[0, 0])
        else:
            # find eigenvalues of each 2x2 matrix
            a = m[0, 0]
            b = m[0, 1]
            c = m[1, 0]
            d = m[1, 1]
            e_list.extend(np.roots([1., -a - d, a * d - b * c]))

    return np.array(e_list)


def implicit_QR(A: NDArray, tol: float = 1e-10) -> Tuple[NDArray, int]:
    """
    Perform the implicit QR algorithm to find the eigenvalues of a matrix.

    Parameters
    ----------
    A : NDArray
        The matrix to find the eigenvalues of.
    tol : float
        The tolerance to use for the algorithm.

    Returns
    -------
    Tuple[NDArray, int]
        The eigenvalues of the matrix and the number of iterations taken.
    """
    n = A.shape[0]

    # Upper Hessenberg decomposition
    Q = np.eye(n)
    for k in range(n - 2):
        v, beta = householder(A[k + 1:, k].copy())
        v = v.reshape(-1, 1)
        A[k + 1:, k:] -= beta * v @ (v.T @ A[k + 1:, k:])
        A[:, k + 1:] -= beta * A[:, k + 1:] @ v @ v.T
        Q[:, k + 1:] -= beta * Q[:, k + 1:] @ v @ v.T

    num_iter = 0
    while True:
        # convergence check
        for i in range(n - 1):
            if np.abs(A[i, i - 1]) <= tol * (np.abs(A[i, i]) + np.abs(A[i + 1, i + 1])):
                A[i, i - 1] = 0

        # find biggest upper Hessenberg matrix
        flag = False
        m = 0
        for i in range(n - 1, 0, -1):
            if np.abs(A[i, i - 1]) > tol:
                if flag:
                    m -= 1
                    break
                else:
                    flag = True
            else:
                flag = False
            m += 1

        # all eigenvalues found
        if m == n - 1:
            break

        num_iter += 1
        m = n - 1 - m
        l = m
        while l > 0:
            if np.abs(A[l, l - 1]) <= tol:
                break
            l -= 1

        H = A[l:m + 1, l:m + 1]
        P = QR_iteration(H)
        Q[:, l:m + 1] = Q[:, l:m + 1] @ P
        A[:l, l:m + 1] = A[:l, l:m + 1] @ P
        A[l:m + 1, m + 1:] = P.T @ A[l:m + 1, m + 1:]

    return extract_eigenvalues(A, tol), num_iter


def inverse_power_method(A: NDArray, eig: float, tol: float = 1e-10) -> Tuple[NDArray, int]:
    """
    Perform the inverse power method to find the eigenvector of a matrix.

    Parameters
    ----------
    A : NDArray
        The matrix to find the eigenvector of.
    eig : float
        The eigenvalue to find the eigenvector of.
    tol : float
        The tolerance to use for the algorithm.

    Returns
    -------
    Tuple[NDArray, int]
        The eigenvector of the matrix corresponding to the given eigenvalue and the number of iterations taken.
    """
    n = A.shape[0]
    x = np.random.rand(n)
    x /= np.linalg.norm(x)

    num_iter = 0
    while True:
        num_iter += 1
        y = np.linalg.solve(A - eig * np.eye(n), x)
        x = y / np.linalg.norm(y)
        if np.linalg.norm(A @ x - eig * x) < tol:
            break

    return x, num_iter


def eig(A: NDArray, tol: float = 1e-10) -> EigResult:
    """
    Compute the eigenvalues and eigenvectors of a matrix.

    Parameters
    ----------
    A : NDArray
        The matrix to find the eigenvalues and eigenvectors of.
    tol : float
        The tolerance to use for the implicit QR algorithm.

    Returns
    -------
    EigResult
        The eigenvalues and eigenvectors of the matrix and the number of iterations taken.
    """
    eigenvalues, num_iter = implicit_QR(A, tol)
    eigenvectors = np.zeros((A.shape[0], A.shape[0]), dtype=np.complex128)
    for i, e in enumerate(eigenvalues.tolist()):
        eigenvector, _ = inverse_power_method(A, e)
        eigenvectors[:, i] = eigenvector

    return EigResult(eigenvalues, eigenvectors, num_iter)


def generate_matrix(x: float) -> NDArray:
    return np.array([[9.1, 3.0, 2.6, 4.0], [4.2, 5.3, 4.7, 1.6], [3.2, 1.7, 9.4, x], [6.1, 4.9, 3.5, 6.2]])


def main():
    x_list = [0.9, 1.0, 1.1]
    for x in x_list:
        A = generate_matrix(x)
        result = eig(A)
        print(f"x = {x}")
        print(result)
        print()


if __name__ == '__main__':
    main()
