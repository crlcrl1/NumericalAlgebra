import numpy as np
from numpy.typing import NDArray


def power_method(A: NDArray, num_iter: int) -> float:
    """
    Power method for finding the dominant eigenvalue and eigenvector of a matrix.

    Parameters
    ----------
    A : NDArray
        The matrix to find the dominant eigenvalue and eigenvector of.
    num_iter : int
        The maximum number of iterations to perform.

    Returns
    -------
    float
        The dominant eigenvalue of the matrix.
    """
    u = 0
    x = np.ones(A.shape[0])
    for _ in range(num_iter):
        x = A @ x
        u = x[np.argmax(np.abs(x))]
        x /= u
    return u


def f(x):
    return np.poly1d([1., 101., 208.01, 10891.01, 9802.08, 79108.9, -99902., 790., -1000])(x)


def main():
    A = np.zeros((8, 8), dtype=float)
    l = [-1000., -790., -99902., -79108.9, 9802.08, -10891.01, 208.01, -101.]
    for i, v in enumerate(l):
        A[i, -1] = v
    for i in range(7):
        A[i + 1, i] = -1.
    num_iter = 100
    u = power_method(A, num_iter)
    print(f"The maximum root of the polynomial is {u}")
    print(f(u))


if __name__ == '__main__':
    main()
