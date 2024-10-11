import numpy as np
import scipy
from numpy.typing import NDArray

from shared import condition_number, solve

def generate_matrix(n: int) -> NDArray:
    """
    Generate a Hilbert matrix of size n x n.
    """
    a = np.eye(n)
    for i in range(1, n):
        for j in range(i - 1):
            a[i, j] = -1.0
    
    for i in range(n):
        a[i, n - 1] = 1.0
    
    return a


def error_estimate(a: NDArray, b: NDArray) -> float:
    """
    Estimate the error of the solution to the system of linear equations Ax = b.
    """
    lu = scipy.linalg.lu(a)
    x = solve(lu, b)
    r = np.linalg.norm(b - a @ x, ord=np.inf)
    cond = condition_number(a, lu=lu)
    return cond * r / np.linalg.norm(b, ord=np.inf)


def main():
    dims = range(5, 31)
    for n in dims:
        matrix = generate_matrix(n)
        b = np.random.rand(n)
        print(f"n = {n}, error estimate = {error_estimate(matrix, b):.3e}")


if __name__ == "__main__":
    main()
