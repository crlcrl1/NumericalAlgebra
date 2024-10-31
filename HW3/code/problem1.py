from shared import solve_linear_system

import numpy as np
from numpy.typing import NDArray

np.random.seed(0)

def generate_fixed_matrix(n: int) -> NDArray:
    """
    Generate a matrix with given dimension described in the problem.
    """
    res = np.eye(n, dtype=np.float64) * 6
    for i in range(n - 1):
        res[i, i + 1] = 1.0
        res[i + 1, i] = 8.0
    return res


def generate_fixed_vector(n: int) -> NDArray:
    """
    Generate a vector with given dimension described in the problem.
    """
    res = np.array([15.0 for _ in range(n)])
    res[0] = 7.0
    res[n - 1] = 14.0
    return res


def generate_random_matrix(n: int) -> NDArray:
    """
    Generate a random matrix with given dimension.
    """
    a = np.eye(n, dtype=np.float64) * 10
    for i in range(n - 1):
        a[i, i + 1] = 1.
        a[i + 1, i] = 1.
    return a


def generate_random_vector(n: int) -> NDArray:
    """
    Generate a random vector with given dimension.
    """
    return np.random.rand(n)


def fixed_matrix():
    dim_list = [2, 12, 24, 48, 84]
    errors = []
    for dim in dim_list:
        a = generate_fixed_matrix(dim)
        b = generate_fixed_vector(dim)
        x = solve_linear_system(a, b)
        errors.append(np.linalg.norm(x - np.ones(dim), ord=2))
    
    for (dim, error) in zip(dim_list, errors):
        print(f"Dimension: {dim}, Error: {error}")


def hilbert_matrix():
    dim = 40
    a = np.array([[1 / (i + j + 1) for j in range(dim)] for i in range(dim)])
    b = a @ np.ones(dim)
    x = solve_linear_system(a, b)
    print(np.linalg.norm(x - np.ones(dim), ord=2))


def random_matrix():
    a = generate_random_matrix(100)
    b = generate_random_vector(100)
    x = solve_linear_system(a, b)
    print(np.linalg.norm(a @ x - b, ord=2))


if __name__ == "__main__":
    print("Fixed matrix:")
    fixed_matrix()
    print("\nRandom matrix:")
    random_matrix()
    print("\nHilbert matrix:")
    hilbert_matrix()
