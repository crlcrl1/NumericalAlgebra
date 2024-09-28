import numpy as np
from numpy.typing import NDArray

from shared import forward_substitution, backward_substitution

np.random.seed(0)


def square_method(a: NDArray, b: NDArray):
    """
    Use square method to solve the linear system. The input matrix should be a symmetric matrix.
    """
    n = a.shape[0]
    for k in range(n):
        a[k, k] = np.sqrt(a[k, k])
        a[k + 1:, k] = a[k + 1:, k] / a[k, k]
        for j in range(k + 1, n):
            a[j:, j] -= a[j:, k] * a[j, k]
    
    forward_substitution(a, b)
    backward_substitution(a.T, b)
    return b


def optimized_square_method(a: NDArray, b: NDArray):
    """
    Use optimized square method to solve the linear system. The input matrix should be a symmetric matrix.
    """
    n = a.shape[0]
    v = np.zeros(n)
    for k in range(n):
        for i in range(k):
            v[i] = a[k, i] * a[i, i]
        a[k, k] -= a[k, :k] @ v[:k]
        a[k + 1:, k] = (a[k + 1:, k] - a[k + 1:, :k] @ v[:k]) / a[k, k]
    
    temp = a.diagonal().copy()
    for i in range(n):
        a[i, i] = 1
    forward_substitution(a, b)
    b /= temp
    backward_substitution(a.T, b)
    return b


def strip_matrix():
    a = np.eye(100) * 10
    for i in range(99):
        a[i, i + 1] = 1
        a[i + 1, i] = 1
    
    b = np.random.rand(100) * 100
    
    return a, b


def hilbert_matrix(n: int):
    a = np.fromfunction(lambda i, j: 1 / (i + j + 1), (n, n))
    b = a @ np.ones(n)
    
    return a, b


def main():
    n = 40
    a, b = strip_matrix()
    print("Simple strip matrix")
    square_ans = square_method(a.copy(), b.copy())
    print(f"Square method: {np.linalg.norm(a @ square_ans - b)}")
    optimized_ans = optimized_square_method(a.copy(), b.copy())
    print(f"Optimized square method: {np.linalg.norm(a @ optimized_ans - b)}")
    
    print("Hilbert matrix")
    
    a, b = hilbert_matrix(n)
    square_ans = square_method(a.copy(), b.copy())
    print(f"Square method: {np.linalg.norm(a @ square_ans - b)}")
    optimized_ans = optimized_square_method(a.copy(), b.copy())
    print(f"Optimized square method: {np.linalg.norm(a @ optimized_ans - b)}")


if __name__ == "__main__":
    main()
