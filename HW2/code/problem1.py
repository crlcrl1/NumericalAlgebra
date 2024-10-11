import numpy as np
from numpy.typing import NDArray

from shared import condition_number


def hilbert_matrix(n: int) -> NDArray:
    """
    Generate a Hilbert matrix of size n x n.
    """
    return np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])


def main():
    dims = range(5, 21)
    for n in dims:
        matrix = hilbert_matrix(n)
        print(f"n = {n}, condition number = {condition_number(matrix):.3e}")


if __name__ == "__main__":
    main()
