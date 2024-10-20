import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from shared import solve_least_squares

data = {-1.: 1., -0.75: 0.8125, -0.5: 0.75, 0.: 1., 0.25: 1.3125, 0.5: 1.75, 0.75: 2.3125}

def generate_matrix_vector() -> Tuple[NDArray, NDArray]:
    """
    Generate matrix and vector for least squares problem.
    """
    n = len(data)
    matrix = np.zeros((n, 3))
    vector = np.zeros(n)
    for i, (x, y) in enumerate(data.items()):
        matrix[i] = np.array([1, x, x ** 2])
        vector[i] = y
    
    return matrix, vector


def main():
    matrix, b = generate_matrix_vector()
    print(solve_least_squares(matrix, b))


if __name__ == "__main__":
    main()
