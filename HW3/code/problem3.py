import numpy as np

from shared import solve_least_squares


def main():
    matrix = np.load("matrix.npz")["arr_0"]
    vector = np.load("vector.npz")["arr_0"]
    x = solve_least_squares(matrix, vector)
    print(x)


if __name__ == "__main__":
    main()
