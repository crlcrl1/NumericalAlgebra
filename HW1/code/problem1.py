from typing import Tuple
import numpy as np
from numpy.typing import *

from shared import forward_substitution, backward_substitution


def gauss_transform(a: NDArray) -> None:
    n = a.shape[0]
    for i in range(n - 1):
        a[i + 1:, i] = a[i + 1:, i] / a[i, i]
        a[i + 1:, i + 1:] = (
                a[i + 1:, i + 1:] - a[i + 1:, i: i + 1] @ a[i: i + 1, i + 1:]
        )


def find_all_max(a: NDArray) -> Tuple[int, int]:
    n = a.shape[0]
    max_i, max_j = 0, 0
    max_val = 0
    for i in range(n):
        for j in range(n):
            if abs(a[i, j]) > max_val:
                max_val = abs(a[i, j])
                max_i, max_j = i, j
    return max_i, max_j


def gauss_transform_with_full_pivot(a: NDArray, u: NDArray, v: NDArray) -> None:
    n = a.shape[0]
    for k in range(n - 1):
        p, q = find_all_max(a[k:, k:])
        a[k, :], a[p + k, :] = a[p + k, :].copy(), a[k, :].copy()
        a[:, k], a[:, q + k] = a[:, q + k].copy(), a[:, k].copy()
        u[k], v[k] = p + k, q + k
        if a[k, k] == 0:
            print("Error")
            return
        a[k + 1:, k] = a[k + 1:, k] / a[k, k]
        a[k + 1:, k + 1:] = (
                a[k + 1:, k + 1:] - a[k + 1:, k: k + 1] @ a[k: k + 1, k + 1:]
        )


def gauss_transform_with_column_pivot(a: NDArray, u: NDArray) -> None:
    n = a.shape[0]
    for k in range(n - 1):
        p = np.argmax(np.abs(a[k:, k]))
        a[k, :], a[p + k, :] = a[p + k, :].copy(), a[k, :].copy()
        u[k] = p + k
        if a[k, k] == 0:
            print("Error")
            return
        a[k + 1:, k] = a[k + 1:, k] / a[k, k]
        a[k + 1:, k + 1:] = (
                a[k + 1:, k + 1:] - a[k + 1:, k: k + 1] @ a[k: k + 1, k + 1:]
        )


def generate_matrix(n: int) -> NDArray:
    res = np.eye(n, dtype=np.float64) * 6
    for i in range(n - 1):
        res[i, i + 1] = 1.0
        res[i + 1, i] = 8.0
    return res


def generate_vector(n: int) -> NDArray:
    res = np.array([15.0 for _ in range(n)])
    res[0] = 7.0
    res[n - 1] = 14.0
    return res


def normal(a: NDArray, b: NDArray) -> NDArray:
    gauss_transform(a)
    temp = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        temp[i] = a[i, i]
    for i in range(a.shape[0]):
        a[i, i] = 1
    forward_substitution(a, b)
    for i in range(a.shape[0]):
        a[i, i] = temp[i]
    backward_substitution(a, b)
    return b


def all_max(a: NDArray, b: NDArray) -> NDArray:
    u = np.zeros(a.shape[0], dtype=np.int32)
    v = np.zeros(a.shape[0], dtype=np.int32)
    gauss_transform_with_full_pivot(a, u, v)
    temp = np.zeros(a.shape[0])
    for i in range(a.shape[0] - 1):
        b[i], b[u[i]] = b[u[i]], b[i]
    for i in range(a.shape[0]):
        temp[i] = a[i, i]
    for i in range(a.shape[0]):
        a[i, i] = 1
    forward_substitution(a, b)
    for i in range(a.shape[0]):
        a[i, i] = temp[i]
    backward_substitution(a, b)
    for i in range(a.shape[0] - 1):
        b[i], b[v[i]] = b[v[i]], b[i]
    return b


def column_max(a: NDArray, b: NDArray) -> NDArray:
    u = np.zeros(a.shape[0], dtype=np.int32)
    gauss_transform_with_column_pivot(a, u)
    temp = np.zeros(a.shape[0])
    for i in range(a.shape[0] - 1):
        b[i], b[u[i]] = b[u[i]], b[i]
    for i in range(a.shape[0]):
        temp[i] = a[i, i]
    for i in range(a.shape[0]):
        a[i, i] = 1
    forward_substitution(a, b)
    for i in range(a.shape[0]):
        a[i, i] = temp[i]
    backward_substitution(a, b)
    return b


def generate_title():
    title = (
        "|Dimension | Normal(2 norm) | Normal(inf norm) | All Max(2 norm) | All Max(inf norm) | "
        "Column Max(2 norm) | Column Max(inf norm)|"
    )
    print(title)
    print("-" * len(title))


def generate_line(n: int) -> None:
    a = generate_matrix(n)
    b = generate_vector(n)
    print(f"|{n:^10}|", end="")
    ans_normal = normal(a.copy(), b.copy())
    ans_all_max = all_max(a.copy(), b.copy())
    ans_column_max = column_max(a.copy(), b.copy())
    print(
        f"{np.linalg.norm(ans_normal - np.ones(n), 2):^16.6e}|"
        f"{np.linalg.norm(ans_normal - np.ones(n), np.inf):^18.6e}|"
        f"{np.linalg.norm(ans_all_max - np.ones(n), 2):^17.6e}|"
        f"{np.linalg.norm(ans_all_max - np.ones(n), np.inf):^19.6e}|"
        f"{np.linalg.norm(ans_column_max - np.ones(n), 2):^20.6e}|"
        f"{np.linalg.norm(ans_column_max - np.ones(n), np.inf):^21.6e}|"
    )


def main() -> None:
    generate_title()
    dims = [2, 12, 24, 48, 84]
    for dim in dims:
        generate_line(dim)


if __name__ == "__main__":
    main()
