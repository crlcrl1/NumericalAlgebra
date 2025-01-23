import abc
from typing import Self

import numpy as np


class StrokesEquation(metaclass=abc.ABCMeta):
    def __init__(self, n: int):
        self.n = n

        def f(x, y):
            return -4 * np.pi ** 2 * np.sin(2 * np.pi * y) * (2 * np.cos(2 * np.pi * x) - 1) + x ** 2

        def g(x, y):
            return 4 * np.pi ** 2 * np.sin(2 * np.pi * x) * (2 * np.cos(2 * np.pi * y) - 1)

        def du(x):
            return 2 * np.pi * (1 - np.cos(2 * np.pi * x))

        def dv(y):
            return -2 * np.pi * (1 - np.cos(2 * np.pi * y))

        self.u = np.zeros((n, n - 1), dtype=np.float64)
        self.v = np.zeros((n - 1, n), dtype=np.float64)
        self.p = np.zeros((n, n), dtype=np.float64)
        self.fu = np.fromfunction(lambda i, j: f((j + 1) / n, (i + 0.5) / n), (n, n - 1))
        self.fv = np.fromfunction(lambda i, j: g((j + 0.5) / n, (i + 1) / n), (n - 1, n))

        self.d = np.zeros((n, n), dtype=np.float64)
        self.real_u = np.fromfunction(
            lambda i, j: np.sin(2 * np.pi * (i + 0.5) / n) * (1 - np.cos(2 * np.pi * (j + 1) / n)), (n, n - 1))
        self.real_v = np.fromfunction(
            lambda i, j: -np.sin(2 * np.pi * (j + 0.5) / n) * (1 - np.cos(2 * np.pi * (i + 1) / n)), (n - 1, n))

        boundary_u = np.fromfunction(lambda i: n * du((i + 1) / n), (n - 1,))
        boundary_v = np.fromfunction(lambda j: n * dv((j + 1) / n), (n - 1,))
        self.fu[0, :] -= boundary_u
        self.fu[-1, :] += boundary_u
        self.fv[:, 0] -= boundary_v
        self.fv[:, -1] += boundary_v

    @staticmethod
    def log_error(iteration: int, error: float):
        print(f"[Iter {iteration}] Error: {error}")
        print("\033[1A", end="")

    @abc.abstractmethod
    def solve(self, tol=1e-8, silent=False):
        raise NotImplementedError

    @abc.abstractmethod
    def solve_error(self):
        raise NotImplementedError

    @abc.abstractmethod
    def pde_error(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> Self:
        raise NotImplementedError
