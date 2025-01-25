import argparse
import time
from typing import Callable

from tqdm import tqdm

from backend import get_equation

n_list = [64, 128, 256, 512, 1024, 2048, 4096]


def timeit(func: Callable, number: int):
    start = time.time()
    for _ in tqdm(range(number), desc="Running benchmark", leave=False):
        func()
    end = time.time()
    return (end - start) / number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", "-b", dest="backend", type=str, help="The backend to use",
                        choices=["jax", "torch"], default="torch")
    namespace = parser.parse_args()
    backend = namespace.backend
    for n in n_list:
        eq = get_equation(backend, n)
        iteration = eq.solve()
        benchmark_time = timeit(lambda: eq.reset().solve(silent=True), number=10)
        print(
            f"n={n}, Solving error: {eq.solve_error():.6e}, PDE error: {eq.pde_error():.6e}, Iteration:{iteration}, Time: {benchmark_time:.6f}s")


if __name__ == '__main__':
    main()
