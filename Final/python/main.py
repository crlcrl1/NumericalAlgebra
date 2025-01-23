import time
from typing import Callable
from tqdm import tqdm

from backend import get_equation


def timeit(func: Callable, number: int):
    start = time.time()
    for _ in tqdm(range(number), desc="Running benchmark", leave=False):
        func()
    end = time.time()
    return (end - start) / number


def main():
    n = 2048
    # eq = get_equation("jax", n)
    eq = get_equation("torch", n)
    eq.solve()
    benchmark_time = timeit(lambda: eq.reset().solve(silent=True), number=10)
    print(f"Solving error: {eq.solve_error():.6e}, PDE error: {eq.pde_error():.6e}, Time: {benchmark_time:.6f}s")


if __name__ == '__main__':
    main()
