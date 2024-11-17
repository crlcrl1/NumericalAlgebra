from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def read_data(n: int) -> NDArray:
    """
    Read data from file.

    Parameters
    ----------
    n : int
        The size of the matrix.

    Returns
    -------
    NDArray
        The matrix.
    """
    data = np.loadtxt(f"u_{n}.txt")
    n = int(np.sqrt(data.shape[0]))
    return np.pad(data.reshape((n, n)), ((1, 1), (1, 1)), constant_values=1)


def plot_solution(n_list: List):
    """
    Plot the solution of PDE in area [0, 1] x [0, 1].

    Parameters
    ----------
    n_list : List
        A list of n.
    """
    fig = plt.figure()
    for i, n in enumerate(n_list):
        u = read_data(n)
        x, y = np.meshgrid(np.linspace(0, 1, n + 2), np.linspace(0, 1, n + 2))

        # plot the solution
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        # ax3 = plt.axes(projection='3d')
        ax.plot_surface(x, y, u, cmap='rainbow')
        plt.title(f"Solution of PDE with n = {n}")
    plt.show()


def plot_time(n_list: List[int], time_list: List[float]):
    """
    Plot the time cost of different n.

    Parameters
    ----------
    n_list : List[int]
        A list of n.
    time_list : List[float]
        A list of time cost.
    """
    plt.loglog(n_list, time_list, marker='o')
    plt.xlabel('n')
    plt.ylabel('Time cost (ms)')
    plt.title('Time cost of different n')
    plt.show()


def plot_iteration(n_list: List[int], iteration_list: List[int]):
    """
    Plot the iteration number of different n.

    Parameters
    ----------
    n_list : List[int]
        A list of n.
    iteration_list : List[int]
        A list of iteration number.
    """
    plt.loglog(n_list, iteration_list, marker='o')
    plt.xlabel('n')
    plt.ylabel('Iteration number')
    plt.title('Iteration number of different n')
    plt.show()


def main():
    n_list = [20, 40, 80, 160]
    plot_solution(n_list)
    plot_time(n_list, [3.42548, 52.442, 1559.4, 114860])
    plot_iteration(n_list, [637, 2323, 8642, 32448])


if __name__ == '__main__':
    main()
