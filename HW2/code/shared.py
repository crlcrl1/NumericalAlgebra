"""
This file contains the shared functions for  computing the condition number of a matrix.
"""

import numpy as np
import scipy
from numpy.typing import NDArray
from typing import Tuple


def forward_substitution(a: NDArray, b: NDArray) -> NDArray:
    """
    Use backward substitution to solve the linear system.
    The input matrix should be in lower triangular form.
    """
    n = a.shape[0]
    for i in range(n - 1):
        b[i] /= a[i, i]
        b[i + 1:] = b[i + 1:] - b[i] * a[i + 1:, i]
    b[n - 1] /= a[n - 1, n - 1]
    return b


def backward_substitution(a: NDArray, b: NDArray) -> NDArray:
    """
    Use backward substitution to solve the linear system.
    The input matrix should be in upper triangular form.
    """
    n = a.shape[0]
    for i in range(n - 1, 0, -1):
        b[i] /= a[i, i]
        b[:i] = b[:i] - b[i] * a[:i, i]
    b[0] /= a[0, 0]
    return b

def solve(lu: Tuple, b: NDArray) -> NDArray:
    """
    Solve the system of linear equations Ax = b, where A = PLU.
    """
    b = b.copy()
    p, l, u = lu
    # Solve Ly = Pb
    y = forward_substitution(l, p @ b)
    # Solve Ux = y
    x = backward_substitution(u, y)
    return x


def solve_T(lu: Tuple, b: NDArray) -> NDArray:
    """
    Solve the system of linear equations A^T x = b, where A = PLU.
    """
    b = b.copy()
    p, l, u = lu
    # Solve U^T y = b
    y = forward_substitution(u.T, b)
    # Solve L^T P^T x = y
    x = backward_substitution(l.T, y)
    return p @ x


def condition_number(matrix: NDArray, lu: Tuple = None) -> float:
    """
    Calculate the inf-norm condition number of a matrix.
    """
    a_norm = np.linalg.norm(matrix, ord=np.inf)
    n = matrix.shape[0]
    
    # We can also use the LU decomposition implemented in homework 1
    if lu is None:
        lu = scipy.linalg.lu(matrix)
    
    x = np.array([1 / n] * n)
    # Store the tried x to avoid infinite loop
    tried_x = [x]
    while True:
        w = solve_T(lu, x)
        v = np.sign(w)
        z = solve(lu, v)
        z_norm = np.linalg.norm(z, ord=np.inf)
        
        # Check if the stopping criterion is met
        if z_norm <= z.T @ x:
            return a_norm * np.linalg.norm(w, ord=1)
        x_new = (np.abs(z) >= np.array([z_norm - 1e-8] * n)).astype(np.float64)
        
        # Check if x_new has been tried before
        for x_ in tried_x:
            if np.allclose(x_, x_new):
                return a_norm * np.linalg.norm(w, ord=1)
        
        x = x_new
        tried_x.append(x)
