"""
Naive implementation of algorithms on CPU for reference.
"""

import torch
from torch import Tensor


def gauss_seidel_u(u: Tensor, fu: Tensor, n: int):
    for i in range(0, n, 2):
        for j in range(n - 1):
            left = u[i, j - 1] if j != 0 else 0
            right = u[i, j + 1] if j != n - 2 else 0
            top = u[i + 1, j] if i != n - 1 else 0
            bottom = u[i - 1, j] if i != 0 else 0
            cnt = (i != 0) + (i != n - 1) + 2
            u[i, j] = (fu[i, j] + n * n * (right + left + top + bottom)) / (n * n * cnt)

    for i in range(1, n, 2):
        for j in range(n - 1):
            left = u[i, j - 1] if j != 0 else 0
            right = u[i, j + 1] if j != n - 2 else 0
            top = u[i + 1, j] if i != n - 1 else 0
            bottom = u[i - 1, j] if i != 0 else 0
            cnt = (i != 0) + (i != n - 1) + 2
            u[i, j] = (fu[i, j] + n * n * (right + left + top + bottom)) / (n * n * cnt)


def gauss_seidel_v(v: Tensor, fv: Tensor, n: int):
    for i in range(0, n, 2):
        for j in range(n - 1):
            bottom = v[j - 1, i] if j != 0 else 0
            top = v[j + 1, i] if j != n - 2 else 0
            right = v[j, i + 1] if i != n - 1 else 0
            left = v[j, i - 1] if i != 0 else 0
            cnt = (i != 0) + (i != n - 1) + 2
            v[j, i] = (fv[j, i] + n * n * (right + left + top + bottom)) / (n * n * cnt)

    for i in range(1, n, 2):
        for j in range(n - 1):
            bottom = v[j - 1, i] if j != 0 else 0
            top = v[j + 1, i] if j != n - 2 else 0
            right = v[j, i + 1] if i != n - 1 else 0
            left = v[j, i - 1] if i != 0 else 0
            cnt = (i != 0) + (i != n - 1) + 2
            v[j, i] = (fv[j, i] + n * n * (right + left + top + bottom)) / (n * n * cnt)


def update_pressure(u: Tensor, v: Tensor, p: Tensor, d: Tensor, n: int):
    for i in range(n):
        for j in range(n):
            right = u[i, j] if j != n - 1 else 0
            left = u[i, j - 1] if j != 0 else 0
            top = v[i, j] if i != n - 1 else 0
            bottom = v[i - 1, j] if i != 0 else 0
            r = -d[i, j] - n * (right - left + top - bottom)
            cnt = (i != 0) + (i != n - 1) + (j != 0) + (j != n - 1)

            delta = r / (n * cnt)
            p[i, j] += r
            if i != 0:
                p[i - 1, j] -= r / cnt
                v[i - 1, j] -= delta
            if i != n - 1:
                p[i + 1, j] -= r / cnt
                v[i, j] += delta
            if j != 0:
                p[i, j - 1] -= r / cnt
                u[i, j - 1] -= delta
            if j != n - 1:
                p[i, j + 1] -= r / cnt
                u[i, j] += delta


def restrict_u(u: Tensor, n: int):
    n_new = n // 2
    u_restrict = torch.zeros((n_new, n_new - 1), device='cuda', dtype=torch.float64)
    for i in range(n_new):
        for j in range(n_new - 1):
            x, y = 2 * i, 2 * j
            new_val = (u[x, y + 1] + u[x + 1, y + 1]) / 4 + (u[x, y] + u[x + 1, y] + u[x, y + 2] + u[x + 1, y + 2]) / 8
            u_restrict[i, j] = new_val
    return u_restrict


def restrict_v(v: Tensor, n: int):
    n_new = n // 2
    v_restrict = torch.zeros((n_new - 1, n_new), device='cuda', dtype=torch.float64)
    for i in range(n_new - 1):
        for j in range(n_new):
            x, y = 2 * i, 2 * j
            new_val = (v[x + 1, y] + v[x + 1, y + 1]) / 4 + (v[x, y] + v[x, y + 1] + v[x + 2, y] + v[x + 2, y + 1]) / 8
            v_restrict[i, j] = new_val
    return v_restrict


def restrict_p(p: Tensor, n: int):
    n_new = n // 2
    p_restrict = torch.zeros((n_new, n_new), device='cuda', dtype=torch.float64)
    for i in range(n_new):
        for j in range(n_new):
            x, y = 2 * i, 2 * j
            new_val = (p[x, y] + p[x + 1, y] + p[x, y + 1] + p[x + 1, y + 1]) / 4
            p_restrict[i, j] = new_val
    return p_restrict


def lift_u(u: Tensor, n: int):
    u_lift = torch.zeros((n, n - 1), device='cuda', dtype=torch.float64)
    for i in range(n):
        for j in range(n - 1):
            x, y = i // 2, j // 2
            nearest = u[x, y] if j != n - 2 else 0
            left = u[x, y - 1] if j != 0 else 0
            u_lift[i, j] = (nearest + left) / 2 if j % 2 == 0 else nearest
    return u_lift


def lift_v(v: Tensor, n: int):
    v_lift = torch.zeros((n - 1, n), device='cuda', dtype=torch.float64)
    for i in range(n - 1):
        for j in range(n):
            x, y = i // 2, j // 2
            nearest = v[x, y] if i != n - 2 else 0
            bottom = v[x - 1, y] if i != 0 else 0
            v_lift[i, j] = (nearest + bottom) / 2 if i % 2 == 0 else nearest
    return v_lift


def lift_p(p: Tensor, n: int):
    p_lift = torch.zeros((n, n), device='cuda', dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            x, y = i // 2, j // 2
            nearest = p[x, y]
            p_lift[i, j] = nearest
    return p_lift
