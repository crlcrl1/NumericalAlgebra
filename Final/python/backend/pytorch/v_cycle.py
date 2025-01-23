import torch
from torch import Tensor

from kernel import *

stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()


def error(u: Tensor, v: Tensor, p: Tensor, fu: Tensor, fv: Tensor, d: Tensor, n: int):
    block_size = 64 if n >= 64 else n
    u_error = torch.empty_like(u)
    v_error = torch.empty_like(v)
    p_error = torch.empty_like(p)

    grid = lambda meta: (triton.cdiv(n * (n - 1), meta["block_size"]),)
    error_u_kernel[grid](u, p, fu, u_error, n, block_size)
    error_v_kernel[grid](v, p, fv, v_error, n, block_size)
    grid = lambda meta: (triton.cdiv(n * n, meta["block_size"]),)
    error_p_kernel[grid](u, v, d, p_error, n, block_size)
    torch.cuda.synchronize()

    return u_error, v_error, p_error


def dgs(u: Tensor, v: Tensor, p: Tensor, fu: Tensor, fv: Tensor, d: Tensor, n: int):
    block_size = 64 if n >= 64 else n
    fu_new = torch.empty_like(fu)
    fv_new = torch.empty_like(fv)
    grid = lambda meta: (triton.cdiv(n * (n - 1), meta["block_size"]),)
    cal_fu_kernel[grid](fu, p, fu_new, n, block_size)
    cal_fv_kernel[grid](fv, p, fv_new, n, block_size)
    torch.cuda.synchronize()

    gauss_seidel(u, v, fu_new, fv_new, n)

    block_size = 64 if n >= 256 else n // 4
    grid = lambda meta: (triton.cdiv(n // 4, meta["block_size"]),)
    update_pressure_kernel_inplace[grid](u, v, p, d, n, 4, block_size)
    torch.cuda.synchronize()


def gauss_seidel(u: Tensor, v: Tensor, fu: Tensor, fv: Tensor, n: int):
    block_size = 128 if n >= 256 else n // 2
    grid = lambda meta: (triton.cdiv(n // 2, meta["block_size"]),)

    with torch.cuda.stream(stream1):
        gs_kernel_u_inplace[grid](u, fu, n, block_size)
    with torch.cuda.stream(stream2):
        gs_kernel_v_inplace[grid](v, fv, n, block_size)
    torch.cuda.synchronize()


def restrict(u: Tensor, v: Tensor, p: Tensor, n: int):
    n_new = n // 2
    block_size = 64 if n_new >= 64 else n_new
    u_restrict = torch.empty((n_new, n_new - 1), device='cuda', dtype=torch.float64)
    v_restrict = torch.empty((n_new - 1, n_new), device='cuda', dtype=torch.float64)
    p_restrict = torch.empty((n_new, n_new), device='cuda', dtype=torch.float64)

    grid = lambda meta: (triton.cdiv(n_new * (n_new - 1), meta["block_size"]),)
    restrict_u_kernel[grid](u, u_restrict, n, block_size)
    restrict_v_kernel[grid](v, v_restrict, n, block_size)
    grid = lambda meta: (triton.cdiv(n_new * n_new, meta["block_size"]),)
    restrict_p_kernel[grid](p, p_restrict, n, block_size)
    torch.cuda.synchronize()

    return u_restrict, v_restrict, p_restrict


def lift(u: Tensor, v: Tensor, p: Tensor, n: int):
    block_size = 64 if n >= 64 else n
    u_lift = torch.empty((n, n - 1), device='cuda', dtype=torch.float64)
    v_lift = torch.empty((n - 1, n), device='cuda', dtype=torch.float64)
    p_lift = torch.empty((n, n), device='cuda', dtype=torch.float64)

    grid = lambda meta: (triton.cdiv(n * (n - 1), meta["block_size"]),)
    lift_u_kernel[grid](u, u_lift, n, block_size)
    lift_v_kernel[grid](v, v_lift, n, block_size)
    grid = lambda meta: (triton.cdiv(n * n, meta["block_size"]),)
    lift_p_kernel[grid](p, p_lift, n, block_size)
    torch.cuda.synchronize()

    return u_lift, v_lift, p_lift


def v_cycle_iter(u: Tensor, v: Tensor, p: Tensor, fu: Tensor, fv: Tensor, d: Tensor, n: int, v1: int, v2: int):
    for i in range(v1):
        dgs(u, v, p, fu, fv, d, n)

    if n > 4:
        u_err, v_err, p_err = error(u, v, p, fu, fv, d, n)
        u_err, v_err, p_err = restrict(u_err, v_err, p_err, n)
        u_new = torch.zeros_like(u_err)
        v_new = torch.zeros_like(v_err)
        p_new = torch.zeros_like(p_err)

        v_cycle_iter(u_new, v_new, p_new, u_err, v_err, p_err, n // 2, v1, v2)

        u_new, v_new, p_new = lift(u_new, v_new, p_new, n)
        u += u_new
        v += v_new
        p += p_new

    for i in range(v2):
        dgs(u, v, p, fu, fv, d, n)
