import jax
import jax.numpy as jnp
import jax_triton as jt
from jax import Array

from kernel import *


def error(u: Array, v: Array, p: Array, fu: Array, fv: Array, d: Array, n: int):
    block_size = 64 if n >= 64 else n
    grid = (n * (n - 1) // block_size,)
    out_shape_u = jax.ShapeDtypeStruct(shape=u.shape, dtype=u.dtype)
    out_shape_v = jax.ShapeDtypeStruct(shape=v.shape, dtype=v.dtype)
    out_shape_p = jax.ShapeDtypeStruct(shape=p.shape, dtype=p.dtype)
    u_error: Array = jt.triton_call(u, p, fu, kernel=error_u_kernel, out_shape=out_shape_u, grid=grid,  # type: ignore
                                    n=n, block_size=block_size)
    v_error: Array = jt.triton_call(v, p, fv, kernel=error_v_kernel, out_shape=out_shape_v, grid=grid,  # type: ignore
                                    n=n, block_size=block_size)
    grid = (n * n // block_size,)
    p_error: Array = jt.triton_call(u, v, d, kernel=error_p_kernel, out_shape=out_shape_p, grid=grid,  # type: ignore
                                    n=n, block_size=block_size)

    return u_error, v_error, p_error


def gauss_seidel(u: Array, v: Array, fu: Array, fv: Array, n: int):
    block_size = 128 if n >= 256 else n // 2
    grid = (n // 2 // block_size,)

    out_shape_u = jax.ShapeDtypeStruct(shape=u.shape, dtype=u.dtype)
    out_shape_v = jax.ShapeDtypeStruct(shape=v.shape, dtype=v.dtype)
    u = jt.triton_call(u, fu, kernel=gs_kernel_u, out_shape=out_shape_u, grid=grid, n=n,  # type: ignore
                       block_size=block_size)
    v = jt.triton_call(v, fv, kernel=gs_kernel_v, out_shape=out_shape_v, grid=grid, n=n,  # type: ignore
                       block_size=block_size)
    return u, v


def dgs(u: Array, v: Array, p: Array, fu: Array, fv: Array, d: Array, n: int):
    block_size = 64 if n >= 64 else n
    grid = (n * (n - 1) // block_size,)
    out_shape_fu = jax.ShapeDtypeStruct(shape=fu.shape, dtype=fu.dtype)
    out_shape_fv = jax.ShapeDtypeStruct(shape=fv.shape, dtype=fv.dtype)
    fu_new: Array = jt.triton_call(fu, p, kernel=cal_fu_kernel, out_shape=out_shape_fu, grid=grid,  # type: ignore
                                   n=n, block_size=block_size)
    fv_new: Array = jt.triton_call(fv, p, kernel=cal_fv_kernel, out_shape=out_shape_fv, grid=grid,  # type: ignore
                                   n=n, block_size=block_size)

    u, v = gauss_seidel(u, v, fu_new, fv_new, n)

    block_size = 64 if n >= 256 else n // 4
    grid = (n // 4 // block_size,)
    out_shape_u = jax.ShapeDtypeStruct(shape=u.shape, dtype=u.dtype)
    out_shape_v = jax.ShapeDtypeStruct(shape=v.shape, dtype=v.dtype)
    out_shape_p = jax.ShapeDtypeStruct(shape=p.shape, dtype=p.dtype)
    u, v, p = jt.triton_call(u, v, p, d, kernel=update_pressure_kernel,  # type: ignore
                             out_shape=(out_shape_u, out_shape_v, out_shape_p), grid=grid,
                             n=n, block_size=block_size, total_num=4, num_warps=2)
    return u, v, p


def restrict(u: Array, v: Array, p: Array, n: int):
    n_new = n // 2
    block_size = 64 if n_new >= 64 else n_new

    grid = (n_new * (n_new - 1) // block_size,)
    out_shape_u = jax.ShapeDtypeStruct(shape=(n_new, n_new - 1), dtype=u.dtype)
    out_shape_v = jax.ShapeDtypeStruct(shape=(n_new - 1, n_new), dtype=v.dtype)
    out_shape_p = jax.ShapeDtypeStruct(shape=(n_new, n_new), dtype=p.dtype)
    u_restrict: Array = jt.triton_call(u, kernel=restrict_u_kernel, out_shape=out_shape_u, grid=grid,  # type: ignore
                                       n=n, block_size=block_size)
    v_restrict: Array = jt.triton_call(v, kernel=restrict_v_kernel, out_shape=out_shape_v, grid=grid,  # type: ignore
                                       n=n, block_size=block_size)
    grid = (n_new * n_new // block_size,)
    p_restrict: Array = jt.triton_call(p, kernel=restrict_p_kernel, out_shape=out_shape_p, grid=grid,  # type: ignore
                                       n=n, block_size=block_size)

    return u_restrict, v_restrict, p_restrict


def lift(u: Array, v: Array, p: Array, n: int):
    block_size = 64 if n >= 64 else n

    grid = (n * (n - 1) // block_size,)
    out_shape_u = jax.ShapeDtypeStruct(shape=(n, n - 1), dtype=u.dtype)
    out_shape_v = jax.ShapeDtypeStruct(shape=(n - 1, n), dtype=v.dtype)
    out_shape_p = jax.ShapeDtypeStruct(shape=(n, n), dtype=p.dtype)
    u_lift: Array = jt.triton_call(u, kernel=lift_u_kernel, out_shape=out_shape_u, grid=grid,  # type: ignore
                                   n=n, block_size=block_size)
    v_lift: Array = jt.triton_call(v, kernel=lift_v_kernel, out_shape=out_shape_v, grid=grid,  # type: ignore
                                   n=n, block_size=block_size)
    grid = (n * n // block_size,)
    p_lift: Array = jt.triton_call(p, kernel=lift_p_kernel, out_shape=out_shape_p, grid=grid,  # type: ignore
                                   n=n, block_size=block_size)

    return u_lift, v_lift, p_lift


def v_cycle_iter(u: Array, v: Array, p: Array, fu: Array, fv: Array, d: Array, n: int, v1: int, v2: int):
    u, v, p = jax.lax.fori_loop(0, v1, lambda i, args: jax.jit(dgs, static_argnames='n')(*args, fu, fv, d, n),
                                (u, v, p))

    if n > 4:
        u_err, v_err, p_err = jax.jit(error, static_argnames='n')(u, v, p, fu, fv, d, n)
        u_err, v_err, p_err = jax.jit(restrict, static_argnames='n')(u_err, v_err, p_err, n)
        u_new = jnp.zeros_like(u_err)
        v_new = jnp.zeros_like(v_err)
        p_new = jnp.zeros_like(p_err)

        u_new, v_new, p_new = jax.jit(v_cycle_iter, static_argnames='n')(u_new, v_new, p_new, u_err, v_err, p_err,
                                                                         n // 2, v1, v2)

        u_new, v_new, p_new = jax.jit(lift, static_argnames='n')(u_new, v_new, p_new, n)
        u += u_new
        v += v_new
        p += p_new

    u, v, p = jax.lax.fori_loop(0, v2, lambda i, args: jax.jit(dgs, static_argnames='n')(*args, fu, fv, d, n),
                                (u, v, p))
    return u, v, p
