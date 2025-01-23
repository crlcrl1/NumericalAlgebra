import jax
import jax.numpy as jnp

from ..equation import StrokesEquation
from .v_cycle import error, v_cycle_iter

jax.config.update("jax_enable_x64", True)

error = jax.jit(error, static_argnames='n')


def iterate(u, v, p, u_err, v_err, p_err, fu, fv, d, n):
    ru, rv, rp = jnp.zeros_like(u_err), jnp.zeros_like(v_err), jnp.zeros_like(p_err)
    ru, rv, rp = jax.jit(v_cycle_iter, static_argnames='n')(ru, rv, rp, u_err, v_err, p_err, n, 2, 2)
    u = u + ru
    v = v + rv
    p = p + rp
    u_err, v_err, p_err = error(u, v, p, fu, fv, d, n)
    iter_error = jnp.sqrt(jnp.linalg.norm(u_err) ** 2 + jnp.linalg.norm(v_err) ** 2)
    return u, v, p, u_err, v_err, p_err, iter_error


class JaxStrokesEquation(StrokesEquation):

    def __init__(self, n: int):
        super().__init__(n)
        self.u = jnp.array(self.u, dtype=jnp.float64)
        self.v = jnp.array(self.v, dtype=jnp.float64)
        self.p = jnp.array(self.p, dtype=jnp.float64)
        self.fu = jnp.array(self.fu, dtype=jnp.float64)
        self.fv = jnp.array(self.fv, dtype=jnp.float64)
        self.d = jnp.array(self.d, dtype=jnp.float64)
        self.real_u = jnp.array(self.real_u, dtype=jnp.float64)
        self.real_v = jnp.array(self.real_v, dtype=jnp.float64)
        print("AOT compiling...")
        self.iter_fn = jax.jit(iterate, static_argnames='n')
        self.iter_fn = self.iter_fn.lower(self.u, self.v, self.p,
                                          jnp.empty_like(self.u), jnp.empty_like(self.v), jnp.empty_like(self.p),
                                          self.fu, self.fv, self.d,
                                          self.n).compile()

    def solve_error(self):
        u_err, v_err, p_err = error(self.u, self.v, self.p, self.fu, self.fv, self.d, self.n)
        return jnp.sqrt(jnp.linalg.norm(u_err) ** 2 + jnp.linalg.norm(v_err) ** 2)

    def solve(self, tol=1e-8, silent=False):
        u_err, v_err, p_err = error(self.u, self.v, self.p, self.fu, self.fv, self.d, self.n)
        init_error = jnp.sqrt(jnp.linalg.norm(u_err) ** 2 + jnp.linalg.norm(v_err) ** 2)
        iter_error = init_error
        iteration = 0
        if not silent:
            self.log_error(iteration, init_error)
        while iter_error > tol * init_error and iteration < 40:
            self.u, self.v, self.p, u_err, v_err, p_err, iter_error = self.iter_fn(self.u, self.v, self.p,
                                                                                   u_err, v_err, p_err,
                                                                                   self.fu, self.fv, self.d)
            iteration += 1
            if not silent:
                self.log_error(iteration, iter_error)
        return iteration

    def pde_error(self):
        u_err = jnp.linalg.norm(self.u - self.real_u) ** 2
        v_err = jnp.linalg.norm(self.v - self.real_v) ** 2
        return jnp.sqrt(u_err + v_err) / self.n

    def reset(self):
        self.u = jnp.zeros_like(self.u)
        self.v = jnp.zeros_like(self.v)
        self.p = jnp.zeros_like(self.p)
        self.d = jnp.zeros_like(self.d)
        return self
