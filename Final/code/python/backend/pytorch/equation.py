import torch

from .v_cycle import v_cycle_iter, error
from ..equation import StrokesEquation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TorchStrokesEquation(StrokesEquation):
    def __init__(self, n: int):
        super().__init__(n)

        self.u = torch.from_numpy(self.u).to(device=DEVICE)
        self.v = torch.from_numpy(self.v).to(device=DEVICE)
        self.p = torch.from_numpy(self.p).to(device=DEVICE)
        self.fu = torch.from_numpy(self.fu).to(device=DEVICE)
        self.fv = torch.from_numpy(self.fv).to(device=DEVICE)
        self.d = torch.from_numpy(self.d).to(device=DEVICE)
        self.real_u = torch.from_numpy(self.real_u).to(device=DEVICE)
        self.real_v = torch.from_numpy(self.real_v).to(device=DEVICE)

    def solve(self, tol=1e-8, silent=False):
        u_err, v_err, p_err = error(self.u, self.v, self.p, self.fu, self.fv, self.d, self.n)
        init_error = torch.sqrt(torch.norm(u_err) ** 2 + torch.norm(v_err) ** 2)
        iter_error = init_error
        iteration = 0
        if not silent:
            self.log_error(iteration, init_error)
        while iter_error > tol * init_error and iteration < 40:
            u, v, p = torch.zeros_like(self.u), torch.zeros_like(self.v), torch.zeros_like(self.p)
            v_cycle_iter(u, v, p, u_err, v_err, p_err, self.n)
            self.u += u
            self.v += v
            self.p += p
            u_err, v_err, p_err = error(self.u, self.v, self.p, self.fu, self.fv, self.d, self.n)
            iter_error = torch.sqrt(torch.norm(u_err) ** 2 + torch.norm(v_err) ** 2)
            iteration += 1
            if not silent:
                self.log_error(iteration, iter_error)
        return iteration

    def solve_error(self):
        u_err, v_err, p_err = error(self.u, self.v, self.p, self.fu, self.fv, self.d, self.n)
        return torch.sqrt(torch.norm(u_err) ** 2 + torch.norm(v_err) ** 2)

    def pde_error(self):
        u_err = torch.norm(self.u - self.real_u) ** 2
        v_err = torch.norm(self.v - self.real_v) ** 2
        return torch.sqrt(u_err + v_err) / self.n

    def reset(self):
        self.u = torch.zeros_like(self.u)
        self.v = torch.zeros_like(self.v)
        self.p = torch.zeros_like(self.p)
        self.d = torch.zeros_like(self.d)
        return self
