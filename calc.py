import numpy as np
from multiprocessing import Pool
import pickle
from math import log10, ceil
from numba import njit
from datetime import datetime
from tqdm import tqdm


@njit(fastmath=True)
def laplacian(Z, dt, dx):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (dt) * (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx / dx


@njit(fastmath=True)
def iter(U, V, Du, Dv, p, q, dt, dx):
    deltaU = laplacian(U, dt, dx)
    deltaV = laplacian(V, dt, dx)
    # We take the values of u and v inside the grid.
    Uc = U[1:-1, 1:-1]
    Vc = V[1:-1, 1:-1]
    # We update the variables.
    U[1:-1, 1:-1], V[1:-1, 1:-1] = Uc + Du * deltaU + dt * (
        1 - Uc * Vc
    ), Vc + Dv * deltaV + dt * (p * Vc * (Uc - (1 + q) / (q + Vc)))
    # Neumann conditions: derivatives at the edges
    # are null.
    for Z in (U, V):
        Z[0, :] = Z[1, :]
        Z[-1, :] = Z[-2, :]
        Z[:, 0] = Z[:, 1]
        Z[:, -1] = Z[:, -2]
    return U, V


def calc_process(
    save_every_n_steps,
    U,
    V,
    du,
    dv,
    p,
    q,
    dt,
    dx,
    eps=1e-4,
    progress=False,
    t_max=10_000,
):
    should_continue = True
    i = 0
    U_old = None
    res_U = []
    res_V = []
    if progress:
        bar = tqdm()
    while should_continue and i * dt <= t_max:
        if i % save_every_n_steps == 0:
            if np.isnan(U).any() > 0:
                print("boom")
                return np.array(res_U), np.array(res_V), "boom"
            if progress and i != 0:
                bar.update(save_every_n_steps)
            res_U.append(U.copy())
            res_V.append(V.copy())
            U_old = U.copy()
            U, V = iter(U, V, du, dv, p, q, dt, dx)
            if np.linalg.norm(U - U_old) < eps:
                should_continue = False
        else:
            U, V = iter(U, V, du, dv, p, q, dt, dx)
        i += 1
    return np.array(res_U), np.array(res_V), str(not should_continue)


class Exp:
    def __init__(
        self, save_every_n_steps, init_U, init_V, du, dv, p, q, dt, dx, eps, t_max
    ) -> None:
        du_star = (q + 1) / p * (2 * q + 1 + 2 * (q * (q + 1)) ** 0.5)
        print(f"du/dv {du/dv} >= du_star {du_star}")
        assert du / dv >= du_star
        print(f"dx = {dx} > {(dt * 2 * max(du, dv)) ** 0.5}")
        assert dx >= (dt * 2 * max(du, dv)) ** 0.5

        self.save_every_n_steps = save_every_n_steps
        self.init_U = init_U
        self.init_V = init_V
        self.du = du
        self.dv = dv
        self.p = p
        self.q = q
        self.dt = dt
        self.dx = dx
        self.eps = eps
        self.t_max = t_max

    def calc(self, progress=False):
        self.process_U, self.process_V, self.converged = calc_process(
            self.save_every_n_steps,
            self.init_U,
            self.init_V,
            self.du,
            self.dv,
            self.p,
            self.q,
            self.dt,
            self.dx,
            self.eps,
            progress,
            self.t_max,
        )


if __name__ == "__main__":
    size = 161
    exps = []
    p = 1
    dv = 0.0001
    for q in [0.1]:
        du_star = (q + 1) / p * (2 * q + 1 + 2 * (q * (q + 1)) ** 0.5)
        du_min = round((du_star + 1) * dv, 4)
        leng = ceil(-log10(du_min))
        for du in [0.0003 + 0.000025*i for i in range(0,51)]:
            for _ in range(50):
                exps.append(
                    Exp(
                        save_every_n_steps=1000000,
                        init_U=np.random.rand(size, size) * 0.1 + 1,
                        init_V=np.random.rand(size, size) * 0.1 + 1,
                        du=du,
                        dv=dv,
                        p=p,
                        q=q,
                        dt=0.0005,
                        dx=0.00625,
                        eps=1e-5,
                        t_max=5000,
                    )
                )
    from collections import defaultdict

    res = defaultdict(list)
    for e in exps:
        res[e.q].append(e.du)
    import json

    print(len(exps))
    print(json.dumps(res, indent=2))

    def run(data):
        i, x = data
        x.calc()
        print(datetime.now(), i, x.converged )
        with open(f"q01_snd_data/{i}", "wb") as out:
            pickle.dump(x, out)

    with Pool(30) as p:
        p.map(run, [(f"3_{i}", x) for i, x in tqdm(enumerate(exps))])
