#!/usr/bin/env python

# Copyright (c) 2021
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                          Jim Mainprice on Firday July 28 2021

import torch
import numpy as np
from scipy import optimize
import time


class TrajectoryFunctionNetwork:

    """
    The trajectory is commanded in velocities
    """

    def __init__(self, n, T, dt, q_init, q_goal):
        assert n == len(q_init)
        assert n == len(q_goal)
        self.n = n
        self.q_init = torch.from_numpy(np.asarray(q_init, dtype=np.float64))
        self.q_goal = torch.from_numpy(np.asarray(q_goal, dtype=np.float64))
        self.f = [None] * T
        self.dq = [None] * T
        self.dt = dt
        self.T = T

    def create_network(self, x):
        self.f_t = [None] * self.T
        for t in range(self.T):
            q = self.q_init if t == 0 else self.f[t - 1]
            dq = x[t * self.n: (t + 1) * self.n]
            self.dq[t] = torch.tensor(dq, requires_grad=True)
            self.f[t] = q + self.dq[t] * self.dt
        return torch.nn.MSELoss()(self.f[self.T - 1], self.q_goal)

    def forward(self, x):
        with torch.no_grad():
            cost = self.create_network(x)
            value = cost.item()
        return value

    def gradient(self, x):
        cost = self.create_network(x)
        cost.backward()
        g = np.array([0.] * len(x))
        for t in range(T):
            g[t * self.n: (t + 1) * self.n] = self.dq[t].grad
        return g

    def shoot(self, x):
        q = np.zeros((self.T + 1) * self.n)
        q[:self.n] = self.q_init.numpy()
        for t in range(1,  self.T + 1):
            qt = q[(t - 1) * self.n: t * self.n]
            dq = x[(t - 1) * self.n: t * self.n]
            q[t * self.n: (t + 1) * self.n] = qt + dq * self.dt
        return q


if __name__ == "__main__":
    n = 2
    T = 10
    dt = .1
    x = np.random.rand(T * n)
    q_init = [0., 0.]
    q_goal = [1., 1.]
    network = TrajectoryFunctionNetwork(n, T, dt, q_init, q_goal)
    verbose = True
    t_0 = time.time()
    res = optimize.minimize(
        x0=x,
        method='BFGS',
        fun=network.forward,
        jac=network.gradient,
        tol=1e-9,
        options={'maxiter': 100, 'disp': verbose}
    )
    if verbose:
        print(("optimization done in {} sec.".format(time.time() - t_0)))
        print(("gradient norm : ", np.linalg.norm(res.jac)))
    print("res : ", res.x)
    print("trajectory : ", network.shoot(res.x))
