#!/usr/bin/env python

# Copyright (c) 2015 Max Planck Institute
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
# Jim Mainprice on Sunday June 17 2018

from . import common_imports
from geometry.differentiable_geometry import *
import numpy as np


class UnconstraintedOptimizer:

    def __init__(self, f):
        self._eta = 0.0003
        self._f = f

    @abstractmethod
    def one_step(self, x):
        raise NotImplementedError()

    def objective(self, x):
        return self._f(x)

    def gradient(self, x):
        return self._f.gradient(x)

    def set_eta(self, eta):
        self._eta = eta


class GradientDescent(UnconstraintedOptimizer):

    def one_step(self, x):
        g = self.f_.gradient(x)
        return x - self._eta * g / np.linalg.norm(g)


class NaturalGradientDescent(UnconstraintedOptimizer):

    def __init__(self, f, A):
        UnconstraintedOptimizer.__init__(self, f)
        # self.A_inv = np.eye(A.shape[0])
        self.A_inv = np.linalg.inv(A)
        self.A_inv /= np.max(self.A_inv)
        # np.savetxt('A_inv.txt', self.A_inv, fmt='%.2f')
        # self.A_inv = np.eye(self.A_inv.shape[0])

    def one_step(self, x):
        return x - self.delta(x)

    def delta(self, x):
        g = self._f.gradient(x)
        g_t = np.matrix(g).T
        delta = self.A_inv * g_t / np.linalg.norm(g)
        return self._eta * np.array(delta).reshape(x.size)


class NetwtonAlgorithm(UnconstraintedOptimizer):

    def one_step(self, x):
        H = self.f_.hessian(x)
        g = self.f_.gradient(x)
        return x - self._eta * np.linalg.solve(H, g)
