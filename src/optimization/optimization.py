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

import common_imports
from geometry.differentiable_geometry import *
import numpy as np


class UnconstraintedOptimizer:

    def __init__(self):
        self._eta = 0.01
        self._f = None

    @abstractmethod
    def one_step(self):
        raise NotImplementedError()


class GradientDescent:

    def one_step(self):
        self.x = self.x - self._eta * self.f_.gradient(x)


class NaturalGradientDescent:

    def __init__(self, A):
        self.A_inv = np.linalg.inv(A)

    def one_step(self):
        self.x = self.x - self._eta * self.A_inv * self.gradient(x)


class NetwtonAlgorithm:

    def one_step(self):
        H = self.f_.hessian(x)
        g = self.f_.gradient(x)
        self.x = self.x - self._eta * np.linalg.solve(H, g)
