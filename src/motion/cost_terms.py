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

from __future__ import print_function
from common_imports import *
from geometry.differentiable_geometry import *


class FiniteDifferencesAcceleration(AffineMap):

    """ This class allows to define accelerations"""

    def __init__(self, dim, dt):
        self._a = np.matrix(np.zeros((dim, 3 * dim)))
        self._b = np.matrix(np.zeros((dim, 1)))
        self._initialize_matrix(dim, dt)

    def _initialize_matrix(self, dim, dt):
        # Acceleration = [ x_{t+1} + x_{t-1} - 2 * x_t ] / dt^2
        I = np.eye(dim)
        self._a[0:dim, 0:dim] = I
        self._a[0:dim, dim:(2 * dim)] = -2 * I
        self._a[0:dim, (2 * dim):(3 * dim)] = I
        self._a /= (dt * dt)
