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

# from __future__ import print_function
from common_imports import *
from geometry.differentiable_geometry import *


class FiniteDifferencesAcceleration(AffineMap):

    """ This class allows to define accelerations"""

    def __init__(self, dim, dt):
        self._a = np.matrix(np.zeros((dim, 3 * dim)))
        self._b = np.matrix(np.zeros((dim, 1)))
        self._initialize_matrix(dim, dt)
        print "input dimension : ", self.input_dimension()
        print "output dimension : ", self.output_dimension()

    def _initialize_matrix(self, dim, dt):
        # Acceleration = [ x_{t+1} + x_{t-1} - 2 * x_t ] / dt^2
        I = np.eye(dim)
        self._a[0:dim, 0:dim] = I
        self._a[0:dim, dim:(2 * dim)] = -2 * I
        self._a[0:dim, (2 * dim):(3 * dim)] = I
        self._a /= (dt * dt)

    def a(self):
        return self._a.copy()


class FiniteDifferencesVelocity(AffineMap):

    """ This class allows to define velocities"""

    def __init__(self, dim, dt):
        self._a = np.matrix(np.zeros((dim, 2 * dim)))
        self._b = np.matrix(np.zeros((dim, 1)))
        self._initialize_matrix(dim, dt)
        print "input dimension : ", self.input_dimension()
        print "output dimension : ", self.output_dimension()

    def _initialize_matrix(self, dim, dt):
        # Acceleration = [ x_{t+1} - x_{t} ] / dt
        I = np.eye(dim)
        self._a[0:dim, 0:dim] = -I
        self._a[0:dim, dim:(2 * dim)] = I
        self._a /= dt

    def a(self):
        return self._a.copy()


class ObstaclePotential2D(DifferentiableMap):

    def __init__(self, signed_distance_field):
        assert signed_distance_field.input_dimension() == 2
        assert signed_distance_field.output_dimension() == 1
        self._sdf = signed_distance_field
        self._rho_scaling = 1.e-3
        self._alpha = 20.

    def output_dimension(self):
        return 3

    def input_dimension(self):
        return 2

    def forward(self, x):
        rho = np.exp(-self._alpha * self._sdf.forward(x))
        y = np.zeros(3)
        y[0] = self._rho_scaling * rho
        y[1] = x[0]
        y[2] = x[1]
        return y

    def jacobian(self, x):
        [sdf, J_sdf] = self._sdf.evaluate(x)
        rho = np.exp(-self._alpha * sdf)
        J = np.matrix(np.zeros((3, 2)))
        J[0, :] = -self._alpha * self._rho_scaling * rho * J_sdf
        J[1:3, :] = np.matrix(np.eye(2, 2))
        return J
