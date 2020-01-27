#!/usr/bin/env python

# Copyright (c) 2018, University of Stuttgart
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
#                                        Jim Mainprice on Sunday June 13 2018

from geometry.differentiable_geometry import *
from geometry.rotations import *


class PlanarRotation(DifferentiableMap):
    """ Planar Rotation as DifferentiableMap """

    def __init__(self, p0):
        assert p0.size == 2
        self._p = p0
        return

    def output_dimension(self):
        return 2

    def input_dimension(self):
        return 1

    def forward(self, q):
        assert q.size == 1
        return np.dot(rotation_matrix_2d_radian(q[0]), self._p)

    def jacobian(self, q):
        J = np.zeros((2, 1))
        theta = q[0]
        c, s = np.cos(theta), np.sin(theta)
        T = np.array(((-s, -c), (c, -s)))
        J[:, 0] = np.dot(T, self._p)
        return J


class HomogenousTransform(DifferentiableMap):
    """ HomeogenousTransform as DifferentiableMap """

    def __init__(self, p0=np.zeros(2)):
        assert p0.size == 2
        self._n = 3
        self._T = np.eye(self.input_dimension())
        self._p = np.ones(self.input_dimension())
        self._p[:self.output_dimension()] = p0

    def output_dimension(self):
        return self._n - 1

    def input_dimension(self):
        return self._n

    def forward(self, q):
        dim = self.output_dimension()
        self._T[:dim, :dim] = rotation_matrix_2d_radian(q[dim])
        self._T[:dim, dim] = q[:dim]
        return np.dot(self._T, self._p)[:dim]

    def jacobian(self, q):
        """ Should return a matrix or single value of
                m x n : [output : 2 x input : 3] (dimensions)
            by default the method returns the finite difference jacobian.
            WARNING the object returned by this function is a numpy matrix."""
        dim = self.output_dimension()
        J = np.zeros((self.output_dimension(), self.input_dimension()))
        J[0, 0] = 1.
        J[1, 1] = 1.
        J[0, 2] = -np.sin(q[dim]) * self._p[0] - np.cos(q[dim]) * self._p[1]
        J[1, 2] = np.cos(q[dim]) * self._p[0] - np.sin(q[dim]) * self._p[1]
        return J
