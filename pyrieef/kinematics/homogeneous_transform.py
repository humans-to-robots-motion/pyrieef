#!/usr/bin/env python

# Copyright (c) 2020, University of Stuttgart
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
#                                      Jim Mainprice on Sunday January 13 2020

from geometry.differentiable_geometry import *
from geometry.rotations import *


class PlanarRotation(DifferentiableMap):
    """
    Planar Rotation as DifferentiableMap

        Takes an angle and rotates the point p0 by this angle

            f(theta) = R(theta) * p_0

        p0 can be defined as a keypoin, defined constant in a frame of
        reference. Theta is a degree of freedom.
    """

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
        assert q.size == 1
        J = np.zeros((2, 1))
        c, s = np.cos(q[0]), np.sin(q[0])
        J[:, 0] = np.dot(np.array(((-s, -c), (c, -s))), self._p)
        return J


class HomogeneousTransform(DifferentiableMap):
    """
    Homeogeneous transformation as DifferentiableMap

        Takes an angle and rotates the point p0 by this angle

            f(q) = T(q) * p_0

        where T defines a rotation and translation (3DoFs)
            q_{0,1}     => translation
            q_{2}       => rotation

                T = [ R(q)  p(q) ]
                    [ 0 0    1   ]
    """

    def __init__(self, p0=np.zeros(2)):
        assert p0.size == 2
        self._n = p0.size + 1
        self._T = np.eye(self.input_dimension())
        self._p = np.ones(self.input_dimension())
        self._p[:self.output_dimension()] = p0

    def output_dimension(self):
        return self._n - 1

    def input_dimension(self):
        return self._n

    def forward(self, q):
        assert q.size == self.input_dimension()
        dim = self.output_dimension()
        self._T[:dim, :dim] = rotation_matrix_2d_radian(q[dim])
        self._T[:dim, dim] = q[:dim]
        return np.dot(self._T, self._p)[:dim]

    def jacobian(self, q):
        """ Should return a matrix or single value of
                m x n : [output : 2 x input : 3] (dimensions)"""
        assert q.size == self.input_dimension()
        J = np.zeros((self.output_dimension(), self.input_dimension()))
        J[0, 0] = 1.
        J[1, 1] = 1.
        dim = self.output_dimension()
        c, s = np.cos(q[dim]), np.sin(q[dim])
        J[:, 2] = np.dot(np.array(((-s, -c), (c, -s))), self._p[:dim])
        return J
