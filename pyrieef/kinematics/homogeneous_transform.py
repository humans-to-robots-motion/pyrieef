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
from math import pi, cos, sin, atan2


class Rotation2D:
    """
    2D rotation

        Uses multiplication operator to combine operations

    Members
    ----------
    _matrix : array
        Orthogonal matrix

    Parameters
    ----------
    rotation : float or array
        in radians
    """

    _matrix = None

    def __init__(self, rotation=None):

        if rotation is not None:
            if isinstance(rotation, float):
                self._matrix = rotation_matrix_2d_radian(rotation)
            elif isinstance(rotation, np.array):
                assert rotation.shape == (2, 2)
                self._matrix = rotation

    def __mul__(self, x):
        return np.dot(self._matrix, x)

    def matrix(self):
        return self._matrix

    def inverse(self):
        return self._matrix.T

    def angle(self):
        return angle_modulo_2i(angle_from_matrix_2d(self._matrix))


class Isometry:
    """
    Interface for isometries
    (i.e., affine transform with orthogonal linear part)

    Members
    ----------
    rotation : array like (n, n)
        rotation matrix

    translation : array like (n, )
        vector offset
    """

    _rotation = None
    _translation = None

    def __str__(self):
        return str(self.matrix())

    def linear(self):
        return self._rotation

    def translation(self):
        return self._translation

    def matrix(self):
        return None

    def inverse(self):
        return None


class Isometry2D(Isometry):
    """
    Affine 2d rigid body transformation.

    Parameters
    ----------
    rotation : float or array
        in radians

    translation : array like (2, )
        vector offset
    """

    def __init__(self, rotation=None, translation=None):

        if rotation is not None:
            self._rotation = Rotation2D(rotation).matrix()

        if translation is not None:
            assert len(translation) == 2
            translation = np.asarray(translation)
            self._translation = translation

    def __mul__(self, p):

        if isinstance(p, np.ndarray):
            return np.dot(self._rotation, p) + self._translation
        elif isinstance(p, Isometry2D):
            m = np.dot(self.matrix(), p.matrix())
            affine = Isometry2D()
            affine._rotation = m[:2, :2]
            affine._translation = m[:2, 2].T
            return affine
        else:
            raise TypeError("transforms only compose vectors or transforms")

    def matrix(self):
        m = np.identity(3)
        m[:2, :2] = self._rotation
        m[:2, 2] = self._translation.T
        return m

    def inverse(self):
        T_inv = Isometry2D()
        T_inv._rotation = self._rotation.T
        T_inv._translation = -np.dot(T_inv._rotation, self._translation)
        return T_inv

    def angle(self):
        return angle_modulo_2i(angle_from_matrix_2d(self._rotation))

    def pose(self):
        return np.hstack([self._translation, self.angle()])


class Isometry3D(Isometry):
    """
    Affine rigid body transformation.
    """

    def __init__(self, rotation=None, translation=None):

        if rotation is not None:
            rotation = np.asarray(rotation)
            assert(rotation.shape == (3, 3))
            self._rotation = rotation
        else:
            self._rotation = None

        if translation is not None:
            assert(translation.size == 3)
            self._translation = translation
        else:
            self._translation = None

    def __mul__(self, p):
        if isinstance(p, np.ndarray):
            return np.dot(self._rotation, p) + self._translation
        elif isinstance(p, Isometry3D):
            m = np.dot(self.matrix(), p.matrix())
            affine = Isometry3D()
            affine._rotation = m[:3, :3]
            affine._translation = m[:3, 3].T
            return affine
        else:
            raise TypeError("transforms only compose vectors or transforms")

    def linear(self):
        return self._rotation

    def translation(self):
        return self._translation

    def matrix(self):
        m = np.identity(4)
        m[:3, :3] = self._rotation
        m[:3, 3] = self._translation.T
        return m

    def inverse(self):
        T_inv = Isometry3D()
        T_inv._rotation = self._rotation.T
        T_inv._translation = -np.dot(T_inv._rotation, self._translation)
        return T_inv


class PlanarRotation(DifferentiableMap):
    """
    Planar Rotation as DifferentiableMap

    details:

        Takes an angle and rotates the point p0 by this angle

            f(theta) = R(theta) * p_0

        p0 can be defined as a keypoin, defined constant in a frame of
        reference. Theta is a degree of freedom.

    Parameters
    ----------
    p0 : array-like, shape (2, )
            Points of which are rotated
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


class HomogeneousTransform2D(DifferentiableMap):
    """
    Homeogeneous transformation in the plane as DifferentiableMap

    details:

        Takes an angle and rotates the point p0 by this angle

            f(q) = T(q) * p_0

        where T defines a rotation and translation (3DoFs)
            q_{0,1}     => translation
            q_{2}       => rotation

                T = [ R(q)  p(q) ]
                    [ 0 0    1   ]

    Parameters
    ----------
    p0 : array-like, shape (2, )
            Points of which are rotated
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

    def point(self):
        return self._p[:self.output_dimension()]

    def forward(self, q):
        assert len(q) == self.input_dimension()
        q = np.asarray(q)
        dim = self.output_dimension()
        self._T[:dim, :dim] = rotation_matrix_2d_radian(q[dim])
        self._T[:dim, dim] = q[:dim]
        return np.dot(self._T, self._p)[:dim]

    def jacobian(self, q):
        """ Should return a matrix or single value of
                m x n : [output : 2 x input : 3] (dimensions)"""

        assert len(q) == self.input_dimension()
        q = np.asarray(q)
        J = np.zeros((self.output_dimension(), self.input_dimension()))
        J[0, 0] = 1.
        J[1, 1] = 1.
        dim = self.output_dimension()
        c, s = np.cos(q[dim]), np.sin(q[dim])
        J[:, 2] = np.dot(np.array(((-s, -c), (c, -s))), self._p[:dim])
        return J


class HomogeneousTransform3D(DifferentiableMap):
    """
    Homeogeneous transformation as DifferentiableMap

    details:
        Takes an angle and rotates the point p0 by this angle

            f(q) = T(q) * p_0

        where T defines a rotation and translation (6DoFs)
            q_{0,1}     => translation
            q_{2}       => rotation

                T = [ R(q)  p(q) ]
                    [ 0 0    1   ]

   We use the Euler angel convention 3-2-1, which is found
   TODO it would be nice to match the ROS convention
   we simply use this one because it was available as derivation
   in termes for sin and cos. It seems that ROS dos
   Static-Z-Y-X so it should be the same. Still needs to test.
    """

    def __init__(self, p0=np.zeros(3)):
        assert p0.size == 3
        self._n = p0.size + 1
        self._T = np.eye(self.input_dimension())
        self._p = np.ones(self.input_dimension())
        self._p[:self.output_dimension()] = p0

    def output_dimension(self):
        return self._n - 1

    def input_dimension(self):
        return self._n

    def point(self):
        return self._p[:self.output_dimension()]

    def forward(self, q):
        assert q.size == self.input_dimension()
        dim = self.output_dimension()
        self._T[:dim, :dim] = rotation_matrix_2d_radian(q[dim])
        self._T[:dim, dim] = q[:dim]
        return np.dot(self._T, self._p)[:dim]
