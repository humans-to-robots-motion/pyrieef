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

import numpy as np
from math import pi, cos, sin, atan2


def vectors_angle(v1, v2):
    """
    Computes an angle between two vectors
    the angle is restricted to the range [0, 2pi]

    Parameters
    ----------
    v1 : array like (n, )
        vector 1

    v1 : array like (n, )
        vector 2
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    assert v1.shape == (2,) or v1.shape == (3,)
    assert v1.shape == v2.shape
    return np.mod(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)), 2 * np.pi)


def angle_modulo_2i(theta):
    return theta % (2 * pi)


def angle_from_matrix_2d(matrix):
    """
    Returns the angle from a 2d matrix

    Parameters
    ----------
    matrix : np.array
        Orthogonal matrix
    """
    assert matrix.shape == (2, 2)
    return atan2(matrix[1, 0], matrix[0, 0])


def rotation_matrix_2d(degree):
    """
    Compose a rotation matrix using using a conversion from degrees

    Parameters
    ----------
    degree : float
        angle in degree
    """
    theta = np.radians(degree)
    return rotation_matrix_2d_radian(theta)


def rotation_matrix_2d_radian(theta):
    """
    Compose a rotation matrix using radians

    Parameters
    ----------
    theta : float
        angle of rotation
    """
    c, s = cos(theta), sin(theta)
    return np.array(((c, -s), (s, c)))


def rand_rotation_3d_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    Parameters
    ----------
    deflection : float
        the magnitude of the rotation.
        For 0, no rotation; for 1, competely random rotation.
        Small deflection => small perturbation.
        randnums: 3 random numbers in the range [0, 1]. If `None`,
        they will be auto-generated.
    """
    # copied from blog post
    # http://www.realtimerendering.com/resources/\
    # GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi                   # For direction of pole deflect.
    z = z * 2.0 * deflection                  # For magnitude of pole deflect.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M
