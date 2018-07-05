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
# Jim Mainprice on Sunday June 17 2017

from test_common_imports import *
from geometry.differentiable_geometry import *


def check_jacobian_against_finite_difference(phi):
    q = np.random.rand(phi.input_dimension())
    J = phi.jacobian(q)
    J_diff = finite_difference_jacobian(phi, q)
    print "J : "
    print J
    print "J_diff : "
    print J_diff
    return check_is_close(J, J_diff, 1e-4)


def test_finite_difference():
    dim = 3
    identity = IdentityMap(dim)
    q = np.random.rand(dim)
    [x, J] = identity.evaluate(q)
    J_eye = np.eye(dim)

    print "----------------------"
    print "Check identity (J implementation) : "
    assert check_jacobian_against_finite_difference(identity)

    print "Check identity (x) 1 : "
    assert check_is_close(q, x)

    print "Check identity (J) 2 : "
    assert check_is_close(J, J_eye)


def test_square_norm():
    x_0 = np.array([1., 2.])
    x_1 = np.array([3., 4.])
    norm = SquaredNorm(x_0)
    [v, J] = norm.evaluate(x_1)
    g = norm.gradient(x_1)
    x_2 = np.zeros(x_1.shape) + g

    print "----------------------"
    print "Check square_norm (J implementation) : "
    assert check_jacobian_against_finite_difference(norm)

    print "v : ", v
    print "x_0 : ", x_0
    print "x_1 : ", x_1
    print "x_1.shape", x_1.shape
    print "x_2.shape : ", x_2.shape 
    print "x_2 : ", x_2
    print "J.shape", J.shape
    print "g.shape", g.shape
    print "zero : ", np.zeros(x_1.shape).shape
    assert np.array_equal(x_2, g)
    print "J : "
    print J
    print "Check square_norm (x) 1 : "
    assert check_is_close(v, 4.)
    # success = check_is_close(J, J_zero)
    # print "Check square_norm (J) 2 : ", success

if __name__ == "__main__":
    test_finite_difference()
    test_square_norm()
