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
from geometry.pixel_map import *


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


def test_affine():
    dim = 3
    a = np.random.rand(dim, dim)
    b = np.random.rand(dim)
    f = AffineMap(a, b)

    print "Check AffineMap (J implementation) : "
    assert check_jacobian_against_finite_difference(f)


def test_regressed_grid():

    l = 0.5

    # Regularly-spaced, coarse grid
    ds1 = 0.1
    x = np.arange(-l, l, ds1)
    y = np.arange(-l, l, ds1)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(2 * X)**2 - (Y / 2)**2)

    # spline with grid data
    interp_spline = RectBivariateSpline(x, y, Z.transpose())

    # same but with DifferentiableMap structure
    f = RegressedPixelGridSpline(Z.transpose(), ds1, Extends(l))

    # Regularly-spaced, fine grid

    ds2 = 0.05
    x2 = np.arange(-l, l, ds2)
    y2 = np.arange(-l, l, ds2)
    Z2 = interp_spline(x2, y2)
    g2_x = interp_spline(x2, y2, dx=1)  # Gradient x
    g2_y = interp_spline(x2, y2, dy=1)  # Gradient y

    # Function interpolation
    g1_x = np.zeros((x2.size, y2.size))
    g1_y = np.zeros((x2.size, y2.size))
    z1 = np.zeros((x2.size, y2.size))
    print "g1 : ", g1_x.shape
    for i, x in enumerate(x2):
        for j, y in enumerate(y2):
            p = np.array([x, y])
            z1[i, j] = f.forward(p)
            grad = f.gradient(p)
            g1_x[i, j] = grad[0]  # Gradient x
            g1_y[i, j] = grad[1]  # Gradient y

    print g1_x.shape

    assert check_is_close(Z2, z1, 1e-10)
    assert check_is_close(g2_x, g1_x, 1e-10)
    assert check_is_close(g2_y, g1_y, 1e-10)


def test_quadric():
    # Regular case
    dim = 3
    f = QuadricFunction(                # g = x'Ax + b'x + c
        np.random.rand(dim, dim),       # A
        np.random.rand(dim),            # b
        .3)                             # c
    print "Check quadric (J implementation) : "
    assert check_jacobian_against_finite_difference(f)

    # Symetric positive definite case
    k = np.matrix(np.random.rand(dim, dim))
    f = QuadricFunction(                # g = x'Ax + b'x + c
        k.transpose() * k,              # A
        np.random.rand(dim),            # b
        1.)                             # c
    print "Check quadric (J implementation) : "
    assert check_jacobian_against_finite_difference(f)


def test_composition():

    # Test constant Jacobian.
    dim = 3
    g = AffineMap(                      # g = Ax + b
        np.random.rand(dim, dim),       # A
        np.random.rand(dim))            # b
    dim_o = 4
    dim_i = 3
    f = AffineMap(                      # f = Ax + b
        np.random.rand(dim_o, dim_i),   # A
        np.random.rand(dim_o))          # b
    print "Check Composition (J implementation) : "
    assert check_jacobian_against_finite_difference(Compose(f, g))

    # Test function jacobian (gradient)
    f = SquaredNorm(np.random.rand(dim))
    print "Check Composition (J implementation) : "
    assert check_jacobian_against_finite_difference(Compose(f, g))

    # Symetric positive definite case
    k = np.matrix(np.random.rand(dim, dim))
    f = QuadricFunction(                # g = x'Ax + b'x + c
        k.transpose() * k,              # A
        np.random.rand(dim),            # b
        1.)                             # c
    print "Check quadric (J implementation) : "
    assert check_jacobian_against_finite_difference(Compose(f, g))


def test_rangesubspace():
    dim = 10
    a = np.random.rand(dim, dim)
    b = np.random.rand(dim)
    f = AffineMap(a, b)
    indices = [1, 3, 7]
    f_sub = Compose(RangeSubspaceMap(f.output_dimension(), indices), f)

    print "Check SubRangeMap (J implementation) : "
    assert check_jacobian_against_finite_difference(f_sub)

if __name__ == "__main__":
    test_finite_difference()
    test_square_norm()
    test_affine()
    test_regressed_grid()
    test_quadric()
    test_composition()
    test_rangesubspace()
