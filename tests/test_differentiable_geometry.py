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

import __init__
from geometry.differentiable_geometry import *


def test_finite_difference():
    dim = 3
    identity = IdentityMap(dim)
    q = np.random.rand(dim)
    [x, J] = identity.evaluate(q)
    J_eye = np.eye(dim)

    print("Check identity (x) 1 : ")
    assert check_is_close(q, x)

    print("Check identity (J) 2 : ")
    assert check_is_close(J, J_eye)

    print("----------------------")
    print("Check identity (J implementation) : ")
    assert check_jacobian_against_finite_difference(identity)


def test_zero():
    dim = 3
    zero = ZeroMap(dim + 3, dim)
    q = np.random.rand(dim)
    [x, J] = zero.evaluate(q)

    print("----------------------")
    print("Check zero (J implementation) : ")
    assert check_jacobian_against_finite_difference(zero)


def test_square_norm():
    x_0 = np.array([1., 2.])
    x_1 = np.array([3., 4.])
    norm = SquaredNorm(x_0)
    [v, J] = norm.evaluate(x_1)
    g = norm.gradient(x_1)
    x_2 = np.zeros(x_1.shape) + g

    print("----------------------")
    print("Check square_norm (J implementation) : ")
    assert check_jacobian_against_finite_difference(norm)

    print("v : ", v)
    print("x_0 : ", x_0)
    print("x_1 : ", x_1)
    print("x_1.shape", x_1.shape)
    print("x_2.shape : ", x_2.shape)
    print("x_2 : ", x_2)
    print("J.shape", J.shape)
    print("g.shape", g.shape)
    print("zero : ", np.zeros(x_1.shape).shape)
    assert np.array_equal(x_2, g)
    print("J : ")
    print(J)
    print("Check square_norm (x) 1 : ")
    assert check_is_close(v, 4.)
    # success = check_is_close(J, J_zero)
    # print "Check square_norm (J) 2 : ", success

    print("----------------------")
    print("Check zero (H implementation) : ")
    assert check_hessian_against_finite_difference(norm)


def test_affine():
    dim = 3
    a = np.random.rand(dim, dim)
    b = np.random.rand(dim)
    f = AffineMap(a, b)

    print("Check AffineMap (J implementation) : ")
    assert check_jacobian_against_finite_difference(f)

    a = np.random.rand(1, dim)
    b = np.random.rand(1)
    f = AffineMap(a, b)

    print("Check AffineMap (function) (J implementation) : ")
    assert check_jacobian_against_finite_difference(f)

    print("Check AffineMap (function) (H implementation) : ")
    assert check_hessian_against_finite_difference(f)


def test_scale():
    dim = 3
    a = np.random.rand(dim, dim)
    b = np.random.rand(dim)
    f = AffineMap(a, b)

    g = Scale(f, .3)

    print("Check Scale (J implementation) : ")
    assert check_jacobian_against_finite_difference(g)


def test_sum_of_terms():
    dim = 8
    f1 = AffineMap(np.random.rand(1, dim), np.random.rand(1))
    f2 = AffineMap(np.random.rand(1, dim), np.random.rand(1))
    f3 = AffineMap(np.random.rand(1, dim), np.random.rand(1))
    sum_of_terms = SumOfTerms([f1, f2, f3])

    print("----------------------")
    print("Check sum of terms (J implementation) : ")
    assert check_jacobian_against_finite_difference(sum_of_terms)

    print("----------------------")
    print("Check sum of terms (H implementation) : ")
    assert check_hessian_against_finite_difference(sum_of_terms)


def test_quadric():
    # Regular case
    dim = 3
    f = QuadricFunction(                # g = x'Ax + b'x + c
        np.random.rand(dim, dim),       # A
        np.random.rand(dim),            # b
        .3)                             # c
    print("Check quadric (J implementation) : ")
    assert check_jacobian_against_finite_difference(f)

    print("Check quadric (H implementation) : ")
    assert check_hessian_against_finite_difference(f)

    # Symetric positive definite case
    k = np.matrix(np.random.rand(dim, dim))
    f = QuadricFunction(                # g = x'Ax + b'x + c
        k.T * k,              # A
        np.random.rand(dim),            # b
        1.)                             # c
    print("Check quadric (J implementation) : ")
    assert check_jacobian_against_finite_difference(f)

    print("Check quadric (H implementation) : ")
    assert check_hessian_against_finite_difference(f)

    print("Check scale quadric (H implementation) : ")
    assert check_hessian_against_finite_difference(Scale(f, .3))


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
    print("Check Composition (J implementation) : ")
    assert check_jacobian_against_finite_difference(Compose(f, g))

    # Test function jacobian (gradient)
    f = SquaredNorm(np.random.rand(dim))
    print("Check Composition (J implementation) : ")
    assert check_jacobian_against_finite_difference(Compose(f, g))

    # Symetric positive definite case
    k = np.matrix(np.random.rand(dim, dim))
    f = QuadricFunction(                # g = x'Ax + b'x + c
        k.T * k,              # A
        np.random.rand(dim),            # b
        1.)                             # c
    print("Check quadric (J implementation) : ")
    assert check_jacobian_against_finite_difference(Compose(f, g))

    dim = 3
    dim_i = 1
    g = AffineMap(                      # g = Ax + b
        np.random.rand(dim_i, dim),     # A
        np.random.rand(dim_i))          # b
    dim_o = 1
    f = AffineMap(                      # f = Ax + b
        np.random.rand(dim_o, dim_i),   # A
        np.random.rand(dim_o))          # b

    print("Check quadric (H implementation) : ")
    assert check_hessian_against_finite_difference(Compose(f, g))


def test_pullback():

    # Test the jacobian and hessian of the following frunction
    # phi : f round g (x) = | Ax + b |^2
    # J_phi = 2Ax + b (TODO verify)
    # H_phi = A'A

    # Test constant Jacobian.
    dim_o = 4
    dim_i = 3
    g = AffineMap(                      # g = Ax + b
        np.random.rand(dim_o, dim_i),   # A
        np.random.rand(dim_o))          # b

    # Test function hessian (gradient)
    f = SquaredNorm(np.zeros(dim_o))

    print("Check Composition (J implementation) : ")
    assert check_jacobian_against_finite_difference(Pullback(f, g))

    # Test function hessian (gradient)
    print("Check Composition (H implementation) : *** ")
    print(f.hessian(np.random.random(dim_o)))
    print(Pullback(f, g).hessian(np.random.random(dim_i)))
    assert check_jacobian_against_finite_difference(Pullback(f, g))


def test_rangesubspace():
    dim = 10
    a = np.random.rand(dim, dim)
    b = np.random.rand(dim)
    f = AffineMap(a, b)
    indices = [1, 3, 7]
    f_sub = Compose(RangeSubspaceMap(f.output_dimension(), indices), f)

    print("Check SubRangeMap (J implementation) : ")
    assert check_jacobian_against_finite_difference(f_sub)


def test_product():

    dim = 3

    g = QuadricFunction(                # g = x'Ax + b'x + c
        np.random.rand(dim, dim),       # A
        np.random.rand(dim),            # b
        .3)                             # c

    h = QuadricFunction(                # g = x'Ax + b'x + c
        np.random.rand(dim, dim),       # A
        np.random.rand(dim),            # b
        .3)                             # c

    assert check_hessian_against_finite_difference(g)
    assert check_hessian_against_finite_difference(h)

    f = ProductFunction(g, h)
    print("Check ProductFunction (J implementation) : ")
    assert check_jacobian_against_finite_difference(f)

    print("Check ProductFunction (H implementation) : ")
    assert check_hessian_against_finite_difference(f)


if __name__ == "__main__":
    test_finite_difference()
    test_zero()
    test_square_norm()
    test_affine()
    test_scale()
    test_sum_of_terms()
    test_quadric()
    test_composition()
    test_pullback()
    test_rangesubspace()
    test_product()
