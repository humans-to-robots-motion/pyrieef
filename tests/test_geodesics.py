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
from utils.misc import *
from geometry.heat_diffusion import *
from geometry.differentiable_geometry import PolynomeTestFunction
from numpy.testing import assert_allclose


def test_matrix_coordinates():
    DIM = 4
    for i in range(100):
        [x, y] = row_major(i, DIM)
        assert i == (x + y * DIM)


def test_gradient_1d_operator_linear():
    """
    TODO test with a linear map

         f(x) = ax
    d/dx f(x) = a

    """

    #
    a = np.random.random((1, 2))
    b = np.zeros((1, ))
    f = AffineMap(a, b)

    # define the gradient
    N = 10
    D = -float(N) * discrete_2d_gradient(N, N, axis=1)
    x = y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    # 3 dimensional array with both x and y values at each grid point
    xxyy = np.concatenate(
        [np.expand_dims(X, axis=2),
         np.expand_dims(Y, axis=2)], axis=2)

    # Same thing directly with stack
    Q = np.stack((X, Y), axis=2)

    print("Q1 shape : ", xxyy.shape)
    print("Q2 shape : ", Q.shape)

    assert_allclose(Q, xxyy)

    f_vect = np.vectorize(f.forward, signature='(2)->(1)')
    g_vect = np.vectorize(f.gradient, signature='(2)->(2)')
    h_vect = np.vectorize(f.hessian, signature='(2)->(2,2)')

    Z = f_vect(Q)
    G = g_vect(Q)
    H = h_vect(Q)

    # print(Z)
    # print(G)
    # print(H)

    print(Z.shape)
    print(G.shape)
    print(H.shape)

    # print(f_vect)
    # print(D.shape)
    # print(np.gradient(Z, axis=0).shape)
    # print(J.shape)

    # grad1 = np.dot(D, Z.flatten())
    # grad2 = np.gradient(Z, 1/float(N), axis=0).flatten()
    # grad3 = J.flatten()
    # print("len(grad1) : ", len(grad1))
    # print("len(grad2) : ", len(grad2))
    # print("len(grad3) : ", len(grad3))
    # print(grad1)
    # print(grad2)
    # print(grad3)
    # d_g = np.linalg.norm(grad1 - grad2)
    # print(d_g)
    # assert d_g < 2


def test_gradient_1d_operator():
    """
    Start testing gradient operator
    It seems that the conventions for gradients between numpy and pyrieef
    are not the same. 

        1) the sign has to be flipped
        2) the axis has to be flipped

    TODO:   The sign should be checked and fixed in pyrieef, this is probably
            a mistake as the gradient should always point to the direction
            of function increase and not decrease. The axis is more of a 
            convention thing and we should probably have it match numpy.
    """

    N = 10
    D = -float(N) * discrete_2d_gradient(N, N, axis=1)

    f = PolynomeTestFunction()
    x = y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    Z = f(np.stack([X, Y]))
    J = f.jacobian_x(np.stack([X, Y]))
    print(Z.shape)
    print(D.shape)
    print(np.gradient(Z, axis=0).shape)
    print(J.shape)

    grad1 = np.dot(D, Z.flatten())
    grad2 = np.gradient(Z, 1/float(N), axis=0).flatten()
    grad3 = J.flatten()
    print("len(grad1) : ", len(grad1))
    print("len(grad2) : ", len(grad2))
    print("len(grad3) : ", len(grad3))
    print(grad1)
    print(grad2)
    print(grad3)
    d_g = np.linalg.norm(grad1 - grad2)
    print(d_g)
    assert d_g < 2


def test_gradient_operator():
    """
    TODO test the function here..
    finish this
    """

    N = 10
    Dx = discrete_2d_gradient(N, N, axis=0)
    # Dy = discrete_2d_gradient(N, N, axis=1)
    D = np.vstack([Dx, Dy])

    f = PolynomeTestFunction()
    x = y = np.linspace(0, 3, N)
    X, Y = np.meshgrid(x, y)
    Z = f(np.stack([X, Y]))
    print(Z.shape)
    print(D.shape)
    print(np.gradient(Z, axis=0).shape)
    print(np.gradient(Z, axis=1).shape)
    grad1 = np.dot(D, Z.flatten())
    grad2 = np.stack([
        np.gradient(Z, axis=0).flatten(),
        np.gradient(Z, axis=1).flatten()]).flatten()
    print("len(grad1) : ", len(grad1))
    print("len(grad2) : ", len(grad2))
    print(grad1)
    print(grad2)
    d_g = np.linalg.norm(grad1 - grad2)
    # assert d_g < 1e-10


def test_distance_from_gradient():
    N = 20
    workspace = Workspace()
    workspace.obstacles = [Circle(origin=[-2, -2], radius=0.1)]
    Dx = float(N) * discrete_2d_gradient(N, N, axis=0)
    Dy = float(N) * discrete_2d_gradient(N, N, axis=1)
    D = np.vstack([Dx, Dy])
    f = sdf(occupancy_map(N, workspace)).T
    grad = np.dot(D, f.flatten())
    phi = np.dot(np.linalg.pinv(D), grad)
    d_g = np.linalg.norm(grad - np.dot(D, phi))
    print("d_g : ", d_g)
    assert d_g < 1e-10

    phi.shape = (N, N)
    phi -= phi.min()
    f -= f.min()
    d_d = np.linalg.norm(phi - f)
    print("d_d : ", d_d)
    assert d_d < 1e-10


if __name__ == "__main__":
    # test_matrix_coordinates()
    test_gradient_1d_operator_linear()
    # test_gradient_1d_operator()
    # test_gradient_operator()
    # test_distance_from_gradient()
