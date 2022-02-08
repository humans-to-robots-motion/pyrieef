#!/usr/bin/env python

# Copyright (c) 2018
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

# Externals
from scipy.linalg import cholesky_banded, solveh_banded
from numpy import zeros, diag


def test_matrix_coordinates():
    DIM = 4
    for i in range(100):
        [x, y] = row_major(i, DIM)
        assert i == (x + y * DIM)


def test_gradient_1d_operator_linear():
    """
    Test Finite Difference (FD) gradients against linear map

    Details:
                 f(x) = ax
            d/dx f(x) = a

        Linear maps are perfectly approximated by finite difference
        as no higher-order curvature terms are involved
        That allows to test an FD implementation.
    """

    np.random.seed(0)

    # Linear map with random coefficients
    a = np.random.random((1, 2))
    b = np.zeros((1, ))
    f = AffineMap(a, b)

    # define the grid points
    N = 10
    l = 1.
    x = y = np.linspace(0, l, N)
    X, Y = np.meshgrid(x, y)

    # Two ways of creating the data for vectorized querry
    # TODO test for speed etc.

    # 1) with expand dims
    # 3 dimensional array with both x and y values at each grid point
    xxyy = np.concatenate(
        [np.expand_dims(X, axis=2),
         np.expand_dims(Y, axis=2)], axis=2)

    # 2) simpler with stack
    # Same thing directly with stack
    Q = np.stack((X, Y), axis=2)

    assert_allclose(Q, xxyy)

    # define vectorize functions
    f_vect = np.vectorize(f.forward, signature='(2)->(1)')
    g_vect = np.vectorize(f.gradient, signature='(2)->(2)')
    h_vect = np.vectorize(f.hessian, signature='(2)->(2,2)')

    Z = f_vect(Q)
    G = g_vect(Q)

    # Test vectorized querry
    G2 = np.empty_like(G)
    for i, j in product(range(X.shape[0]), range(X.shape[0])):
        g = f.gradient(np.array([X[i, j], Y[i, j]]))
        G2[i, j, 0] = g[0]
        G2[i, j, 1] = g[1]

    assert_allclose(G, G2)

    # dx is the distance between the grid samples np.gradient can handle
    # varying grid sizes, discrete_2d_gradient can not do that so far
    # WARNING: the distance between N equaly spaced samples along l is not l/N
    dx = l/float(N-1)

    # Switch between axis diff and dim dimension to easily test equality
    G = np.flip(G, axis=2)

    for a in range(2):

        D = discrete_2d_gradient(N, N, dx=dx, axis=a)

        grad1 = np.dot(D, Z.flatten())
        grad2 = np.gradient(Z, dx, axis=a)
        grad3 = G[:, :, a]

        assert_allclose(grad1, grad2.flatten())
        assert_allclose(grad1, grad3.flatten())


def test_gradient_1d_operator():
    """
    Start testing gradient operator

    TODO: Make sure that it works over the whole code based !!!

    Following is not Fixed !

        It seems that the conventions for gradients between numpy and pyrieef
        are not the same. 

            1) the sign has to be flipped
            2) the axis has to be flipped

        TODO:   The sign should be checked and fixed in pyrieef,
            this is probably a mistake as the gradient should always point
            to the direction
            of function increase and not decrease. The axis is more of a 
            convention thing and we should probably have it match numpy.
    """
    N = 10
    l = 1e-6
    dx = l/(N-1)

    f = PolynomeTestFunction()
    x = y = np.linspace(0, l, N)
    X, Y = np.meshgrid(x, y)
    Z = f(np.stack([X, Y]))
    J = f.jacobian_y(np.stack([X, Y]))

    # inversion of differentiating aling and axis and the axis in the matrix
    jacobian = [
        f.jacobian_y(np.stack([X, Y])),
        f.jacobian_x(np.stack([X, Y]))]

    for a in range(2):

        D = discrete_2d_gradient(N, N, dx, axis=a)

        grad1 = np.dot(D, Z.flatten())
        grad2 = np.gradient(Z, dx, axis=a)
        grad3 = jacobian[a]

        assert_allclose(grad1, grad2.flatten())
        assert_allclose(grad1, grad3.flatten())


def test_gradient_operator():
    """
    This function tests the linear gradient operator

    Note: that we have small deviations from the numpy implementations
          We should check that these deviations come from the boundry
          conditions. The numpy version has some different things happening.
    """

    N = 10

    x = y = np.linspace(0, 1, N)
    Q = np.stack((np.meshgrid(x, y)))
    Z = PolynomeTestFunction().forward(Q)
    # Z = LinearTestFunction().forward(Q)

    grad1 = np.dot(np.vstack([
        discrete_2d_gradient(N, N, axis=0),
        discrete_2d_gradient(N, N, axis=1)]), Z.flatten())

    grad2 = np.stack([
        np.gradient(Z, axis=0).flatten(),
        np.gradient(Z, axis=1).flatten()]).flatten()

    assert_allclose(grad1, grad2, atol=2e-2)


def test_distance_from_gradient():
    """
    Compare the distance obtained from pseudo inverse of the gradient
    operator with the original one
    """

    N = 20

    # Signed distance field
    workspace = Workspace()
    workspace.obstacles = [Circle(origin=[-2, -2], radius=0.1)]
    f = sdf(occupancy_map(N, workspace)).T

    # Gradient operator
    dl = 1/float(N-1)
    Dx = discrete_2d_gradient(N, N, dx=dl, axis=0)
    Dy = discrete_2d_gradient(N, N, dx=dl, axis=1)
    D = np.vstack([Dx, Dy])

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


def test_2d_laplacian():
    n = 15
    u = np.random.random((n, n))
    h = .1
    M = discrete_2d_laplacian(n, n, matrix_form=True)
    v1 = -(1/(h ** 2)) * M @ u.flatten()
    v2 = finite_difference_laplacian_2d(h, u).flatten()

    with np.printoptions(
            formatter={'float': '{:6.1f}'.format},
            linewidth=200):
        print(v1.reshape((n, n)))
        print(v2.reshape((n, n)))

    # TODO figure out why the inner part does not fit.
    assert_allclose(
        v1.reshape((n, n))[1:n-1, 1:n-1],
        v2.reshape((n, n))[1:n-1, 1:n-1])


def test_2d_laplacian_solve():

    verbose = True

    n = 5
    M = discrete_2d_laplacian(n, n, matrix_form=True)

    # convert to banded format
    N = M.shape[0]                              # num of rows in A
    D = np.amax(np.nonzero(M[0, :])) + 1        # num of nonzeros in first row
    ab = np.zeros((D, N))                       # upper triangular structure

    # get all diagonals up to D
    for i in np.arange(1, D):
        ab[i, :] = np.concatenate((np.diag(M, k=i), np.zeros(i,)), axis=None)

    # get main diangonal (np.diag(M, k=0))
    ab[0, :] = np.diagonal(M)

    # todo get the choleski decomposition based on ab
    # c = cholesky_banded(ab)

    u = np.ones(n ** 2)

    print("solve v1..")
    v1 = np.linalg.inv(M) @ u

    print("solve v2..")
    v2 = solveh_banded(ab, u, lower=True)

    print("done.")

    if verbose:
        with np.printoptions(
                formatter={'float': '{:6.1f}'.format},
                linewidth=200):
            print("M \n {}".format(M))
            print("ab \n {}".format(ab))
            print(v1.reshape((n, n)))
            print(v2.reshape((n, n)))

    assert_allclose(v1, v2)


if __name__ == "__main__":
    # test_matrix_coordinates()
    # test_gradient_1d_operator_linear()
    # test_gradient_1d_operator()
    # test_gradient_operator()
    # test_distance_from_gradient()
    # test_2d_laplacian()
    test_2d_laplacian_solve()
