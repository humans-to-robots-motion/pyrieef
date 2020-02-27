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
#                                    Jim Mainprice on Thursday January 23 2020

import numpy as np
from .workspace import *
from .pixel_map import *
from utils.misc import *
import itertools
from functools import reduce
# import scipy

NB_POINTS = 20
VERBOSE = False
ALGORITHM = "forward"
# ALGORITHM = "crank-nicholson"
TIME_FACTOR = 10
# TIME_STEP =  .0002  # (dx ** 2) (ideal)
TIME_STEP = 2e-5
CONSTANT_SOURCE = False
VECTORIZED = True


def kernel(t, d, dim=2):
    return np.exp(-(d**2) / (4 * t)) / pow(4 * np.pi * t, .5 * dim)


def compare_with_kernel(u_t, t, workspace):
    grid = workspace.pixel_map(NB_POINTS)
    u_e = np.zeros(u_t.shape)
    for i, j in itertools.product(range(u_e.shape[0]), range(u_e.shape[1])):
        p = grid.grid_to_world(np.array([i, j]))
        u_e[i, j] = kernel(t, np.linalg.norm(p))
    u_e /= u_e.max()
    u_t /= u_t.max()
    error = abs(u_e - u_t).max()
    print(" -- diff with kernel : abs {}, max {}, min {}".format(
        error, (u_e - u_t).max(), (u_e - u_t).min()))
    print(" -- shape u_t : ", u_t.shape)
    print(" -- shape u_e : ", u_e.shape)
    print(" -- error : ", error)
    assert error < 0.01   # Error is smaller that 1%
    return u_e


def forward_euler_2d(dt, h, source_grid, iterations, occupancy):
    """
    Forward Euler Integration of the heat equation

        h : space discretization
        t : time discretization
    """
    U = []
    t = 0.
    dh2 = h ** 2
    # dt = dh2 * dh2 / (2 * (dh2 + dh2))
    Zero = np.zeros((NB_POINTS, NB_POINTS))
    u_t = Zero.copy()
    u_0 = Zero.copy()
    d = 1. / dh2
    u_0[source_grid[0], source_grid[1]] = 1.e4
    u_0 = np.where(occupancy.T > 0, Zero, u_0)
    for k in range(iterations * TIME_FACTOR):
        if CONSTANT_SOURCE:
            u_0[source_grid[0], source_grid[1]] = 1.e4
        # Propagate with forward-difference in time
        # central-difference in space
        if VECTORIZED:
            u_t[1:-1, 1:-1] = u_0[1:-1, 1:-1] + dt * d * (
                (u_0[2:, 1:-1] - 2 * u_0[1:-1, 1:-1] + u_0[:-2, 1:-1]) +
                (u_0[1:-1, 2:] - 2 * u_0[1:-1, 1:-1] + u_0[1:-1, :-2]))
        else:
            for i, j in itertools.product(
                    range(1, NB_POINTS - 1), range(1, NB_POINTS - 1)):
                u_t[i, j] = u_0[i, j] + dt * d * (
                    - 4 * u_0[i, j] +
                    (u_0[i + 1, j] + u_0[i - 1, j]) +
                    (u_0[i, j + 1] + u_0[i, j - 1]))

        u_t = np.where(occupancy.T > 0, Zero, u_t)
        u_0 = u_t.copy()
        t += dt
        if k % TIME_FACTOR == 0:
            print("t : {:.3E} , u_t.max() : {:.3E}".format(t, u_t.max()))
            U.append(u_t.copy())
    return U


def crank_nicholson_2d(dt, h, source_grid, iterations, occupancy):
    """
    Crank-Nicholson algorithm with matrix inversion
    we use a row major representation of the matrix

        h : space discretization
        t : time discretization

    U(i,j,m+1) = U(i,j,m) + k*Discrete-2D-Laplacian(U)(i,j,m)
                          k
               = (1 - 4* ---) * U(i,j,m) +
                         h^2
                   k
                  --- * (U(i-1,j,m) + U(i+1,j,m) + U(i,j-1,m) + U(i,j+1,m))
                  h^2
    """
    d = 1. / (h ** 2)
    dim = NB_POINTS ** 2
    M = np.zeros((dim, dim))

    a = 2. * dt * d
    c = - dt * d

    if VERBOSE:
        print("a : ", a)
        print("c : ", c)
    print("fill matrix...")

    for p, q in itertools.product(range(dim), range(dim)):
        i0, j0 = row_major(p, NB_POINTS)
        i1, j1 = row_major(q, NB_POINTS)
        if p == q:
            M[p, q] = a
        elif (
                i0 == i1 - 1) and (j0 == j1) or (
                i0 == i1 + 1) and (j0 == j1) or (
                i0 == i1) and (j0 == j1 - 1) or (
                i0 == i1) and (j0 == j1 + 1):
            M[p, q] = c
        if occupancy[i0, j0] == 1. or occupancy[i1, j1] == 1.:
            M[p, q] = 0.
    if VERBOSE:
        with np.printoptions(
                formatter={'float': '{: 0.1f}'.format},
                linewidth=200):
            print("M : \n", M)
    u_0 = np.zeros((dim))
    u_0[source_grid[0] + source_grid[1] * NB_POINTS] = 1.
    if VERBOSE:
        print(" - I.shape : ", I.shape)
        print(" - M.shape : ", M.shape)
        print(" - u_0.shape : ", u_0.shape)
    print("solve...")
    costs = []
    u_t = u_0
    for i in range(iterations * 10):
        for j in range(u_t.size):
            i0, j0 = row_major(j, NB_POINTS)
            if (i0 == 0 or i0 == NB_POINTS - 1 or
                    j0 == 0 or j0 == NB_POINTS - 1):
                u_t[j] = 0
            elif occupancy[i0, j0] == 1.:
                u_t[j] = 0
        u_t = (np.eye(dim) - M).dot(u_t)
        u_t = np.linalg.solve(np.eye(dim) + M, u_t)
        if i % 10 == 0:
            print(u_t.max())
            costs.append(np.reshape(u_t, (-1, NB_POINTS)).copy())
    print("solved!")
    return costs


def discrete_2d_gradient(M, N, axis=0):
    """
    Efficient allocation of the Discrete-2D-Gradient
    """
    dim = M * N
    if axis == 0:
        A = np.identity(dim)
        A[range(dim - 1), range(1, dim)] = -1
        A[range(dim - 1), range(1, dim)] = -1

        # every end of row left difference
        A[range(M - 1, M * (N - 1), M), range(M, dim, M)] = 0
        A[range(M - 1, dim, M), range(M - 1, dim, M)] = -1
        A[range(M - 1, dim, M), range(M - 2, dim, M)] = 1

    if axis == 1:
        A = np.identity(dim)
        A[range(M * (N - 1)), range(M, dim)] = -1

        # every end of collumn left difference
        last_block = range(M * (N - 1), dim)
        A[last_block, last_block] = -1
        A[last_block, range(M * (N - 2), M * (N - 1))] = 1

    return A


def discrete_2d_laplacian(M, N, matrix_form=False):
    """
    Efficient allocation of the Discrete-2D-Laplacian

    TODOs
        1) change allocation in crank_nicholson_2d
        2) can do better with range instead of for loop!

    """
    A = np.zeros((M * N, M * N))
    if matrix_form:
        diagonal = np.ones(M)
        Id = np.diag(-1 * diagonal)
        D = np.diag(4 * diagonal)
        D[range(1, M), range(M - 1)] = -1
        D[range(M - 1), range(1, M)] = -1
        for i in range(N):
            A[i * M:(i + 1) * M, i * M:(i + 1) * M] = D
            if i > 0:
                A[(i - 1) * M:i * M, i * M:(i + 1) * M] = Id
            if i < N - 1:
                A[(i + 1) * M:(i + 2) * M, i * M:(i + 1) * M] = Id
        return A
    else:
        # This actually seems wrong...
        for p, q in itertools.product(range(A.shape[0]), range(A.shape[1])):
            i0, j0 = row_major(p, A.shape[0])
            i1, j1 = row_major(q, A.shape[0])
            if p == q:
                A[p, q] = 4
            elif (
                    i0 == i1 - 1) and (j0 == j1) or (
                    i0 == i1 + 1) and (j0 == j1) or (
                    i0 == i1) and (j0 == j1 - 1) or (
                    i0 == i1) and (j0 == j1 + 1):
                A[p, q] = -1
        return A


def normalized_gradient(field):
    """
    Compute the discrete normalized gradient from scalar field
    """
    gradient = np.gradient(field)
    norms = np.linalg.norm(gradient, axis=0)
    gradient = np.array([np.where(norms == 0, 0, i / norms) for i in gradient])
    return gradient


def divergence(f):
    """
    Compute the discrete divergence of a vector field
    """
    # return reduce(np.add, gradient)
    return np.ufunc.reduce(
        np.add,
        [np.gradient(f[i], axis=i) for i in range(len(f))])


def distance_from_gradient(U, V, dh):
    N = U.shape[0]
    Dx = (1. / dh) * discrete_2d_gradient(N, N, axis=0)
    Dy = (1. / dh) * discrete_2d_gradient(N, N, axis=1)
    grad = np.hstack([U.flatten(), V.flatten()])
    D = np.vstack([Dx, Dy])
    phi = np.dot(np.linalg.pinv(D), grad)
    phi.shape = (N, N)
    return phi


def poisson_equation(D, dh):
    """
    Find the distance of a gradient on a 2d grid

        https://en.wikipedia.org/wiki/Discrete_Poisson_equation

        D : divergence

        we use a row major convention
        A = [a11, a12, a13, a21, a22, a23, ..., aMN]
    """
    # gradient = -1. * np.array(gradient)
    M = D.shape[0]
    N = D.shape[1]
    A = (1. / (dh**2)) * discrete_2d_laplacian(M, N, True)
    D = D.flatten()
    A_inv = np.linalg.inv(A)
    phi = np.dot(A_inv, D)
    phi -= phi.min()  # solve by setting minimum to 0
    phi.shape = (M, N)
    return phi


def distance(U, V, D, dh):
    """
    Find the distance of a gradient on a 2d grid

        [U, V] : gradient is given as two matrices
        D : divergence

        we use a row major convention
        A = [a11, a12, a13, a21, a22, a23, ..., aMN]

        # D = divergence(gradient)
        # Dc = D.copy()
        # D[source[1], source[0]] = 1000
        # D[np.where(occupancy.T > 0)] = np.Inf
        D = D.flatten()
        # idxs_o = np.where(D == np.Inf)
        # idxs_i = np.where(D < np.Inf)
        # A = np.delete(A, idxs_o, axis=0)
        # A = np.delete(A, idxs_o, axis=1)
        # D = np.delete(D, idxs_o)
        # phi = D
        # phi -= phi.min()
        # dist = np.empty(M * N)
        # for i in range(len(idxs_i[0])):
        #     dist[idxs_i[0][i]] = phi[i]
        # dist[idxs_o] = 10
    """
    M = D.shape[0]
    N = D.shape[1]
    Dx = (1. / dh) * discrete_2d_gradient(N, N, axis=0)
    Dy = (1. / dh) * discrete_2d_gradient(N, N, axis=1)

    # D2xy = (1. / (dh**2)) * discrete_2d_laplacian(M, N, True)
    # b = np.hstack([U.flatten(), V.flatten(), D.flatten()])
    # A = np.vstack([Dx, Dy, D2xy])

    b = np.hstack([U.flatten(), V.flatten()])
    A = np.vstack([Dx, Dy])

    print("inverse matrix...")
    A_inv = np.linalg.pinv(A)

    phi = np.dot(A_inv, b)
    phi -= phi.min()

    d = np.linalg.norm(np.dot(Dx, phi) - U.flatten())
    print("d1 = ", d)

    d = np.linalg.norm(np.dot(Dy, phi) - V.flatten())
    print("d2 = ", d)

    phi.shape = (M, N)
    return phi


def heat_diffusion(workspace, source, iterations):
    """
    Diffuses heat from a source point on a 2D grid defined
    over a workspace populated by obstacles.

        The function was implemented by following
        https://people.eecs.berkeley.edu/~demmel/\
            cs267/lecture17/lecture17.html#link_1.5

    TODO test it agains the heat kernel
    """
    grid = workspace.pixel_map(NB_POINTS)
    h = grid.resolution
    occupancy = occupancy_map(NB_POINTS, workspace).T
    print("Max t size : ", (h ** 2))
    t = TIME_STEP
    print(" -- h : ", h)
    print(" -- t : ", t)
    source_grid = grid.world_to_grid(source)
    if ALGORITHM == "crank-nicholson":
        return crank_nicholson_2d(t, h, source_grid, iterations, occupancy)
    else:
        return forward_euler_2d(
            t, h, source_grid, iterations, occupancy)
