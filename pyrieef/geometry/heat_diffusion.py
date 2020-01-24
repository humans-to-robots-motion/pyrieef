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
# import scipy

NB_POINTS = 20
VERBOSE = False
ALGORITHM = "forward"
# ALGORITHM = "crank-nicholson"
TIME_FACTOR = 10
# TIME_STEP =  .0002  # (dx ** 2) (ideal)
TIME_STEP = 2e-5
CONSTANT_SOURCE = False


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
    for i in range(iterations * TIME_FACTOR):
        if CONSTANT_SOURCE:
            u_0[source_grid[0], source_grid[1]] = 1.e4
        # Propagate with forward-difference in time
        # central-difference in space
        u_t[1:-1, 1:-1] = u_0[1:-1, 1:-1] + dt * d * (
            (u_0[2:, 1:-1] - 2 * u_0[1:-1, 1:-1] + u_0[:-2, 1:-1]) +
            (u_0[1:-1, 2:] - 2 * u_0[1:-1, 1:-1] + u_0[1:-1, :-2]))
        u_t = np.where(occupancy.T > 0, Zero, u_t)
        u_0 = u_t.copy()
        t += dt
        if i % TIME_FACTOR == 0:
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
               = (1 - 4*---) * U(i,j,m) +
                         h^2
                   k
                  ---*(U(i-1,j,m) + U(i+1,j,m) + U(i,j-1,m) + U(i,j+1,m))
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
