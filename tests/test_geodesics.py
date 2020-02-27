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

from __init__ import *
from utils.misc import *
from geometry.heat_diffusion import *


def test_matrix_coordinates():
    DIM = 4
    for i in range(100):
        [x, y] = row_major(i, DIM)
        assert i == (x + y * DIM)


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
    d = np.linalg.norm(grad - np.dot(D, phi))
    print("d : ", d)
    assert d < 1e-10


if __name__ == "__main__":
    test_matrix_coordinates()
    test_distance_from_gradient()
