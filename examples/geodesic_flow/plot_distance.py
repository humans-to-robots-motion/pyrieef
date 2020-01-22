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

from demos_common_imports import *
import numpy as np
from pyrieef.geometry.workspace import *
from pyrieef.geometry.charge_simulation import *
from pyrieef.geometry.pixel_map import *
from pyrieef.geometry.geodesics import *
from pyrieef.geometry.diffeomorphisms import *
from pyrieef.geometry.utils import *
from pyrieef.rendering.workspace_renderer import WorkspaceDrawer
from pyrieef.utils.misc import *
import itertools
import matplotlib.pyplot as plt


NB_POINTS = 20


def heat_diffusion(workspace, source):
    # sdf = SignedDistanceWorkspaceMap(workspace)
    extent = workspace.box.extent()
    dx = (extent.x_max - extent.x_min) / NB_POINTS
    dy = (extent.y_max - extent.y_min) / NB_POINTS
    X, Y = workspace.box.meshgrid(NB_POINTS)
    assert dx == dy
    dim = NB_POINTS ** 2
    M = np.zeros((dim, dim))
    I = np.eye(dim)
    k = 1000.
    h = 1. / NB_POINTS
    d = 1. / (h ** 2)
    # c = k * d
    # a = 4. * c
    c = -k / (2. * (h ** 2))
    a = 2. * k / (h ** 2)
    # we use a row major representation of the matrix
    for p, q in itertools.product(range(dim), range(dim)):
        i0, j0 = row_major(p, NB_POINTS)
        i1, j1 = row_major(q, NB_POINTS)
        if p == q:
            M[p, q] = a
        elif (i0 == j0 - 1 or i0 + 1 == j0 or i0 == j0 - 1 or i0 == j0 + 1 or
              i1 == j1 - 1 or i1 + 1 == j1 or i1 == j1 - 1 or i1 == j1 + 1):
            # p0 = np.array([X[i0, j0], Y[i0, j0]])
            # p1 = np.array([X[i1, j1], Y[i1, j1]])
            M[p, q] = c
    u_0 = np.zeros((dim))
    u_0[source[0] + source[1] * NB_POINTS] = 1.
    print(u_0)
    print("solve...")
    print(" - I.shape : ", I.shape)
    print(" - M.shape : ", M.shape)
    print(" - u_0.shape : ", u_0.shape)
    costs = []
    u_t = u_0
    for i in range(1):
        print("solve : ", i)
        # u_t = (I + M).dot(u_t)
        u_t = np.linalg.solve(I + M, u_t)
        print(u_t)
        costs.append(np.reshape(u_t, (-1, NB_POINTS)).copy())
    print("solved!")
    print(" - u_t.shape : ", u_t.shape)
    # print(" - cost.shape : ", cost.shape)
    return costs


cmap = plt.get_cmap('viridis')

circles = []
circles.append(AnalyticCircle(origin=[.1, .0], radius=0.1))
circles.append(AnalyticCircle(origin=[.1, .25], radius=0.05))
circles.append(AnalyticCircle(origin=[.2, .25], radius=0.05))
circles.append(AnalyticCircle(origin=[.0, .25], radius=0.05))

workspace = Workspace()
workspace.obstacles = [circle.object() for circle in circles]
renderer = WorkspaceDrawer(workspace)

x_goal = np.array([0.4, 0.4])
nx, ny = (5, 4)
x = np.linspace(-.2, -.05, nx)
y = np.linspace(-.5, -.1, ny)

analytical_circles = AnalyticMultiDiffeo(circles)

U = heat_diffusion(workspace, [7, 7])

sclar_color = 0.
for i, j in itertools.product(list(range(nx)), list(range(ny))):
    sclar_color += 1. / (nx * ny)
    x_init = np.array([x[i], y[j]])
    print("x_init : ", x_init)

    # Does not have an inverse.
    # [line, line_inter] = InterpolationGeodescis(
    #     analytical_circles, x_init, x_goal)

    # line = NaturalGradientGeodescis(analytical_circles, x_init, x_goal)
    # renderer.draw_ws_line(line, color=cmap(sclar_color))
    # renderer.draw_ws_point([x_init[0], x_init[1]], color='r', shape='o')
sdf = SignedDistanceWorkspaceMap(workspace)
for i in range(1):
    renderer.set_drawing_axis(i)
    renderer.draw_ws_obstacles()
    renderer.draw_ws_point([x_goal[0], x_goal[1]], color='r', shape='o')
    renderer.background_matrix_eval = False
    # renderer.draw_ws_background(sdf, nb_points=200)
    renderer.draw_ws_img(U[i])
renderer.show()
