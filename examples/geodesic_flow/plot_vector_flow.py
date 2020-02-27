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
#                                         Jim Mainprice on Wed January 22 2020

from demos_common_imports import *
import numpy as np
import pyrieef.geometry.heat_diffusion as hd
from pyrieef.geometry.workspace import *
from pyrieef.rendering.workspace_renderer import WorkspaceDrawer
import matplotlib.pyplot as plt
import itertools


ROWS = 1
COLS = 1

hd.NB_POINTS = 101
hd.TIME_FACTOR = 200
hd.TIME_STEP = 2e-5
hd.ALGORITHM = "forward"
hd.CONSTANT_SOURCE = True
N = 40

circles = []
circles.append(Circle(origin=[.1, .0], radius=0.1))
circles.append(Circle(origin=[.1, .25], radius=0.05))
circles.append(Circle(origin=[.2, .25], radius=0.05))
circles.append(Circle(origin=[.0, .25], radius=0.05))

workspace = Workspace()
workspace.obstacles = circles
renderer = WorkspaceDrawer(workspace, rows=ROWS, cols=COLS)
x_source = np.array([0.21666667, 0.15])

# ------------------------------------------------------------------------------
# iterations = ROWS * COLS
iterations = 9
u_t = hd.heat_diffusion(workspace, x_source, iterations)
grid = workspace.pixel_map(hd.NB_POINTS)
X, Y = workspace.box.meshgrid(N)
U, V = np.zeros(X.shape), np.zeros(Y.shape)

print(u_t[-1].shape)
print(X.shape)
print(Y.shape)
phi = np.empty(X.shape)
f = RegressedPixelGridSpline(u_t[-1], grid.resolution, grid.extent)
for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
    p = np.array([X[i, j], Y[i, j]])
    phi[i, j] = f(p)
    g = f.gradient(p)
    g /= max(np.linalg.norm(g), 1e-30)
    U[i, j] = g[0]
    V[i, j] = g[1]

div = np.zeros(X.shape)
grid_sparse = workspace.pixel_map(N)
vx = RegressedPixelGridSpline(U.T, grid_sparse.resolution, grid_sparse.extent)
vy = RegressedPixelGridSpline(V.T, grid_sparse.resolution, grid_sparse.extent)
for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
    p = np.array([X[i, j], Y[i, j]])
    vxx = vx.gradient(p)[0]
    vyy = vy.gradient(p)[1]
    div[i, j] = vxx + vyy

for i in range(iterations):
    if ROWS * COLS == 1 and i < iterations - 1:
        continue
    print("plot..")
    p_source = grid_sparse.world_to_grid(x_source)
    p = grid_sparse.grid_to_world(p_source)
    phi = phi.T
    # phi = hd.distance(U, V, div, 1. / N).T
    renderer.set_drawing_axis(i)
    renderer.draw_ws_obstacles()
    renderer.draw_ws_point(p, color='r', shape='o')
    renderer.background_matrix_eval = False
    renderer.draw_ws_img(phi, interpolate="bicubic", color_style=plt.cm.hsv)
    f = RegressedPixelGridSpline(
        phi, grid_sparse.resolution, grid_sparse.extent)
    for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
        g = -1 * f.gradient(np.array([X[i, j], Y[i, j]]))
        g /= np.linalg.norm(g)
        U[i, j] = g[0]
        V[i, j] = g[1]
    renderer._ax.quiver(X, Y, U, V, units='width')
renderer.show()
