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
#                                        Jim Mainprice on Wed February 12 2019

from demos_common_imports import *
import numpy as np
from pyrieef.geometry.workspace import *
from pyrieef.geometry.interpolation import *
from pyrieef.rendering.workspace_renderer import WorkspaceDrawer
from pyrieef.planning.mdp import GridMDP
from pyrieef.planning.mdp import value_iteration
import matplotlib.pyplot as plt
import itertools

USE_LWR = True

# Creates a workspace with just one circle
nb_points = 30
workspace = Workspace()
workspace.obstacles = [Circle(origin=[.1, .2], radius=0.1)]
renderer = WorkspaceDrawer(workspace)
grid = workspace.pixel_map(nb_points)
occupany = occupancy_map(nb_points, workspace)
reward = np.where(occupany > 0, -10., -.001)
reward[0, 0] = 10

# Calculate value function using value iteration
mdp = GridMDP(reward.tolist(), terminals=[(0, 0)])
X = value_iteration(mdp)
value = np.zeros(reward.shape)
for x in X:
    value[x] = X[x]
value = np.flip(value, 1).T

if USE_LWR:
    # Regress using LWR (Linear Weighted Regression)
    X_data = np.empty((nb_points ** 2, 2))
    Y_data = np.empty(nb_points ** 2)
    k = 0
    for i, j in itertools.product(range(nb_points), range(nb_points)):
        X_data[k] = grid.grid_to_world(np.array([i, j]))
        Y_data[k] = value[i, j]
        k += 1
    f = LWR(1, 2)
    f.X = [X_data]
    f.Y = [Y_data]
    f.D = [8 * np.eye(2)]
    f.ridge_lambda = [.1, .1]
else:
    # Regress using cubic-splines
    f = RegressedPixelGridSpline(value, grid.resolution, grid.extent)

print("calculate gradient...")
X, Y = workspace.box.meshgrid(15)
U, V = np.zeros(X.shape), np.zeros(Y.shape)
for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
    p = np.array([X[i, j], Y[i, j]])
    g = f.gradient(p)
    g /= np.linalg.norm(g)
    U[i, j] = g[0]
    V[i, j] = g[1]

renderer.set_drawing_axis(i)
renderer.draw_ws_obstacles()
renderer.draw_ws_point([0, 0], color='r', shape='o')
renderer.draw_ws_img(value, interpolate="none", color_style=plt.cm.Blues)
renderer._ax.quiver(X, Y, U, V, units='width', color='k')
renderer.show()
