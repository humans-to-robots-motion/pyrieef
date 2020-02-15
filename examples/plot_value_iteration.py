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


# Creates a workspace with just one circle
nb_points = 30
workspace = Workspace()
workspace.obstacles = [Circle(origin=[.1, .2], radius=0.1)]
renderer = WorkspaceDrawer(workspace)
sdf = SignedDistanceWorkspaceMap(workspace)
occupany = occupancy_map(nb_points, workspace)
reward = np.where(occupany > 0, -10., -.001)
reward[0, 0] = 10
mdp = GridMDP(reward.tolist(), terminals=[(0, 0)])
X = value_iteration(mdp)
value = np.zeros(reward.shape)
for x in X:
    value[x] = X[x]
value = np.flip(value, 1).T

# Creates a vector field as the gradient of the signed distance field
grid = workspace.pixel_map(nb_points)
f = RegressedPixelGridSpline(value, grid.resolution, grid.extent)
X, Y = workspace.box.meshgrid(20)
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
