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
workspace = Workspace()
workspace.obstacles = [Circle(origin=[.0, .0], radius=0.1)]
renderer = WorkspaceDrawer(workspace)
sdf = SignedDistanceWorkspaceMap(workspace)
occupany = occupancy_map(20, workspace)
reward = np.where(occupany > 0, None, -.001)
reward[0, 0] = 100
mdp = GridMDP(reward.tolist(), terminals=[(0, 0)])
X = value_iteration(mdp)
value = np.zeros(reward.shape)
for x in X:
    value[x] = X[x]

# Creates a vector field as the gradient of the signed distance field
nb_points = 14
X, Y = workspace.box.meshgrid(nb_points)
U, V = np.zeros(X.shape), np.zeros(Y.shape)
for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
    p = np.array([X[i, j], Y[i, j]])
    g = sdf.gradient(p)
    g /= np.linalg.norm(g)
    U[i, j] = g[0]
    V[i, j] = g[1]

renderer.set_drawing_axis(i)
renderer.draw_ws_obstacles()
renderer.draw_ws_point([0, 0], color='r', shape='o')
renderer.background_matrix_eval = True
# renderer.draw_ws_background(sdf, color_style=plt.cm.Blues)
renderer.draw_ws_img(value, interpolate="bicubic", color_style=plt.cm.Blues)
renderer._ax.quiver(X, Y, U, V, units='width', color='k')
renderer.show()
