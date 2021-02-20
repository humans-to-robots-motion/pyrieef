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
import matplotlib.pyplot as plt
import itertools


# Creates a workspace with just one circle
workspace = Workspace()
workspace.obstacles = [Circle(origin=[.0, .0], radius=0.1)]
sdf = SignedDistanceWorkspaceMap(workspace)

# Creates a vector field as the gradient of the signed distance field
nb_points = 14
X, Y = workspace.box.meshgrid(nb_points)
U, V = np.zeros(X.shape), np.zeros(Y.shape)
X_data = np.empty((nb_points ** 2, 2))
Y1 = np.empty((nb_points ** 2))
Y2 = np.empty((nb_points ** 2))
k = 0
for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
    p = np.array([X[i, j], Y[i, j]])
    g = sdf.gradient(p)
    g /= np.linalg.norm(g)
    U[i, j] = g[0]
    V[i, j] = g[1]

    X_data[k, :] = p
    Y1[k] = g[0]
    Y2[k] = g[1]
    k += 1

# Store the Data in a LWR (Linear Weighted Regression)
# object where dimension of the vector field are being abstracted
f = LWR(2, 2)
f.X = [X_data, X_data]
f.Y = [Y1, Y2]
f.D = [np.eye(2), np.eye(2)]
f.ridge_lambda = [.1, .1]

# Querries the regressed function f
# And then draws the interolated field in red and original points in black
X_int, Y_int = workspace.box.meshgrid(30)
U_int, V_int = np.zeros(X_int.shape), np.zeros(Y_int.shape)
for i, j in itertools.product(range(X_int.shape[0]), range(X_int.shape[1])):
    g = f(np.array([X_int[i, j], Y_int[i, j]]))
    U_int[i, j] = g[0]
    V_int[i, j] = g[1]

renderer = WorkspaceDrawer(workspace)
renderer.set_drawing_axis(i)
renderer.draw_ws_obstacles()
renderer.draw_ws_point([0, 0], color='r', shape='o')
renderer.background_matrix_eval = True
renderer.draw_ws_background(sdf, color_style=plt.cm.Blues)
renderer._ax.quiver(X, Y, U, V, units='width', color='k')
renderer._ax.quiver(X_int, Y_int, U_int, V_int, units='width', color='r')
renderer.show()
