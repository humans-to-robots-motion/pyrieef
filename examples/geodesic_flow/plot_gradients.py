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
#                                         Jim Mainprice on Wed February 25 2020

from demos_common_imports import *
import numpy as np
import pyrieef.geometry.heat_diffusion as hd
from pyrieef.geometry.workspace import *
from pyrieef.rendering.workspace_planar import WorkspaceDrawer
import matplotlib.pyplot as plt

N = 27

print(hd.discrete_2d_gradient(3, 3, axis=0))
print(hd.discrete_2d_gradient(3, 3, axis=1))

# exit()
circles = []
# circles.append(Circle(origin=[-2, -2], radius=0.1))
circles.append(Circle(origin=[.1, .25], radius=0.05))
circles.append(Circle(origin=[.2, .25], radius=0.05))
circles.append(Circle(origin=[.0, .25], radius=0.05))
circles.append(Circle(origin=[-.2, 0], radius=0.1))
workspace = Workspace()
workspace.obstacles = circles
X, Y = workspace.box.meshgrid(N)
occ = occupancy_map(N, workspace)
f = sdf(occ).T
U = -1 * np.gradient(f.T, axis=0).T
V = -1 * np.gradient(f.T, axis=1).T
phi = hd.distance_from_gradient(U, V, 1. / N, f)
phi -= phi.min()  # set min to 0 for comparison
f -= f.min()
# d = np.linalg.norm(phi - f)
# print(d)
renderer = WorkspaceDrawer(workspace, rows=1, cols=2)
renderer.set_drawing_axis(0)
renderer.draw_ws_obstacles()
renderer.draw_ws_img(sdf(occupancy_map(100, workspace)),
                     interpolate="none", color_style=plt.cm.hsv)
renderer._ax.quiver(X, Y, U, V, units='width')
renderer._ax.set_title("original")
renderer.set_drawing_axis(1)
renderer.draw_ws_obstacles()
renderer.draw_ws_img(phi.T, interpolate="none", color_style=plt.cm.hsv)
renderer._ax.quiver(X, Y, U, V, units='width')
renderer._ax.set_title("recovered from vector field")
renderer.show()
