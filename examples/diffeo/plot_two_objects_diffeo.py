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
from pyrieef.rendering.workspace_planar import WorkspaceDrawer
import itertools
import matplotlib.pyplot as plt

cmap = plt.get_cmap('viridis')

p1 = ellipse_polygon(
    a=.2,
    b=.1,
    focus=[.0, .0],
    translation=[.1, .25],
    orientation=0.)
obstacles = []
obstacles.append(AnalyticConvexPolygon(polygon=p1))
obstacles.append(AnalyticCircle(origin=[.1, .0], radius=0.1))

workspace = Workspace()
workspace.obstacles = [phi.object() for phi in obstacles]
renderer = WorkspaceDrawer(workspace)

x_goal = np.array([0.4, 0.4])
nx, ny = (5, 4)
x = np.linspace(-.2, -.05, nx)
y = np.linspace(-.5, -.1, ny)

analytical_circles = AnalyticMultiDiffeo(obstacles)

sclar_color = 0.
for i, j in itertools.product(list(range(nx)), list(range(ny))):
    sclar_color += 1. / (nx * ny)
    x_init = np.array([x[i], y[j]])
    print("x_init : ", x_init)

    # Does not have an inverse.
    [line, line_inter] = InterpolationGeodescis(
        analytical_circles, x_init, x_goal)

    # line = NaturalGradientGeodescis(analytical_circles, x_init, x_goal)
    renderer.draw_ws_line(line, color=cmap(sclar_color))
    renderer.draw_ws_point([x_init[0], x_init[1]], color='r', shape='o')
renderer.draw_ws_obstacles()
renderer.draw_ws_point([x_goal[0], x_goal[1]], color='r', shape='o')
renderer.show()
