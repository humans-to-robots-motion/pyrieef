#!/usr/bin/env python

# Copyright (c) 2019, University of Stuttgart
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
#                                         Jim Mainprice on Wed January 22 2019

from demos_common_imports import *
import numpy as np
import pyrieef.geometry.heat_diffusion as hd
from pyrieef.geometry.workspace import *
from pyrieef.rendering.workspace_planar import WorkspaceDrawer
import matplotlib.pyplot as plt
import matplotlib

ROWS = 1
COLS = 3

hd.NB_POINTS = 101       # 101
hd.TIME_FACTOR = 2000
hd.TIME_STEP = 1e-5
# hd.ALGORITHM = "crank-nicholson"
hd.ALGORITHM = "euler"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

style = plt.cm.tab20c

circles = []
circles.append(Circle(origin=[.1, .0], radius=0.1))
circles.append(Circle(origin=[.1, .25], radius=0.05))
circles.append(Circle(origin=[.2, .25], radius=0.05))
circles.append(Circle(origin=[.0, .25], radius=0.05))

workspace = Workspace()
workspace.obstacles = circles
renderer = WorkspaceDrawer(workspace, rows=ROWS, cols=COLS)
x_source = np.array([0.2, 0.15])

# ------------------------------------------------------------------------------
iterations = ROWS * COLS
U = hd.heat_diffusion(workspace, x_source, iterations)
for i in range(iterations):
    renderer.set_drawing_axis(i)
    renderer.draw_ws_obstacles()
    renderer.draw_ws_point([x_source[0], x_source[1]], color='r', shape='o')
    renderer.background_matrix_eval = False
    renderer.draw_ws_img(U[i], interpolate="bicubic", color_style=style)
renderer.show()
