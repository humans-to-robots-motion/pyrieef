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
from pyrieef.geometry.workspace import *
from pyrieef.geometry import heat_diffusion
from pyrieef.rendering.workspace_renderer import WorkspaceDrawer
import matplotlib.pyplot as plt


ROWS = 3
COLS = 3

heat_diffusion.NB_POINTS = 21
iterations = ROWS * COLS
workspace = Workspace()
source = [0, 0]
renderer = WorkspaceDrawer(workspace, rows=ROWS, cols=COLS)
U = heat_diffusion.heat_diffusion(workspace, source, iterations)
for i in range(iterations):
    renderer.set_drawing_axis(i)
    renderer.draw_ws_obstacles()
    renderer.draw_ws_point(source, color='k', shape='o')
    renderer.background_matrix_eval = False
    renderer.draw_ws_img(U[i], interpolate="bicubic", color_style=plt.cm.hsv)
renderer.show()
