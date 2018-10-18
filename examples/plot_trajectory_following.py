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

import demos_common_imports
import numpy as np
from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import *
from pyrieef.motion.objective import MotionOptimization2DCostMap
from pyrieef.motion.control import KinematicTrajectoryFollowingLQR
from pyrieef.optimization import algorithms
from pyrieef.rendering.optimization import TrajectoryOptimizationViewer
from pyrieef.rendering.workspace_renderer import WorkspaceDrawer


trajectory = linear_interpolation_trajectory(
    q_init=-.22 * np.ones(2),
    q_goal=.4 * np.ones(2),
    T=22
)

dt = 0.1
lqr = KinematicTrajectoryFollowingLQR(dt, trajectory)
lqr.solve_ricatti(Q_p=13, Q_v=3, R_a=1)

workspace = sample_workspace(nb_circles=0)
extent = workspace.box.box_extent()
viewer = WorkspaceDrawer(workspace, wait_for_keyboard=True)
viewer.draw_ws_line(trajectory.list_configurations())
nb_lines = 5
x = np.linspace(-.4, 0.1, nb_lines)
y = np.linspace(-.4, 0.1, nb_lines)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.ravel(), Y.ravel()]).transpose()
for point in points:
    trajectory = lqr.integrate(point)
    viewer.draw_ws_point(trajectory.initial_configuration())
    viewer.draw_ws_line_fill(
        trajectory.list_configurations(),
        color='k',
        linewidth=.1)

viewer.show_once()
