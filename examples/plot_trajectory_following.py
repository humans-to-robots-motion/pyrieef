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
import time
import numpy as np
from numpy.testing import assert_allclose

from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import *
from pyrieef.motion.objective import MotionOptimization2DCostMap
from pyrieef.motion.control import KinematicTrajectoryFollowingLQR
from pyrieef.optimization import algorithms
from pyrieef.rendering.optimization import TrajectoryOptimizationViewer
from pyrieef.rendering.workspace_renderer import WorkspaceDrawer


def integrate_lqr_policy(lqr, q_init):
    trajectory = Trajectory(T=lqr.T(), n=q_init.size)
    dt = lqr.dt()
    x_t = np.hstack([q_init, np.zeros(q_init.size)])
    for i in range(lqr.T() + 1):
        # compute acceleration
        u_t = lqr.policy(i * dt, x_t.reshape(q_init.size * 2, 1))
        a_t = np.array(u_t).reshape((q_init.size, ))
        v_t = x_t[q_init.size:]
        q_t = x_t[:q_init.size]

        # 2 integrate foward and update state
        q_t1 = q_t + v_t * dt + a_t * (dt ** 2)
        v_t1 = v_t + a_t * dt
        trajectory.configuration(i)[:] = q_t1
        x_t = np.hstack([q_t1, v_t1])
    return trajectory


trajectory = linear_interpolation_trajectory(
    q_init=-.22 * np.ones(2),
    q_goal=.3 * np.ones(2),
    T=22
)

dt = 0.1
lqr = KinematicTrajectoryFollowingLQR(dt, trajectory)
lqr.solve_ricatti(Q_p=2, Q_v=1, R_a=0.1)

workspace = sample_workspace(nb_circles=0)
extent = workspace.box.box_extent()
viewer = WorkspaceDrawer(workspace, wait_for_keyboard=True)
viewer.draw_ws_line(trajectory.list_configurations())
nb_lines = 7
X, Y = workspace.box.meshgrid(nb_lines)
points = np.vstack([X.ravel(), Y.ravel()]).transpose()
for point in points:
    trajectory = integrate_lqr_policy(lqr, point)
    viewer.draw_ws_line_fill(trajectory.list_configurations(), color='b')

viewer.show_once()
