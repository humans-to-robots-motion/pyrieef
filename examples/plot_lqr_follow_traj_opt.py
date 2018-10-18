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

from __future__ import print_function
import demos_common_imports
import numpy as np
import time
from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import *
from pyrieef.motion.objective import MotionOptimization2DCostMap
from pyrieef.motion.control import KinematicTrajectoryFollowingLQR
from pyrieef.optimization import algorithms
from pyrieef.rendering.optimization import TrajectoryOptimizationViewer
from pyrieef.rendering.workspace_renderer import WorkspaceDrawer

""" 
Optimize a grasping trajectory and compute LQR to track it
"""

q_init = np.array([-.3, .3])
q_goal = np.array([.3, .3])
q_grasp = np.array([.0, -.3])
q_grasp_approach = np.array([.0, -.2])

T_length = 20
t_grasp = int(T_length / 2)
trajectory = linear_interpolation_trajectory(q_init, q_goal, T_length)

objective = MotionOptimization2DCostMap(
    box=EnvBox(), T=T_length,
    q_init=trajectory.initial_configuration(),
    q_goal=trajectory.final_configuration())
objective.create_clique_network()
objective.add_smoothness_terms(2)
objective.add_box_limits()
objective.add_init_and_terminal_terms()
objective.add_waypoint_terms(q_grasp_approach, t_grasp - 2, 100000.)
objective.add_waypoint_terms(q_grasp, t_grasp, 100000.)
objective.create_objective()

# optimize trajectorty
t_start = time.time()
algorithms.newton_optimize_trajectory(
    objective.objective, trajectory, verbose=True)
print("optim time : {}".format(time.time() - t_start))

# solve LQR
lqr = KinematicTrajectoryFollowingLQR(dt=0.1, trajectory=trajectory)
lqr.solve_ricatti(Q_p=10, Q_v=1, R_a=0.1)

# drawing
viewer = WorkspaceDrawer(Workspace(), wait_for_keyboard=True)

# create squared meshgrid
x = np.linspace(-.4, -.1, 3)
y = np.linspace(-.4, .4, 5)
X, Y = np.meshgrid(x, y)
start_points = np.vstack([X.ravel(), Y.ravel()]).transpose()

# integrate all points in grid forward in time
for p_init in start_points:
    viewer.draw_ws_point(p_init, color='k')
    viewer.draw_ws_line_fill(
        lqr.integrate(p_init).list_configurations(),
        color='b',
        linewidth=.1)

viewer.draw_ws_line(trajectory.list_configurations())
viewer.draw_ws_point(trajectory.configuration(t_grasp), color='g')
viewer.show_once()
