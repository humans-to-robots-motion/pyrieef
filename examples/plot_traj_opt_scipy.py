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

from pyrieef.geometry.workspace import EnvBox
from pyrieef.motion.trajectory import linear_interpolation_trajectory
from pyrieef.motion.trajectory import no_motion_trajectory
from pyrieef.motion.objective import MotionOptimization2DCostMap
from pyrieef.optimization import algorithms
from pyrieef.rendering.optimization import TrajectoryOptimizationViewer

# ----------------------------------------------------------------------------
print("Run Motion Optimization")
trajectory = linear_interpolation_trajectory(
    q_init=-.22 * np.ones(2),
    q_goal=.3 * np.ones(2),
    T=22
)
# trajectory = no_motion_trajectory(q_init=-.22 * np.ones(2), T=22)

# -----------------------------------------------------------------------------
# The Objective function is wrapped in the optimization viewer object
# which draws the trajectory as it is being optimized
objective = TrajectoryOptimizationViewer(
    MotionOptimization2DCostMap(
        box=EnvBox(
            origin=np.array([0, 0]),
            dim=np.array([1., 1.])),
        T=trajectory.T(),
        q_init=trajectory.initial_configuration(),
        q_goal=trajectory.final_configuration()),
    draw=False,
    draw_gradient=True)

# ----------------------------------------------------------------------------
# Runs a Newton optimization algorithm on the objective
# ----------------------------------------------------------------------------
t_0 = time.time()
algorithms.newton_optimize_trajectory(
    objective, trajectory, verbose=True, maxiter=100)

print("Done. ({} sec.)".format(time.time() - t_0))
while True:
    objective.draw(trajectory)
