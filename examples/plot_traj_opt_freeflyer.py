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
#                                      Jim Mainprice on Tuesday January 28 2020

import demos_common_imports
import time
import numpy as np
from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import linear_interpolation_trajectory
from pyrieef.motion.freeflyer import FreeflyerObjective
from pyrieef.motion.cost_terms import *
from pyrieef.kinematics.robot import *
from pyrieef.optimization import algorithms
from pyrieef.rendering.optimization import TrajectoryOptimizationViewer

DRAW_MODE = "pyglet2d"  # None, pyglet2d, pyglet3d or matplotlib
ALPHA = 10.
MARGIN = .20
OFFSET = 0.1

print("Run Motion Optimization")
robot = create_robot_from_file(scale=.02)
trajectory = linear_interpolation_trajectory(
    q_init=np.array([-.2, -.2, 0]),
    q_goal=np.array([.3, .3, 0]),
    T=22
)
# trajectory = no_motion_trajectory(q_init=-.22 * np.ones(2), T=22)
# The Objective function is wrapped in the optimization viewer object
# which draws the trajectory as it is being optimized
objective = FreeflyerObjective(
    T=trajectory.T(),
    n=trajectory.n(),
    q_init=trajectory.initial_configuration(),
    q_goal=trajectory.final_configuration(),
    robot=robot)

workspace = Workspace()
workspace.obstacles.append(Circle(np.array([0.2, .15]), .1))
workspace.obstacles.append(Circle(np.array([-.1, .15]), .1))
sdf = SignedDistanceWorkspaceMap(workspace)
phi = ObstaclePotential2D(
    signed_distance_field=sdf,
    scaling=100.,
    alpha=1.e-2)
cost = CostGridPotential2D(sdf, ALPHA, MARGIN, OFFSET)
objective.embedding = phi
objective.obstacle_potential = cost
objective.workspace = workspace
objective.create_clique_network()
objective = TrajectoryOptimizationViewer(
    objective, draw=True, draw_gradient=True,
    use_3d=DRAW_MODE == "pyglet3d",
    use_gl=DRAW_MODE == "pyglet2d")

t_0 = time.time()
algorithms.newton_optimize_trajectory(
    objective, trajectory, verbose=True, maxiter=100)

print("Done. ({} sec.)".format(time.time() - t_0))
while True:
    objective.draw(trajectory)
