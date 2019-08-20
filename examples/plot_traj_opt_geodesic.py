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
from tqdm import tqdm

from pyrieef.motion.geodesic import GeodesicObjective2D
from pyrieef.motion.cost_terms import CostGridPotential2D
from pyrieef.motion.cost_terms import ObstaclePotential2D
from pyrieef.optimization import algorithms
from pyrieef.rendering.optimization import TrajectoryOptimizationViewer
from pyrieef.learning.dataset import *
from pyrieef.graph.shortest_path import *
from pyrieef.motion.trajectory import *
from pyrieef.utils.collision_checking import *

DRAW_MODE = "matplotlib"  # None, pyglet2d, pyglet3d or matplotlib
VERBOSE = True
BOXES = False
TRAJ_LENGTH = 20
ALPHA = 10.
MARGIN = .20
OFFSET = 0.1


def optimize_path(objective, trajectory, workspace):
    """ Optimize path using Netwon's method """
    sdf = SignedDistanceWorkspaceMap(workspace)
    objective.objective.embedding = ObstaclePotential2D(
        sdf, scaling=100., alpha=1.e-2)
    objective.objective.obstacle_potential = CostGridPotential2D(
        sdf, ALPHA, MARGIN, OFFSET)
    objective.objective.workspace = workspace
    objective.objective.create_clique_network()
    if DRAW_MODE is not None:
        objective.reset_objective()
        objective.viewer.save_images = True
        objective.viewer.workspace_id += 1
        objective.viewer.image_id = 0
        objective.viewer.draw_ws_obstacles()

    algorithms.newton_optimize_trajectory(
        objective, trajectory, verbose=VERBOSE, maxiter=100)
    return trajectory


trajectory = linear_interpolation_trajectory(
    q_init=-.22 * np.ones(2),
    q_goal=.3 * np.ones(2),
    T=TRAJ_LENGTH
)

motion_objective = GeodesicObjective2D(
    T=trajectory.T(),
    n=2,
    q_init=trajectory.initial_configuration(),
    q_goal=trajectory.final_configuration(),
    embedding=None)
workspace = sample_circle_workspaces(1)
motion_objective.workspace = workspace
sdf = SignedDistanceWorkspaceMap(workspace)
motion_objective.embedding = ObstaclePotential2D(sdf)
motion_objective.obstacle_potential = CostGridPotential2D(
    sdf, ALPHA, MARGIN, OFFSET)
motion_objective.create_clique_network()

objective = TrajectoryOptimizationViewer(
    motion_objective,
    draw=DRAW_MODE is not None,
    draw_gradient=False,
    use_3d=DRAW_MODE == "pyglet3d",
    use_gl=DRAW_MODE == "pyglet2d")

np.random.seed(0)
sampling = sample_box_workspaces if BOXES else sample_circle_workspaces
for k, workspace in enumerate(tqdm([sampling(1) for i in range(1)])):
    trajectory = linear_interpolation_trajectory(
        motion_objective.q_init,
        motion_objective.q_goal,
        motion_objective.T)
    optimize_path(objective, trajectory, workspace)
