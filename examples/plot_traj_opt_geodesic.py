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

from pyrieef.geometry.workspace import EnvBox
from pyrieef.motion.geodesic import GeodesicObjective2D
from pyrieef.motion.cost_terms import ObstaclePotential2D
from pyrieef.optimization import algorithms
from pyrieef.rendering.optimization import TrajectoryOptimizationViewer
import pyrieef.learning.demonstrations as demonstrations
from pyrieef.learning.dataset import *
from pyrieef.graph.shortest_path import *
from pyrieef.motion.trajectory import *
from pyrieef.utils.collision_checking import *

DRAW_MODE = "matplotlib"  # None, pyglet2d, pyglet3d or matplotlib
VERBOSE = True
BOXES = False
demonstrations.TRAJ_LENGTH = 20


def optimize_path(objective, workspace):
    """ Optimize path using Netwon's method """
    obstacle_cost = ObstaclePotential2D(SignedDistanceWorkspaceMap(workspace))
    objective.objective.embbeding = obstacle_cost
    objective.objective.obstacle_potential = obstacle_cost
    objective.objective.workspace = workspace
    if DRAW_MODE is not None:
        objective.viewer.save_images = True
        objective.viewer.workspace_id += 1
        objective.viewer.image_id = 0
        objective.viewer.draw_ws_obstacles()

    algorithms.newton_optimize_trajectory(
        objective, path, verbose=VERBOSE, maxiter=20)
    return path


motion_objective = GeodesicObjective2D(
    T=demonstrations.TRAJ_LENGTH,
    n=2,
    q_init=np.zeros(2),
    q_goal=np.zeros(2),
    embedding=None)
motion_objective.workspace = sample_circle_workspaces(1)
motion_objective.obstacle_potential = ObstaclePotential2D(SignedDistanceWorkspaceMap(motion_objective.workspace))

objective = TrajectoryOptimizationViewer(
    motion_objective,
    draw=DRAW_MODE is not None,
    draw_gradient=False,
    use_3d=DRAW_MODE == "pyglet3d",
    use_gl=DRAW_MODE == "pyglet2d")

nb_points = 40  # points for the grid on which to perform graph search.
grid = np.ones((nb_points, nb_points))

np.random.seed(0)
sampling = sample_box_workspaces if BOXES else sample_circle_workspaces
for k, workspace in enumerate(tqdm([sampling(5) for i in range(100)])):
    optimize_path(objective, workspace)
