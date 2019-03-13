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
from pyrieef.motion.objective import MotionOptimization2DCostMap
from pyrieef.optimization import algorithms
from pyrieef.rendering.optimization import TrajectoryOptimizationViewer
import pyrieef.learning.demonstrations as demonstrations
from pyrieef.learning.dataset import *
from pyrieef.graph.shortest_path import *
from pyrieef.motion.trajectory import *
from pyrieef.utils.collision_checking import *

DRAW = True
DRAW_3D = False
VERBOSE = True
demonstrations.TRAJ_LENGTH = 20


def optimize_path(objective, workspace, path):
    """ Optimize path using Netwon's method """
    obstacle_cost = demonstrations.obsatcle_potential(workspace)
    objective.objective.set_problem(workspace, path, obstacle_cost)
    if DRAW:
        objective.reset_objective()
        objective.viewer.save_images = True
        objective.viewer.workspace_id += 1
        objective.viewer.image_id = 0
        objective.viewer.draw_ws_obstacles()

    algorithms.newton_optimize_trajectory(
        objective, path, verbose=VERBOSE, maxiter=100)
    return path


def graph_search_path(graph, workspace, nb_points):
    """ Find feasible path using Dijkstra's algorithm
            1) samples a path that has collision with the enviroment
                and perform graph search on a grid (nb_points x nb_points)
            2) convert path to world coordinates
            3) interpolate path continuously """
    path = demonstrations.sample_path(workspace, graph, nb_points, True)

    # convert to world coordinates
    path_world = ContinuousTrajectory(len(path) - 1, 2)
    pixel_map = workspace.pixel_map(nb_points)
    for i, p in enumerate(path):
        path_world.configuration(i)[:] = pixel_map.grid_to_world(np.array(p))
    T = demonstrations.TRAJ_LENGTH

    # interpolate the path
    interpolated_path = Trajectory(T, 2)
    for i, s in enumerate(np.linspace(0, 1, T)):
        q = path_world.configuration_at_parameter(s)
        interpolated_path.configuration(i)[:] = q
    q_goal = path_world.final_configuration()
    interpolated_path.configuration(T)[:] = q_goal
    interpolated_path.configuration(T + 1)[:] = q_goal
    return interpolated_path


motion_objective = MotionOptimization2DCostMap(
    box=EnvBox(origin=np.array([0, 0]), dim=np.array([1., 1.])),
    T=demonstrations.TRAJ_LENGTH,
    q_init=np.zeros(2),
    q_goal=np.zeros(2))
objective = TrajectoryOptimizationViewer(
    motion_objective, draw=DRAW, draw_gradient=True, use_3d_viewer=DRAW_3D)

nb_points = 40  # points for the grid on which to perform graph search.
grid = np.ones((nb_points, nb_points))
graph = CostmapToSparseGraph(grid, average_cost=False)
graph.convert()

np.random.seed(0)
workspaces = [sample_workspace(nb_circles=5) for i in range(100)]
# workspaces = [sample_box_workspaces(5) for i in range(100)]
for k, workspace in enumerate(tqdm(workspaces)):
    path = graph_search_path(graph, workspace, nb_points)
    if collision_check_trajectory(workspace, path):
        continue
    optimize_path(objective, workspace, path)
