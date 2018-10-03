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
from tqdm import tqdm

from pyrieef.geometry.workspace import EnvBox
from pyrieef.motion.trajectory import linear_interpolation_trajectory
from pyrieef.motion.objective import MotionOptimization2DCostMap
from pyrieef.optimization import algorithms
from pyrieef.rendering.optimization import TrajectoryOptimizationViewer
import pyrieef.learning.demonstrations as demonstrations
from pyrieef.learning.dataset import *
from pyrieef.graph.shortest_path import *
from pyrieef.motion.trajectory import *
from pyrieef.utils.collision_checking import *

DRAW = True
demonstrations.TRAJ_LENGTH = 20


def optimize_path(objective, workspace, path):
    """ Optimize path using Netwon's method """
    obstacle_cost = demonstrations.obsatcle_potential(workspace)
    objective.objective.set_problem(workspace, path, obstacle_cost)
    if DRAW:
        objective.reset_objective(objective.objective)
        objective.viewer.save_images = False
        objective.viewer.workspace_id += 1
        objective.viewer.image_frame_count = 0

        for o in workspace.obstacles:
            p = o.origin + o.radius * np.array([0, 1])
            c = obstacle_cost(p) + 20
            objective.viewer.draw_ws_circle(
                o.radius,
                o.origin,
                height=objective.viewer.normalize_height(c)
            )

    algorithms.newton_optimize_trajectory(objective, path, verbose=True)
    return path


def graph_search_path(workspace, graph, nb_points):
    """ Graph search using Dijkstra
            1) samples a path that has collision with the enviroment
                and perform graph search on a grid (nb_points x nb_points)
            2) convert path to world coordinates
            3) interpolate path continuously """
    path = demonstrations.sample_path(workspace, graph, nb_points)

    trajectory = ContinuousTrajectory(len(path) - 1, 2)
    pixel_map = workspace.pixel_map(nb_points)
    for i, p in enumerate(path):
        trajectory.configuration(i)[:] = pixel_map.grid_to_world(np.array(p))
    T = demonstrations.TRAJ_LENGTH

    interpolated_path = Trajectory(T, 2)
    for i, s in enumerate(np.linspace(0, 1, T)):
        q = trajectory.configuration_at_parameter(s)
        interpolated_path.configuration(i)[:] = q
    q_goal = trajectory.final_configuration()
    interpolated_path.configuration(T)[:] = q_goal
    interpolated_path.configuration(T + 1)[:] = q_goal
    return interpolated_path

objective = TrajectoryOptimizationViewer(
    MotionOptimization2DCostMap(
        box=EnvBox(
            origin=np.array([0, 0]),
            dim=np.array([1., 1.])),
        T=demonstrations.TRAJ_LENGTH,
        q_init=np.zeros(2),
        q_goal=np.zeros(2)),
    draw=DRAW,
    draw_gradient=True,
    use_3d_viewer=True)

nb_points = 40
show_demo_id = -1

grid = np.ones((nb_points, nb_points))
graph = CostmapToSparseGraph(grid, average_cost=False)
graph.convert()
workspaces = load_workspaces_from_file(filename='workspaces_1k_small.hdf5')
trajectories = [None] * len(workspaces)
for k, workspace in enumerate(tqdm(workspaces)):
    interpolated_path = graph_search_path(workspace, graph, nb_points)
    optimized_trajectory = optimize_path(
        objective, workspace, interpolated_path)
    if collision_check_trajectory(workspace, optimized_trajectory):
        print("Warning: has collision !!!")

    result = [None] * demonstrations.TRAJ_LENGTH
    for i in range(len(result)):
        result[i] = optimized_trajectory.configuration(i)
    if show_demo_id == k and not options.show_result:
        break
