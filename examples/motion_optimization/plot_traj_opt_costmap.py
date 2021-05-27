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
from scipy import optimize
from pyrieef.motion.objective import *
import time
from numpy.testing import assert_allclose
from pyrieef.rendering.optimization import *
from pyrieef.optimization import algorithms
from pyrieef.rendering.workspace_planar import WorkspaceDrawer


def initialize_objective(trajectory, workspace, costmap):
    objective = MotionOptimization2DCostMap(
        box=EnvBox(origin=np.array([0, 0]), dim=np.array([1., 1.])),
        T=trajectory.T(),
        q_init=trajectory.initial_configuration(),
        q_goal=trajectory.final_configuration(),
        signed_distance_field=SignedDistanceWorkspaceMap(workspace)
    )

    objective.create_clique_network()

    if costmap is not None:
        objective.costmap = costmap
        objective.add_costgrid_terms()
    else:
        objective.obstacle_potential_from_sdf()
        objective.add_obstacle_terms()

    objective.add_smoothness_terms(2)
    # objective.add_smoothness_terms(1)

    objective.add_box_limits()
    objective.add_init_and_terminal_terms()
    objective.create_objective()
    return objective


def motion_optimimization(workspace, costmap=None):
    print("Checkint Motion Optimization")
    trajectory = linear_interpolation_trajectory(
        q_init=-.22 * np.ones(2),
        q_goal=.3 * np.ones(2),
        T=22
    )
    objective = initialize_objective(trajectory, workspace, costmap)
    f = TrajectoryOptimizationViewer(objective, draw=False, draw_gradient=True)
    algorithms.newton_optimize_trajectory(f, trajectory)
    f.draw(trajectory)


def plot_costmaps():
    workspace = sample_circle_workspaces(nb_circles=4)
    grid_sparse = workspace.box.stacked_meshgrid(24)
    grid_dense = workspace.box.stacked_meshgrid(100)
    extent = workspace.box.box_extent()
    sdf = SignedDistanceWorkspaceMap(workspace)

    viewer = WorkspaceDrawer(workspace, wait_for_keyboard=True,
                             rows=1, cols=2, scale=1.)

    viewer.set_drawing_axis(0)
    viewer.set_workspace(workspace)
    viewer.draw_ws_img(sdf(grid_dense).T)
    viewer.draw_ws_obstacles()

    viewer.set_drawing_axis(1)
    viewer.set_workspace(workspace)
    viewer.draw_ws_img(sdf(grid_sparse).T)
    viewer.draw_ws_obstacles()

    viewer.show_once()


if __name__ == "__main__":

    while True:
        plot_costmaps()
        if input("Press [q] to quit or enter to continue : ") == "q":
            break
