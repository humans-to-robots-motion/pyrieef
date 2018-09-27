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


def initialize_objective(trajectory):
    objective = MotionOptimization2DCostMap(
        box=EnvBox(origin=np.array([0, 0]), dim=np.array([1., 1.])),
        T=trajectory.T(),
        q_init=trajectory.initial_configuration(),
        q_goal=trajectory.final_configuration()
    )
    objective.create_clique_network()
    objective.add_smoothness_terms(2)
    # objective.add_smoothness_terms(1)
    objective.add_obstacle_terms()
    objective.add_box_limits()
    objective.add_init_and_terminal_terms()
    objective.create_objective()
    return objective


def motion_optimimization(workspace, costmap):
    print "Checkint Motion Optimization"
    trajectory = linear_interpolation_trajectory(
        q_init=-.22 * np.ones(2),
        q_goal=.3 * np.ones(2),
        T=22
    )
    sdf = SignedDistanceWorkspaceMap(workspace, costmap)
    objective = initialize_objective(trajectory, sdf)
    f = TrajectoryOptimizationViewer(objective, draw=False, draw_gradient=True)
    algorithms.newton_optimize_trajectory(f, trajectory)
    f.draw(trajectory)


def run_example():
    workspace = sample_workspace(nb_circles=4)


if __name__ == "__main__":

    while True:

        motion_optimimization()

        if raw_input("Press [q] to quit or enter to continue : ") == "q":
            break
