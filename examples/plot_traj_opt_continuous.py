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
from pyrieef.rendering.optimization import *
from pyrieef.optimization import algorithms
import time
from numpy.testing import assert_allclose


def initialize_objective(objective, trajectory):
    objective.q_init = trajectory.initial_configuration()
    objective.create_clique_network()
    objective.add_smoothness_terms(trajectory.n())
    objective.add_obstacle_terms()
    objective.add_box_limits()
    objective.add_attractor(trajectory)
    objective.create_objective()


def resample(trajectory):
    """ remove first configuration and interpolate """
    n = trajectory.n()
    T = trajectory.T()
    new_trajectory = ContinuousTrajectory(T - 1, n)
    new_trajectory.set(trajectory.x()[n:])
    q_prev = new_trajectory.initial_configuration()
    for t in range(T + 1):
        q_t = new_trajectory.configuration_at_parameter(float(t) / float(T))
        trajectory.configuration(t)[:] = q_t
    trajectory.final_configuration()[:] = new_trajectory.final_configuration()


def motion_optimimization():
    print("Checkint Motion Optimization")
    T = 20
    trajectory = linear_interpolation_trajectory(
        q_init=-.22 * np.ones(2),
        q_goal=.3 * np.ones(2),
        T=T
    )
    objective = MotionOptimization2DCostMap(
        box=EnvBox(origin=np.array([0, 0]), dim=np.array([1., 1.])),
        T=trajectory.T(),
        q_init=trajectory.initial_configuration(),
        q_goal=trajectory.final_configuration()
    )
    f = TrajectoryOptimizationViewer(objective, draw=False, draw_gradient=True)
    for t in range(T + 1):
        initialize_objective(objective, trajectory)
        algorithms.newton_optimize_trajectory(objective.objective, trajectory)
        f.draw(trajectory)
        resample(trajectory)

if __name__ == "__main__":
    motion_optimimization()
    input("Press Enter to continue...")
