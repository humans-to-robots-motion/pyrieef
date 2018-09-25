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
from rendering.optimization import *


def initialize_objective(trajectory):
    objective = MotionOptimization2DCostMap(
        box=EnvBox(origin=np.array([0, 0]), dim=np.array([1., 1.])),
        T=trajectory.T(),
        q_init=trajectory.initial_configuration(),
        q_goal=trajectory.final_configuration()
    )
    objective.create_clique_network()
    # objective.add_smoothness_terms(2)
    objective.add_smoothness_terms(1)
    # objective.add_obstacle_terms()
    objective.add_box_limits()
    objective.add_init_and_terminal_terms()
    objective.create_objective()
    return objective


def motion_optimimization():
    print "Checkint Motion Optimization"
    trajectory = linear_interpolation_trajectory(
        q_init=-.22 * np.ones(2),
        q_goal=.3 * np.ones(2),
        T=22
    )
    objective = initialize_objective(trajectory)
    f = TrajectoryOptimizationViewer(objective, draw=False, draw_gradient=True)
    t_start = time.time()
    x = trajectory.active_segment()
    res = optimize.minimize(
        x0=x,
        method='Newton-CG',
        fun=f.forward,
        jac=f.gradient,
        hess=f.hessian,
        options={'maxiter': 100, 'gtol': 1e-05, 'disp': True}
    )
    trajectory.active_segment()[:] = res.x
    f.draw(trajectory)
    print "optimization done in {} sec.".format(time.time() - t_start)
    print "gradient norm : ", np.linalg.norm(res.jac)

if __name__ == "__main__":
    motion_optimimization()
    raw_input("Press Enter to continue...")
    # Can do some profiling here
    # import cProfile
    # cProfile.run('motion_optimimization()')
