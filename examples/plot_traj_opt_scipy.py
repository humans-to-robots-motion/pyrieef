#!/usr/bin/env python

# Copyright (c) 2015 Max Planck Institute
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
# Jim Mainprice on Sunday June 17 2018

import demos_common_imports
from scipy import optimize
from pyrieef.motion.objective import *
import time
from numpy.testing import assert_allclose


class TrajectoryOptimizationViewer:

    def __init__(self, objective, draw):
        self.objective = objective
        self.viewer = None
        if draw:
            self.init_viewer()

    def init_viewer(self):
        from pyrieef.rendering import workspace_renderer
        from pyrieef.rendering import opengl
        self.viewer = workspace_renderer.WorkspaceRender(
            self.objective.workspace)
        self.viewer.draw_ws_background(self.objective.obstacle_cost_map())

    def evaluate(self, x):
        if self.viewer is not None:
            trajectory = Trajectory(self.objective.T)
            trajectory.set(x)
            g_traj = Trajectory(self.objective.T)
            g_traj.set(-1. * self.gradient(x) + trajectory.x()[:])
            for k in range(self.objective.T + 1):
                q = trajectory.configuration(k)
                self.viewer.draw_ws_circle(.01, q)
                self.viewer.draw_ws_line(q, g_traj.configuration(k))
            self.viewer.render()
            time.sleep(0.02)
        return self.objective.objective(x)

    def gradient(self, x):
        return self.objective.objective.gradient(x)

    def hessian(self, x):
        return self.objective.metric
        # return np.eye(self.objective.metric.shape[0])


def motion_optimimization():
    print "Checkint Motion Optimization"
    trajectory = linear_interpolation_trajectory(
        q_init=-.2 * np.ones(2),
        q_goal=.3 * np.ones(2),
        T=22
    )
    objective = MotionOptimization2DCostMap(
        T=trajectory.T(),
        q_init=trajectory.configuration(0),
        q_goal=trajectory.final_configuration()
    )
    gtol = 1e-07
    assert check_jacobian_against_finite_difference(
        objective.objective, verbose=False)
    f = TrajectoryOptimizationViewer(objective, draw=True)
    t_start = time.time()
    res = optimize.minimize(
        f.evaluate,
        trajectory.x(),
        method='BFGS',
        jac=f.gradient,
        options={'gtol': gtol, 'disp': False})
    print "optimization done in {} sec.".format(time.time() - t_start)
    # objective.optimize(q_init=np.zeros(2), trajectory=trajectory)
    # print trajectory.x().shape
    # print res.x.shape
    # print res
    # print trajectory.x()
    # print "- res.jac : {}".format(res.jac.shape)
    print "gradient norm : ", np.linalg.norm(res.jac)
    # print "jac : ", res.jac
    # assert_allclose(res.jac, np.zeros(res.jac.size), atol=1e-1)

if __name__ == "__main__":
    motion_optimimization()
    # import cProfile
    # cProfile.run('motion_optimimization()')
