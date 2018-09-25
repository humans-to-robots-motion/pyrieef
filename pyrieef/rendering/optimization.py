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

from motion.trajectory import Trajectory
import time
import numpy as np


class TrajectoryOptimizationViewer:

    """ Wrapper around a Trajectory objective function 
        tha can draw the inner optimization quantities """

    def __init__(self, objective, draw=True, draw_gradient=True):
        self.objective = objective
        self.viewer = None
        self.draw_gradient_ = False
        self.draw_hessian_ = False
        if draw:
            self.draw_gradient_ = draw_gradient
            self.draw_hessian_ = draw_gradient
            self.init_viewer()

    def init_viewer(self):
        import workspace_renderer as renderer
        self.viewer = renderer.WorkspaceOpenGl(
            self.objective.workspace)
        self.viewer.draw_ws_background(
            self.objective.obstacle_potential_from_sdf())

    def draw_gradient(self, x):
        g = self.objective.objective.gradient(x)
        q_init = self.objective.q_init
        self.draw(
            Trajectory(q_init=q_init, x=x),
            Trajectory(q_init=q_init, x=-0.01 * g + x))

    def forward(self, x):
        return self.objective.objective(x)

    def gradient(self, x):
        if self.draw_gradient_ and self.viewer is not None:
            self.draw_gradient(x)
        return self.objective.objective.gradient(x)

    def hessian(self, x):
        if self.draw_hessian_ and self.viewer is not None:
            self.draw_gradient(x)
        return self.objective.objective.hessian(x)

    def draw(self, trajectory, g_traj=None):
        if self.viewer is None:
            self.init_viewer()
        q_init = self.objective.q_init
        for k in range(self.objective.T + 2):
            q = trajectory.configuration(k)
            self.viewer.draw_ws_circle(
                .01, q, color=(0, 0, 1) if k == 0 else (0, 1, 0))
            if g_traj is not None:
                self.viewer.draw_ws_line(q, g_traj.configuration(k))
        self.viewer.show()
        time.sleep(0.1)
