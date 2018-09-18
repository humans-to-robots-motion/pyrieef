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

import workspace_renderer as renderer
from motion.trajectory import Trajectory
import time
import numpy as np


class TrajectoryOptimizationViewer:

    """ Wrapper around a Trajectory objective function 
        tha can draw the inner optimization quantities """

    def __init__(self, objective, draw):
        self.objective = objective
        self.viewer = None
        if draw:
            self.init_viewer()

    def init_viewer(self):
        self.viewer = renderer.WorkspaceRender(
            self.objective.workspace)
        self.viewer.draw_ws_background(self.objective.obstacle_costmap())

    def evaluate(self, x):
        return self.objective.objective(x)

    def gradient(self, x, draw=True):
        g = self.objective.objective.gradient(x)
        if draw and self.viewer is not None:
            q_init = self.objective.q_init
            self.draw(
                Trajectory(q_init=q_init, x=x),
                Trajectory(q_init=q_init, x=-0.01 * g + x))
        return g

    def hessian(self, x):
        return self.objective.objective.hessian(x)

    def draw(self, trajectory, g_traj=None):
        q_init = self.objective.q_init
        for k in range(self.objective.T + 2):
            q = trajectory.configuration(k)
            self.viewer.draw_ws_circle(
                .01, q, color=(0, 0, 1) if k == 0 else (0, 1, 0))
            if g_traj is not None:
                self.viewer.draw_ws_line(q, g_traj.configuration(k))
        self.viewer.render()
        time.sleep(0.1)
