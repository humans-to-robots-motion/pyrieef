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

    def __init__(self, objective,
                 draw=True,
                 draw_gradient=True,
                 use_3d=False,
                 use_gl=True):
        self.objective = objective
        self.viewer = None
        self._draw_gradient = False
        self._draw_hessian = False
        self._use_3d = use_3d
        self._use_gl = use_gl
        if draw:
            self._draw_gradient = draw_gradient
            self._draw_hessian = draw_gradient
            self.init_viewer()

    def init_viewer(self):
        from . import workspace_renderer as renderer
        if not self._use_3d:
            if self._use_gl:
                self.viewer = renderer.WorkspaceOpenGl(
                    self.objective.workspace)
            else:
                self.viewer = renderer.WorkspaceDrawer(
                    self.objective.workspace)
        else:
            self.viewer = renderer.WorkspaceHeightmap(self.objective.workspace)
            self._draw_gradient = False
            self._draw_hessian = False
        self.reset_objective()

    def reset_objective(self):
        self.viewer.set_workspace(self.objective.workspace)
        self.viewer.draw_ws_background(self.objective.obstacle_potential)
        self.viewer.reset_objects()

    def draw_gradient(self, x):
        g = self.objective.objective.gradient(x)
        q_init = self.objective.q_init
        self.draw(
            Trajectory(q_init=q_init, x=x),
            Trajectory(q_init=q_init, x=-0.01 * g + x))

    def forward(self, x):
        return self.objective.objective(x)

    def gradient(self, x):
        if self.viewer is not None and self._draw_gradient:
            self.draw_gradient(x)
        return self.objective.objective.gradient(x)

    def hessian(self, x):
        if self.viewer is not None:
            if self._draw_hessian:
                self.draw_gradient(x)
            else:
                self.draw(Trajectory(q_init=self.objective.q_init, x=x))
        return self.objective.objective.hessian(x)

    def draw(self, trajectory, g_traj=None):

        if self.viewer is None:
            self.init_viewer()
        if self._use_3d:
            self.viewer.reset_spheres()
        else:
            if not self._use_gl:
                self.viewer.init(1, 1)
                self.viewer.draw_ws_background(
                    self.objective.obstacle_potential)
                self.viewer.draw_ws_obstacles()

        q_init = self.objective.q_init
        for k in range(self.objective.T + 1):
            q = trajectory.configuration(k)
            color = (0, 0, 1) if k == 0 else (0, 1, 0)
            color = (1, 0, 0) if k == trajectory.T() else color
            if not self._use_3d:
                self.viewer.draw_ws_circle(.01, q, color)
            else:
                cost = self.objective.obstacle_potential(q)
                self.viewer.draw_ws_sphere(
                    q, height=self.viewer.normalize_height(cost))
            if g_traj is not None:
                self.viewer.draw_ws_line([q, g_traj.configuration(k)])

        self.viewer.show()
