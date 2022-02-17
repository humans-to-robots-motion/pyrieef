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

    """
    Wrapper around a Trajectory objective function
        that can draw the inner optimization quantities

        Notes:
            - Includes several modes (Matplotlib, OpenGL, 3D hieghtmaps)
            - Can also draw planar freeflyer robots
    """

    def __init__(self, objective,
                 draw=True,
                 draw_gradient=True,
                 use_3d=False,
                 use_gl=True,
                 scale=700.):
        self.objective = objective
        self.viewer = None
        self._draw_gradient = False
        self._draw_hessian = False
        self._use_3d = use_3d
        self._use_gl = use_gl
        self.points_radii = .01
        if draw:
            self._draw_gradient = draw_gradient
            self._draw_hessian = draw_gradient
            self.init_viewer(objective.workspace, scale=scale)
        self.draw_robot = False
        if hasattr(self.objective, 'robot'):
            self.draw_robot = True
            self.robot_verticies = self.objective.robot.shape

    def init_viewer(self, workspace, scale=700.):
        """
        Initializes the viewer

        Parameters
        ----------
        workspace : Workspace
            holds all obstacles
        scale : float
            size of the drawing window
        """
        from . import workspace_planar as renderer
        if not self._use_3d:
            if self._use_gl:
                self.viewer = renderer.WorkspaceOpenGl(workspace, scale=scale)
            else:
                self.viewer = renderer.WorkspaceDrawer(workspace, dynamic=True)
                self.viewer.background_matrix_eval = False
        else:
            self.viewer = renderer.WorkspaceHeightmap(workspace)
            self._draw_gradient = False
            self._draw_hessian = False

    def reset_objective(self):
        """
        Sets up the viewer
        """
        self.viewer.set_workspace(self.objective.workspace)
        self.viewer.draw_ws_background(self.objective.obstacle_potential)
        self.viewer.reset_objects()

    def draw_gradient(self, x):
        """
        Draws the gradient as an offset vector from the trajectory

        Parameters
        ----------
        x : array
            the trajectory vector
        """
        g = self.objective.objective.gradient(x)
        q_init = self.objective.q_init
        self.draw(
            Trajectory(q_init=q_init, x=x),
            Trajectory(q_init=q_init, x=-0.01 * g + x))

    def forward(self, x):
        """
        Calculates the objective

        Parameters
        ----------
        x : array
            the trajectory vector
        """
        return self.objective.objective(x)

    def gradient(self, x):
        """
        Calculates the gradient after drawing the trajectory
        and draws optionaly the gradient

        Parameters
        ----------
        x : array
            the trajectory vector
        """
        if self.viewer is not None and self._draw_gradient:
            self.draw_gradient(x)
        return self.objective.objective.gradient(x)

    def hessian(self, x):
        """
        Calculates the hessian after drawing the trajectory
        and draws optionaly the gradient

        Parameters
        ----------
        x : array
            the trajectory vector
        """
        if self.viewer is not None:
            if self._draw_hessian:
                self.draw_gradient(x)
            else:
                self.draw(Trajectory(q_init=self.objective.q_init, x=x))
        return self.objective.objective.hessian(x)

    def draw_configuration(self, q, color=(1, 0, 0), with_robot=False):
        """
        Draws the configuration along the trajectory

        Parameters
        ----------
        q : array
            positions or joint angles
        color : tuple of float
            RGB colors to draw the configurations
        with_robot : bool
            whent set to false draws only a circle at keypoint(0)
        """
        if not self._use_3d:

            if not self.draw_robot:

                self.viewer.draw_ws_circle(self.points_radii, q[:2], color)

            else:

                if with_robot:

                    # Draw contour
                    self.viewer.draw_ws_polygon(
                        self.robot_verticies, q[:2], q[2], color)

                    # Draw keypoints
                    for i in range(self.objective.robot.nb_keypoints()):
                        p = self.objective.robot.keypoint_map(i)(q)
                        r = self.objective.robot.radii[i]
                        self.viewer.draw_ws_circle(r, p, color)

                else:
                    p = self.objective.robot.keypoint_map(0)(q)
                    self.viewer.draw_ws_circle(self.points_radii, p, color)

        else:
            cost = self.objective.obstacle_potential(q)
            self.viewer.draw_ws_sphere(
                q, height=self.viewer.normalize_height(cost))

    def draw(self, trajectory, g_traj=None):
        """
        Draws the trajectory

        Parameters
        ----------
        trajectory : Trajectory
            positions, stores the configurations
        g_traj : Trajectory
            gradient, stores the configuration deltas
        """
        if self.viewer is None:
            self.init_viewer()
        if self._use_3d:
            self.viewer.reset_spheres()
        else:
            if not self._use_gl:
                # WARNING: When using matplotlib this
                # is already called in the WorkspaceDrawer constructuor
                # self.viewer.init(1, 1)
                self.viewer.draw_ws_background(
                    self.objective.obstacle_potential)
                self.viewer.draw_ws_obstacles()

        with_robot = False

        for k in range(self.objective.T + 1):

            if self.draw_robot:
                with_robot = k % 5 == 0

            q = trajectory.configuration(k)

            # Draw the initial configuration red
            # and the goal is blue and all the other are green
            # RGB (R first, G then, B Goal)
            color = (1, 0, 0) if k == 0 else (0, 1, 0)
            color = (0, 0, 1) if k == trajectory.T() else color
            # color = (1, 1, 0) if k == 31 else color

            self.draw_configuration(q, color, with_robot)

            if g_traj is not None:
                self.viewer.draw_ws_line([q[:2], g_traj.configuration(k)[:2]])

        self.viewer.show()
