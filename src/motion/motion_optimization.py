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

import common_imports
from motion.trajectory import *
from motion.cost_terms import *
from optimization.optimization import *
from geometry.differentiable_geometry import *
from geometry.workspace import *


class MotionOptimization2DCostMap:

    def __init__(self, extends, signed_distance_field):
        self.T = 20      # time steps
        self.dt = 0.1    # sample rate
        self.config_space_dim = 2
        self.trajectory_space_dim = (self.config_space_dim * (self.T + 2))
        self.extends = extends
        self.q_final = np.ones(2)

        # We only need the signed distance field
        # to create a trajectory optimization problem
        self.workspace = None
        self.sdf = signed_distance_field
        if self.sdf is None:
            self.create_workspace()

        # Here we combine everything to make an objective
        self.create_objective()

    def center_of_clique_map(self):
        dim = self.config_space_dim
        return RangeSubspaceMap(dim * 3, range(dim, 2 * dim))

    def obstacle_cost_map(self):
        return Compose(RangeSubspaceMap(2, [1]), SDFPotential2D(self.sdf))

    def cost(self, trajectory):
        """ compute sum of acceleration """
        return self.objective.forward(trajectory.x())

    def create_workspace(self):
        self.workspace = Workspace()
        self.workspace.obstacles.append(Circle(np.array([0.4, 0.2]), .2))
        self.workspace.obstacles.append(Circle(np.array([0.0, 0.2]), .1))
        self.sdf = SignedDistanceWorkspaceMap(self.workspace)

    def create_objective(self):
        # Creates a differentiable clique function.
        self.objective = CliquesFunctionNetwork(
            self.trajectory_space_dim,
            self.config_space_dim)

        # Smoothness term.
        squared_norm_acc = Compose(
            SquaredNorm(np.zeros(self.config_space_dim)),
            FiniteDifferencesAcceleration(self.config_space_dim, self.dt))
        self.objective.register_function_for_all_cliques(squared_norm_acc)

        # Obstacle term.
        obstacle_potential = Compose(
            self.obstacle_cost_map(),
            self.center_of_clique_map())
        squared_norm_vel = Compose(
            SquaredNorm(np.zeros(self.config_space_dim)),
            FiniteDifferencesVellocity(self.config_space_dim, self.dt))
        isometric_obstacle_cost = Product(
            obstacle_potential,
            squared_norm_vel)
        self.objective.register_function_for_all_cliques(
            isometric_obstacle_cost)

        # Terminal term.
        terminal_potential = Compose(
            SquaredNorm(self.q_final),
            self.center_of_clique_map())
        self.objective.register_function_last_clique(terminal_potential)

        print self.objective.nb_cliques()
