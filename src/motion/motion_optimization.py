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

    def __init__(self, T=10, n=2, extends=None, signed_distance_field=None):
        self.config_space_dim = n       # nb of dofs
        self.T = T                      # time steps
        self.dt = 0.1                   # sample rate
        self.trajectory_space_dim = (self.config_space_dim * (self.T + 2))
        self.extends = extends
        self.q_final = np.ones(2)
        self.workspace = None
        self.objective = None

        # We only need the signed distance field
        # to create a trajectory optimization problem
        self.sdf = signed_distance_field
        if self.sdf is None:
            self.create_workspace()

        # Here we combine everything to make an objective
        # TODO see why n==1 doesn't work...
        if self.config_space_dim > 1:
            self.create_objective()

    def center_of_clique_map(self):
        """ x_{t} """
        dim = self.config_space_dim
        return RangeSubspaceMap(dim * 3, range(dim, 2 * dim))

    def right_of_clique_map(self):
        """ x_{t+1} ; x_{t} """
        dim = self.config_space_dim
        return RangeSubspaceMap(dim * 3, range(dim, 3 * dim))

    def obstacle_cost_map(self):
        dim = self.config_space_dim
        phi = ObstaclePotential2D(self.sdf)
        return Compose(RangeSubspaceMap(3, [0]), phi)

    def cost(self, trajectory):
        """ compute sum of acceleration """
        return self.objective.forward(trajectory.x())

    def create_workspace(self):
        self.workspace = Workspace()
        self.workspace.obstacles.append(Circle(np.array([0.4, 0.2]), .2))
        self.workspace.obstacles.append(Circle(np.array([0.0, 0.2]), .1))
        self.sdf = SignedDistanceWorkspaceMap(self.workspace)

    def create_smoothness_metric(self):
        a = FiniteDifferencesAcceleration(1, 1).a()
        K_dof = np.matrix(np.zeros((self.T + 2, self.T + 2)))
        for i in range(0, self.T + 2):
            print i
            if i == 0:
                K_dof[i, i:i + 2] = a[0, 1:3]
            elif i == self.T + 1:
                K_dof[i, i - 1:i + 1] = a[0, 0:2]
            elif i > 0:
                K_dof[i, i - 1:i + 2] = a

        A_dof = K_dof.transpose() * K_dof
        print K_dof
        print A_dof

        # represented in the form :  \xi = [q_0 ; q_1; ... ; q_2]
        K_full = np.matrix(np.zeros((
            self.config_space_dim * (self.T + 2),
            self.config_space_dim * (self.T + 2))))
        for dof in range(0, self.config_space_dim):
            for (i, j), K_ij in np.ndenumerate(K_full):
                id_row = i * self.config_space_dim + dof
                id_col = j * self.config_space_dim + dof
                if id_row < K_full.shape[0] and id_col < K_full.shape[1]:
                    K_full[id_row, id_col] = K_dof[i, j]
        print K_full
        print K_full.shape
        A = K_full.transpose() * K_full
        return A

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
            Compose(
                FiniteDifferencesVelocity(self.config_space_dim, self.dt),
                self.right_of_clique_map())
        )
        isometric_obstacle_cost = ProductFunction(
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
