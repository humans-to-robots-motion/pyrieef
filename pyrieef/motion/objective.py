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

from __init__ import *
from motion.trajectory import *
from motion.cost_terms import *
from optimization.optimization import *
from geometry.differentiable_geometry import *
from geometry.workspace import *


class MotionOptimization2DCostMap:

    def __init__(self, T=10, n=2,
                 extends=None,
                 signed_distance_field=None,
                 q_init=None,
                 q_goal=None):
        self.config_space_dim = n       # nb of dofs
        self.T = T                      # time steps
        self.dt = 0.1                   # sample rate
        self.trajectory_space_dim = (self.config_space_dim * (self.T + 2))
        self.extends = extends

        self.workspace = None
        self.objective = None

        self.q_goal = q_goal if q_goal is not None else .3 * np.ones(2)
        self.q_init = q_init if q_init is not None else np.zeros(2)

        self._eta = 10.
        self._obstacle_scalar = 0.1
        self._init_potential_scalar = 10000000.
        self._term_potential_scalar = 10000000.
        # self._init_potential_scalar = 0.0
        # self._term_potential_scalar = 0.0
        self._smoothness_scalar = 25000.

        # We only need the signed distance field
        # to create a trajectory optimization problem
        self.sdf = signed_distance_field
        if self.sdf is None:
            self.create_workspace()

        # Here we combine everything to make an objective
        # TODO see why n==1 doesn't work...
        if self.config_space_dim > 1:
            # Creates a differentiable clique function.
            self.create_clique_network()
            self.add_all_terms()

        # Create metric for natural gradient descent
        self.create_smoothness_metric()

    def set_eta(self, eta):
        self._eta = eta

    def obstacle_cost_map(self):
        return SimplePotential2D(self.sdf)
        # return CostGridPotential2D(self.sdf,
        #                            alpha=10.,
        #                            margin=.03,
        #                            offset=1.)

    def cost(self, trajectory):
        """ compute sum of acceleration """
        return self.objective.forward(trajectory.x())

    def create_workspace(self):
        self.workspace = Workspace()
        self.workspace.obstacles.append(Circle(np.array([0.2, .15]), .1))
        self.workspace.obstacles.append(Circle(np.array([-.1, .15]), .1))
        self.sdf = SignedDistanceWorkspaceMap(self.workspace)

    def create_smoothness_metric(self):
        a = FiniteDifferencesAcceleration(1, self.dt).a()
        # print "a : "
        # print a
        K_dof = np.matrix(np.zeros((self.T + 2, self.T + 2)))
        for i in range(0, self.T + 2):
            if i == 0:
                K_dof[i, i:i + 2] = a[0, 1:3]
                K_dof[i, i] *= 1000  # No variance at end points
            elif i == self.T + 1:
                K_dof[i, i - 1:i + 1] = a[0, 0:2]
                K_dof[i, i] *= 1000  # No variance at end points
            elif i > 0:
                K_dof[i, i - 1:i + 2] = a
        A_dof = K_dof.transpose() * K_dof
        # print K_dof
        # print A_dof

        # represented in the form :  \xi = [q_0 ; q_1; ... ; q_2]
        K_full = np.matrix(np.zeros((
            self.config_space_dim * (self.T + 2),
            self.config_space_dim * (self.T + 2))))
        for dof in range(self.config_space_dim):
            for (i, j), K_ij in np.ndenumerate(K_dof):
                id_row = i * self.config_space_dim + dof
                id_col = j * self.config_space_dim + dof
                if id_row < K_full.shape[0] and id_col < K_full.shape[1]:
                    K_full[id_row, id_col] = K_ij
        # print K_full
        # print K_full.shape
        A = K_full.transpose() * K_full
        self.metric = A
        return A

    def add_init_and_terminal_terms(self):

        initial_potential = Pullback(
            SquaredNorm(self.q_init),
            self.objective.left_most_of_clique_map())
        self.objective.register_function_for_clique(
            0, Scale(initial_potential, self._init_potential_scalar))

        terminal_potential = Pullback(
            SquaredNorm(self.q_goal),
            self.objective.center_of_clique_map())
        self.objective.register_function_last_clique(
            Scale(terminal_potential, self._term_potential_scalar))

    def add_smoothness_terms(self, deriv_order=2):

        if deriv_order == 1:
            derivative = Pullback(
                FiniteDifferencesVelocity(self.config_space_dim, self.dt),
                self.objective.right_of_clique_map())
        elif deriv_order == 2:
            derivative = FiniteDifferencesAcceleration(
                self.config_space_dim, self.dt)
        else:
            raise ValueError("deriv_order ({}) not suported".format(
                deriv_order))
        self.objective.register_function_for_all_cliques(
            Scale(
                Pullback(SquaredNorm(np.zeros(self.config_space_dim)),
                         derivative), self._smoothness_scalar))

    def add_obstacle_terms(self, geodesic=False):

        if geodesic:
            pass
        else:
            obstacle_potential = Pullback(
                self.obstacle_cost_map(),
                self.objective.center_of_clique_map())
            squared_norm_vel = Pullback(
                SquaredNorm(np.zeros(self.config_space_dim)),
                Pullback(
                    FiniteDifferencesVelocity(self.config_space_dim, self.dt),
                    self.objective.right_of_clique_map())
            )
            isometric_obstacle_cost = ProductFunction(
                obstacle_potential,
                squared_norm_vel)

            self.objective.register_function_for_all_cliques(
                Scale(isometric_obstacle_cost, self._obstacle_scalar))

    def create_clique_network(self):
        """ resets the objective """
        self.objective = CliquesFunctionNetwork(
            self.trajectory_space_dim,
            self.config_space_dim)

    def add_all_terms(self):
        self.add_init_and_terminal_terms()
        self.add_smoothness_terms()
        self.add_obstacle_terms()
        # print self.objective.nb_cliques()

    def optimize(self, q_init, nb_steps=100, trajectory=None):
        if trajectory is None:
            trajectory = linear_interpolation_trajectory(
                q_init, self.q_goal, self.T)
        optimizer = NaturalGradientDescent(self.objective, self.metric)
        optimizer.set_eta(self._eta)
        xi = trajectory.x()
        dist = float("inf")
        for i in range(nb_steps):
            xi = optimizer.one_step(xi)
            trajectory.x()[:] = xi
            dist = np.linalg.norm(
                trajectory.final_configuration() - self.q_goal)
            # print "dist[{}] : {}, objective : {}, gnorm {}".format(
            #     i, dist, optimizer.objective(xi),
            #     np.linalg.norm(optimizer.gradient(xi)))
        return [dist < 1.e-3,
                trajectory,
                optimizer.gradient(xi),
                optimizer.delta(xi)]
