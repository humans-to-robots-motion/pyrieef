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

from .__init__ import *
from motion.trajectory import *
from motion.cost_terms import *
from optimization.optimization import *
from geometry.differentiable_geometry import *
from geometry.workspace import *
from scipy import optimize


class GeodesicObjective2D:

    def __init__(self, T=10, n=2,
                 q_init=None,
                 q_goal=None,
                 embedding=None):
        self.verbose = False
        self.config_space_dim = n       # nb of dofs
        self.T = T                      # time steps
        self.dt = 0.1                   # sample rate.
        self.embedding = embedding
        self._init_potential_scalar = 0.
        self._term_potential_scalar = 10000000.
        self._velocity_scalar = 1.

    def set_test_objective(self):
        """ This objective does not collide with the enviroment"""
        self.create_sdf_test_workspace()
        self.obstacle_potential_from_sdf()
        self.create_clique_network()
        self.create_objective()

    def add_init_and_terminal_terms(self):

        if self._init_potential_scalar > 0.:
            initial_potential = Pullback(
                SquaredNorm(self.q_init),
                self.function_network.left_most_of_clique_map())
            self.function_network.register_function_for_clique(
                0, Scale(initial_potential, self._init_potential_scalar))

        terminal_potential = Pullback(
            SquaredNorm(self.q_goal),
            self.function_network.center_of_clique_map())
        self.function_network.register_function_last_clique(
            Scale(terminal_potential, self._term_potential_scalar))

    def add_smoothness_terms(self):

        derivative = Pullback( 
            Pullback(
                SquaredNorm(np.zeros(2)),
                Pullback(
                    FiniteDifferencesVelocity(self.config_space_dim, self.dt)),
                    self.embedding),
            self.function_network.left_of_clique_map())

        self.function_network.register_function_for_all_cliques(
            Scale(derivative, self._velocity_scalar))

    def create_clique_network(self):
        self.add_init_and_terminal_terms()
        self.add_smoothness_terms(1)

        self.function_network = CliquesFunctionNetwork(
            self.trajectory_space_dim,
            self.config_space_dim)

        """ resets the objective """
        self.objective = TrajectoryObjectiveFunction(
            self.q_init, self.function_network)
