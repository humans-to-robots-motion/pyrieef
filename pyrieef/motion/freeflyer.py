#!/usr/bin/env python

# Copyright (c) 2020, University of Stuttgart
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
#                                      Jim Mainprice on Tuesday January 28 2020

from .__init__ import *
from motion.trajectory import *
from motion.cost_terms import *
from optimization.optimization import *
from geometry.differentiable_geometry import *
from geometry.workspace import *


class FreeflyerObjective:

    def __init__(self, T=30, n=3,
                 q_init=None,
                 q_goal=None,
                 embedding=None,
                 robot=None):
        self.verbose = False
        self.config_space_dim = n       # nb of dofs
        self.q_init = q_init            # start configuration
        self.q_goal = q_goal            # goal configuration
        self.T = T                      # time steps
        self.dt = 0.1                   # sample rate.
        self.trajectory_space_dim = (self.config_space_dim * (self.T + 2))
        self.embedding = embedding
        self._init_potential_scalar = 0.
        self._term_potential_scalar = 100000.
        self._velocity_scalar = 10.
        self._robot = robot

    def add_terminal_term(self):
        terminal_potential = Pullback(
            SquaredNorm(self.q_goal),
            self.function_network.center_of_clique_map())
        self.function_network.register_function_last_clique(
            Scale(terminal_potential, self._term_potential_scalar))

    def add_smoothness_terms(self):
        clique_l = Pullback(
            self._robot.keypoint_map(0),
            self.function_network.left_most_of_clique_map())
        clique_c = Pullback(
            self._robot.keypoint_map(0),
            self.function_network.center_of_clique_map())
        ws_map_l = Compose(self.embedding, clique_l)
        ws_map_c = Compose(self.embedding, clique_c)
        fd = FiniteDifferencesVelocity(
            self.embedding.output_dimension(),
            self.dt)
        ws_vel_map = Compose(fd, CombinedOutputMap([ws_map_l, ws_map_c]))
        geodesic_term = Pullback(
            SquaredNorm(np.zeros(ws_vel_map.output_dimension())), ws_vel_map)
        self.function_network.register_function_for_all_cliques(
            Scale(geodesic_term, self._velocity_scalar))

    def create_clique_network(self):
        self.function_network = CliquesFunctionNetwork(
            self.trajectory_space_dim,
            self.config_space_dim)
        self.add_terminal_term()
        self.add_smoothness_terms()
        self.objective = TrajectoryObjectiveFunction(
            self.q_init, self.function_network)
