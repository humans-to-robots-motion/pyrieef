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

import test_common_imports
from motion.trajectory import *
from motion.cost_terms import *
from optimization.optimization import *


def CostSpace2D:
    def __init__(self):
        T = 30  # time steps
        self.config_space_dim = 2
        self.trajectory_space_dim = (self.config_space_dim * (T + 2))
        self.objective = CliquesFunctionNetwork(self.trajectory_space_dim)
        nb_cliques = self.objective.nb_cliques()
        acceration = FiniteDifferencesAcceleration(self.config_space_dim)
        SquaredNorm()
        self.objective.register_function_for_clique

