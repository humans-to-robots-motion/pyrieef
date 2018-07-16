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

from __future__ import print_function
from common_imports import *
from geometry.differentiable_geometry import *


class FunctionNetwork(DifferentiableMap):

    """ Base class to implement a function network
        It allows to register functions and evaluates
        f(x_0, x_1, ... ) = \sum_i f_i(x_0, x_1, ...)
    """

    def __init__(self):
        self._functions = []

    def output_dimension(self):
        return 1

    def input_dimension(self):
        raise NotImplementedError()

    def forward(self, x):
        value = 0.
        for f in range(self._functions):
            value += f.forward(x)
        return value

    def add_function(self, f):
        self._functions.append(f)


class CliquesFunctionNetwork(FunctionNetwork):
    """ Base class to implement a function network
        It allows to register functions and evaluates
        f(x_{i-1}, x_i, x_{i+1}) = \sum_i f_i(x_{i-1}, x_i, x_{i+1})
        """

    def __init__(self, dim):
        FunctionNetwork.__init__(self)
        self._input_size = dim
        self._clique_size = 3
        self._nb_cliques = self._input_size - self._clique_size + 1
        self._functions = self._nb_cliques * [None]
        for i in range(self._nb_cliques):
            self._functions[i] = []

    def input_dimension(self):
        raise input_size_

    def nb_cliques(self):
        return self._nb_cliques

    def foward(self, x):

        cliques = self.all_cliques(x)
        assert len(cliques) == len(x)
        assert len(cliques) == self._nb_cliques

        # We call over all subfunctions in each clique
        value = 0.
        for i, c in enumerate(cliques):
            for f in self._functions[i]:
                value += f.forward(c)
        return value

    def all_cliques(self, x):
        """ returns a dictionary of cliques """
        print("x : ", len(x))
        print("clique size : ", self._clique_size)
        cliques = [x[i:self._clique_size + i]
                   for i in range(self._nb_cliques)]
        return cliques

    def register_function_for_clique(self, i, f):
        """ Register function f for clique i """
        self._functions[i].append(f)

    def register_function_for_all_cliques(self, f):
        """ Register function f """
        for i in range(self._nb_cliques):
            self._functions[i].append(f)


class Trajectory:

    def __init__(self, T=0, n=2):
        assert T > 0 and n > 0
        self._n = n
        self._T = T
        self._x = np.zeros(n * (T + 2))

    def __str__(self):
        ss = ""
        ss += " - n : " + str(self._n) + "\n"
        ss += " - T : " + str(self._T) + "\n"
        ss += " - x : " + str(self._x)
        return ss

    def final_configuration(self):
        return self.configuration(self._T)

    def configuration(self, i):
        """ To get a mutable part : 
            traj.configuration(3)[:] = np.ones(2)
        """
        beg_idx = self._n * i
        end_idx = self._n * (i + 1)
        return self._x[beg_idx:end_idx]

    def clique(self, i):
        assert i > 0
        beg_idx = self._n * (i - 1)
        end_idx = self._n * (i + 2)
        return self._x[beg_idx:end_idx]

    def set(self, x):
        assert x.shape[0] == self._n * (2 + self._T)
        self._x = x
