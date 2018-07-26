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

    def __init__(self, input_dimension, clique_dimension):
        FunctionNetwork.__init__(self)
        self._input_size = input_dimension
        self._clique_size = 3 * clique_dimension
        self._nb_cliques = self._input_size - self._clique_size + 1
        self._functions = self._nb_cliques * [None]
        for i in range(self._nb_cliques):
            self._functions[i] = []

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return self._input_size

    def nb_cliques(self):
        return self._nb_cliques

    def forward(self, x):
        """ We call over all subfunctions in each clique"""
        value = 0.
        for t, x_t in enumerate(self.all_cliques(x)):
            # print("x_c[{}] : {}".format(t, x_t))
            for f in self._functions[t]:
                value += f.forward(x_t)
        return value

    def jacobian(self, x):
        """ 
            The jacboian matrix is of dimension m x n
                m (rows) : output size
                n (cols) : input size
            which can also be viewed becase the first order Taylor expansion
            of any differentiable map is f(x) = f(x_0) + J(x_0)_f x,
            where x is a collumn vector.
            The sub jacobian of the maps are the sum of clique jacobians
            each clique function f : R^n -> R, where n is the clique size.
        """
        J = np.matrix(np.zeros((
            self.output_dimension(),
            self.input_dimension())))
        for t, x_t in enumerate(self.all_cliques(x)):
            for f in self._functions[t]:
                J[0, t:self._clique_size + t] += f.jacobian(x_t)
        return J

    def all_cliques(self, x):
        """ returns a dictionary of cliques """
        # print("x : ", len(x))
        # print("clique size : ", self._clique_size)
        n = self._clique_size
        cliques = [x[t:n + t] for t in range(self._nb_cliques)]
        assert len(cliques) == self._nb_cliques
        return cliques

    def register_function_for_clique(self, t, f):
        """ Register function f for clique i """
        assert f.input_dimension() == self._clique_size
        self._functions[t].append(f)

    def register_function_for_all_cliques(self, f):
        """ Register function f """
        assert f.input_dimension() == self._clique_size
        for t in range(self._nb_cliques):
            self._functions[t].append(f)

    def register_function_last_clique(self, f):
        """ Register function f """
        assert f.input_dimension() == self._clique_size
        T = self._nb_cliques - 1
        self._functions[T].append(f)


class Trajectory:
    """ 
        Implement a trajectory as a single vector of 
        configuration, returns cliques of configurations
    """

    def __init__(self, T=0, n=2):
        assert T > 0 and n > 0
        self._n = n
        self._T = T
        self._x = np.zeros(n * (T + 2))

    def __str__(self):
        ss = ""
        ss += " - n : " + str(self._n) + "\n"
        ss += " - T : " + str(self._T) + "\n"
        ss += " - x : \n" + str(self._x) + "\n"
        ss += " - x.shape : " + str(self._x.shape)
        return ss

    def x(self):
        return self._x

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


def linear_interpolation_trajectory(q_init, q_goal, T):
    assert q_init.size == q_goal.size
    trajectory = Trajectory(T, q_init.size)
    for i in range(T + 2):
        alpha = min(float(i) / float(T), 1)
        trajectory.configuration(i)[:] = (1 - alpha) * q_init + alpha * q_goal
    return trajectory


def no_motion_trajectory(q_init, T):
    trajectory = Trajectory(T, q_init.size)
    for i in range(T + 2):
        trajectory.configuration(i)[:] = q_init
    return trajectory
