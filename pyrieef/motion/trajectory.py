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

    def __init__(self, input_dimension, clique_element_dim):
        FunctionNetwork.__init__(self)
        self._input_size = input_dimension
        self._nb_clique_elements = 3
        self._clique_element_dim = clique_element_dim
        self._clique_dim = self._nb_clique_elements * clique_element_dim
        self._nb_cliques = self._input_size - self._clique_dim + 1
        self._functions = self._nb_cliques * [None]
        for i in range(self._nb_cliques):
            self._functions[i] = []

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return self._input_size

    def nb_cliques(self):
        return self._nb_cliques

    def function_on_clique(self, t, x_t):
        """ calls all functions on one clique """
        value = 0.
        for f in self._functions[t]:
            value += f.forward(x_t)
        return value

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
                assert f.output_dimension() == self.output_dimension()
                J[0, t:self._clique_dim + t] += f.jacobian(x_t)
        return J

    def hessian(self, x):
        """
            The hessian matrix is of dimension m x m
                m (rows) : input size
                m (cols) : input size
        """
        H = np.matrix(np.zeros((
            self.input_dimension(),
            self.input_dimension())))
        for t, x_t in enumerate(self.all_cliques(x)):
            for f in self._functions[t]:
                dim = self._clique_dim

                H[t:dim + t, t:dim + t] += f.hessian(x_t)
        return H

    def all_cliques(self, x):
        """ returns a dictionary of cliques """
        # print("x : ", len(x))
        # print("clique size : ", self._clique_size)
        n = self._clique_dim
        cliques = [x[t:n + t] for t in range(self._nb_cliques)]
        assert len(cliques) == self._nb_cliques
        return cliques

    def register_function_for_clique(self, t, f):
        """ Register function f for clique i """
        assert f.input_dimension() == self._clique_dim
        self._functions[t].append(f)

    def register_function_for_all_cliques(self, f):
        """ Register function f """
        assert f.input_dimension() == self._clique_dim
        for t in range(self._nb_cliques):
            self._functions[t].append(f)

    def register_function_last_clique(self, f):
        """ Register function f """
        assert f.input_dimension() == self._clique_dim
        T = self._nb_cliques - 1
        self._functions[T].append(f)

    def left_most_of_clique_map(self):
        """ x_{t-1} """
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            range(dim))

    def center_of_clique_map(self):
        """ x_{t} """
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            range(dim, (self._nb_clique_elements - 1) * dim))

    def right_of_clique_map(self):
        """ x_{t} ; x_{t+1} """
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            range(dim, self._nb_clique_elements * dim))


class TrajectoryObjectiveFunction(DifferentiableMap):
    """ Wraps the active part of the Clique Function Network

        The idea is that the finite differences are approximated using
        cliques so for the first clique, these values are quite off.
        Since the first configuration is part of the problem statement
        we can take it out of the optimization and through away the gradient
        computed for that configuration.

        TODO Test...
        """

    def __init__(self, q_init, function_network):
        self._q_init = q_init
        self._n = q_init.size
        self._function_network = function_network

    def full_vector(self, x_active):
        assert x_active.size == (
            self._function_network.input_dimension() - self._n)
        x_full = np.array(self._function_network.input_dimension())
        x_full[0:self._n] = self._q_init
        x_full[self._n:] = x_active
        return x_full

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return self._function_network.input_dimension() - self._n

    def forward(self, x):
        x_full = self.full_vector(x)
        return self._function_network(x_full)

    def jacobian(self, x):
        x_full = self.full_vector(x)
        return self._function_network.jacobian(x_full)[self._n:]

    def hessian(self, x):
        x_full = self.full_vector(x)
        return self._function_network.hessian(x_full)[self._n:, self._n:]


class Trajectory:
    """
        Implement a trajectory as a single vector of configuration,
        returns cliques of configurations
        Note there is T active configuration in the trajectory
        indices 
                0 and T + 1 
            are not supposed to be active.
        """

    def __init__(self, T=0, n=2, q_init=None, x=None):
        assert n > 0
        if q_init is not None and x is not None:
            assert x.size % q_init.size == 0
            self._n = q_init.size
            self._T = (x.size / q_init.size) - 1
            self._x = np.zeros(self._n * (self._T + 2))
            self._x[0:self._n] = q_init
            self._x[self._n:] = x
        else:
            assert T > 0
            self._n = n
            self._T = T
            self._x = np.zeros(self._n * (self._T + 2))

    def __str__(self):
        ss = ""
        ss += " - n : " + str(self._n) + "\n"
        ss += " - T : " + str(self._T) + "\n"
        ss += " - x : \n" + str(self._x) + "\n"
        ss += " - x.shape : " + str(self._x.shape)
        return ss

    def T(self):
        return self._T

    def x(self):
        return self._x

    def active_segment(self):
        return self._x[self._n:]

    def initial_configuration(self):
        return self.configuration(0)

    def final_configuration(self):
        return self.configuration(self._T)

    def configuration(self, i):
        """  mutable : traj.configuration(3)[:] = np.ones(2) """
        assert i >= 0 and i <= (self._T + 1)
        beg_idx = self._n * i
        end_idx = self._n * (i + 1)
        return self._x[beg_idx:end_idx]

    def clique(self, i):
        """ returns a clique of 3 configurations """
        assert i >= 0 and i <= (self._T + 1)
        beg_idx = self._n * (i - 1)
        end_idx = self._n * (i + 2)
        return self._x[beg_idx:end_idx]

    def set(self, x):
        assert x.shape[0] == self._n * (self._T + 2)
        self._x = x


class ContinuousTrajectory(Trajectory):
    """ Implements a trajectory that can be continously interpolated """

    def configuration_at_parameter(self, s):
        """ The trajectory is indexed by s \in [0, 1] """
        d_param = s * self.length()
        q_prev = self.configuration(0)
        dist = 0.
        for i in range(1, self._T + 1):
            q_curr = self.configuration(i)
            d = np.linalg.norm(q_curr - q_prev)
            if d_param <= (dist + d):
                return self.interpolate(q_prev, q_curr, d_param - dist, d)
            dist += d
            q_prev = q_curr
        return None

    def length(self):
        """ length in configuration space """
        length = 0.
        q_prev = self.configuration(0)
        for i in range(1, self._T + 1):
            q_curr = self.configuration(i)
            length += np.linalg.norm(q_curr - q_prev)
            q_prev = q_curr
        return length

    @staticmethod
    def interpolate(q_1, q_2, d_param, dist):
        """ interpolate between configurations """
        alpha = min(d_param / dist, 1.)
        assert alpha >= 0 and alpha <= 1., "alpha : {}".format(alpha)
        return (1. - alpha) * q_1 + alpha * q_2


def linear_interpolation_trajectory(q_init, q_goal, T):
    assert q_init.size == q_goal.size
    trajectory = Trajectory(T, q_init.size)
    for i in range(T + 2):
        alpha = float(i) / float(T)
        trajectory.configuration(i)[:] = (1 - alpha) * q_init + alpha * q_goal
        print "config[{}] : {}".format(i, trajectory.configuration(i))
    return trajectory


def no_motion_trajectory(q_init, T):
    trajectory = Trajectory(T, q_init.size)
    for i in range(T + 2):
        trajectory.configuration(i)[:] = q_init
    return trajectory
