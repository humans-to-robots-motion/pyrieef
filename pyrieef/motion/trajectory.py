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
from geometry.differentiable_geometry import *


class FunctionNetwork(DifferentiableMap):
    """
        Base class to implement a function network
        It registers functions and evaluates
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
    """
        Base class to implement a function network
        It allows to register functions and evaluates
        f(x_{i-1}, x_i, x_{i+1}) = \sum_i f_i(x_{i-1}, x_i, x_{i+1})
    """

    def __init__(self, input_dimension, clique_element_dim):
        FunctionNetwork.__init__(self)
        self._input_size = input_dimension
        self._nb_clique_elements = 3
        self._clique_element_dim = clique_element_dim
        self._clique_dim = self._nb_clique_elements * clique_element_dim
        self._nb_cliques = int(self._input_size / clique_element_dim - 2)
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
            each clique function f : R^dim -> R, where dim is the clique size.
        """
        J = np.matrix(np.zeros((
            self.output_dimension(),
            self.input_dimension())))
        for t, x_t in enumerate(self.all_cliques(x)):
            for f in self._functions[t]:
                assert f.output_dimension() == self.output_dimension()
                c_id = t * self._clique_element_dim
                J[0, c_id:c_id + self._clique_dim] += f.jacobian(x_t)
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
        dim = self._clique_dim
        for t, x_t in enumerate(self.all_cliques(x)):
            c_id = t * self._clique_element_dim
            for f in self._functions[t]:
                H[c_id:c_id + dim, c_id:c_id + dim] += f.hessian(x_t)
        return H

    def clique_value(self, t, x_t):
        """
        return the clique value
        TODO create a test using this function.
        """
        value = 0.
        for f in self._functions[t]:
            value += f.forward(x_t)
        return value

    def clique_jacobian(self, J, t):
        """
        return the clique jacobian
        J : the full jacobian
        """
        c_id = t * self._clique_element_dim
        return J[0, c_id:c_id + self._clique_dim]

    def clique_hessian(self, H, t):
        """
        return the clique hessian
        H : the full hessian
        """
        dim = self._clique_dim
        c_id = t * self._clique_element_dim
        return H[c_id:c_id + dim, c_id:c_id + dim]

    def all_cliques(self, x):
        """ returns a list of all cliques """
        n = self._clique_element_dim
        dim = self._clique_dim
        clique_begin_ids = list(range(0, n * self._nb_cliques, n))
        cliques = [x[c_id:c_id + dim] for c_id in clique_begin_ids]
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

    def center_of_clique_map(self):
        """ x_{t} """
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            list(range(dim, (self._nb_clique_elements - 1) * dim)))

    def right_most_of_clique_map(self):
        """ x_{t+1} """
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            list(range((self._nb_clique_elements - 1) * dim,
                       self._nb_clique_elements * dim)))

    def right_of_clique_map(self):
        """ x_{t} ; x_{t+1} """
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            list(range(dim, self._nb_clique_elements * dim)))

    def left_most_of_clique_map(self):
        """ x_{t-1} """
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            list(range(0, dim)))

    def left_of_clique_map(self):
        """ x_{t-1} ; x_{t} """
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            list(range(0, (self._nb_clique_elements - 1) * dim)))


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
        x_full = np.zeros(self._function_network.input_dimension())
        x_full[:self._n] = self._q_init
        x_full[self._n:] = x_active
        return x_full

    def output_dimension(self):
        return self._function_network.output_dimension()

    def input_dimension(self):
        return self._function_network.input_dimension() - self._n

    def forward(self, x):
        x_full = self.full_vector(x)
        return min(1e100, self._function_network(x_full))

    def jacobian(self, x):
        x_full = self.full_vector(x)
        return self._function_network.jacobian(x_full)[0, self._n:]

    def hessian(self, x):
        x_full = self.full_vector(x)
        H = self._function_network.hessian(x_full)[self._n:, self._n:]
        return np.array(H)


class Trajectory:
    """
        Implement a trajectory as a single vector of configuration,
        returns cliques of configurations
        Note there is T active configuration in the trajectory
        indices
                0 and T + 1
            are supposed to be inactive.
        """

    def __init__(self, T=0, n=2, q_init=None, x=None):
        assert n > 0
        if q_init is not None and x is not None:
            assert x.size % q_init.size == 0
            self._n = q_init.size
            self._T = int((x.size / q_init.size) - 1)
            self._x = np.zeros(self._n * (self._T + 2))
            self._x[:self._n] = q_init
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

    def n(self):
        return self._n

    def T(self):
        return self._T

    def x(self):
        return self._x

    def set(self, x):
        assert x.shape[0] == self._n * (self._T + 2)
        self._x = x.copy()

    def active_segment(self):
        """
        The active segment of the trajectory
        removes the first configuration on the trajectory
        """
        return self._x[self._n:]

    def initial_configuration(self):
        """ first configuration """
        return self.configuration(0)

    def final_configuration(self):
        """ last active configuration """
        return self.configuration(self._T)

    def configuration(self, i):
        """  mutable : traj.configuration(3)[:] = np.ones(2) """
        assert i >= 0 and i <= (self._T + 1)
        beg_idx = self._n * i
        end_idx = self._n * (i + 1)
        return self._x[beg_idx:end_idx]

    def velocity(self, i, dt):
        """
        returns velocity at index i
            WARNING It is not the same convention as for the clique
                    Here we suppose the velocity at q_init to be 0,
                    so the finite difference is left sided (q_t - q_t-1)/dt
                    This is different from the right sided version
                    (q_t+1 - q_t)/dt implemented in the cost term module.

            With left side FD we directly get the integration scheme:

                q_{t+1} = q_t + v_t * dt + a_t * dt^2

            where v_t and a_t are velocity and acceleration
            at index t, with v_0 = 0.
            """
        if i == 0:
            return np.zeros(self._n)
        q_i_1 = self.configuration(i - 1)
        q_i_2 = self.configuration(i)
        return (q_i_2 - q_i_1) / dt

    def acceleration(self, i, dt):
        """
        returns acceleration at index i
            Note that we suppose velocity at q_init to be 0 """
        id_init = 0 if i == 0 else i - 1
        q_i_0 = self.configuration(id_init)
        q_i_1 = self.configuration(i)
        q_i_2 = self.configuration(i + 1)
        return (q_i_2 - 2 * q_i_1 + q_i_0) / (dt**2)

    def state(self, i, dt):
        """ return a tuple of configuration and velocity at index i """
        q_t = self.configuration(i)
        v_t = self.velocity(i, dt)
        return np.hstack([q_t, v_t])

    def clique(self, i):
        """ returns a clique of 3 configurations """
        assert i >= 1 and i <= (self._T)
        beg_idx = self._n * (i - 1)
        end_idx = self._n * (i + 2)
        return self._x[beg_idx:end_idx]

    def list_configurations(self):
        """ returns a list of configurations """
        nb_config = self.T() + 1
        line = [None] * nb_config
        for t in range(nb_config):
            line[t] = self.configuration(t)
        return line

    def continuous_trajectory(self):
        """ returns an object of contunious type """
        trajectory = ContinuousTrajectory(self.T(), self.n())
        trajectory._x = self._x
        return trajectory


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
            if d_param <= (d + dist):
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
        # assert d_param / dist <= 1.
        alpha = min(d_param / dist, 1.)
        assert alpha >= 0 and alpha <= 1., "alpha : {}".format(alpha)
        return (1. - alpha) * q_1 + alpha * q_2


class ConstantAccelerationTrajectory(ContinuousTrajectory):
    """ Implements a trajectory that can be continously interpolated """

    def __init__(self, T=0, n=2, dt=0.1, q_init=None, x=None):
        Trajectory.__init__(self, T=T, n=n, q_init=q_init, x=x)
        self._dt = float(dt)

    def config_at_time(self, t):
        """ Get the id of the segment and then interpolate
            using quadric interpolation """
        alpha_t = t / self._dt
        s_id = int(alpha_t)
        s_t = alpha_t - float(s_id)
        return self._config_along_segment(s_id, s_t * self._dt)

    def _config_along_segment(self, s_id, t):
        """ Implements a quadric interpolation """
        assert t >= 0 and t <= self._dt
        if s_id == 0:
            i = 1
            t_0 = t
        else:
            i = s_id
            t_0 = t + self._dt
        q_t = self.configuration(i - 1)
        v_t = self.velocity(i, self._dt)
        a_t = self.acceleration(i, self._dt)
        return q_t + v_t * t_0 + .5 * a_t * t_0 * (t_0 - self._dt)


def linear_interpolation_trajectory(q_init, q_goal, T):
    assert q_init.size == q_goal.size
    trajectory = Trajectory(T, q_init.size)
    for i in range(T + 2):
        alpha = float(i) / float(T)
        trajectory.configuration(i)[:] = (1 - alpha) * q_init + alpha * q_goal
        # print "config[{}] : {}".format(i, trajectory.configuration(i))
    return trajectory


def no_motion_trajectory(q_init, T):
    trajectory = Trajectory(T, q_init.size)
    for i in range(T + 2):
        trajectory.configuration(i)[:] = q_init
    return trajectory
