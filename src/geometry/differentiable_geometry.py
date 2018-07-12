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
# Jim Mainprice on Sunday June 17 2017

import numpy as np
import copy
from abc import abstractmethod


class DifferentiableMap:

    @abstractmethod
    def output_dimension(self):
        raise NotImplementedError()

    @abstractmethod
    def input_dimension(self):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, q):
        raise NotImplementedError()

    def __call__(self, q):
        """ Method called when call object """
        return self.forward(q)

    def gradient(self, q):
        """ Convienience function to get numpy 
            gradients in the same shape as the input vector
            for addition and substraction, of course gradients are
            only availables if the output dimension is one."""
        assert self.output_dimension() == 1
        return np.array(self.jacobian(q)).reshape(self.input_dimension())

    def jacobian(self, q):
        """ by default the method returns the finite difference jacobian.
            WARNING the object returned by this function is a numpy matrix.
            Thhe Jacobian matrix is allways a numpy matrix object."""
        return finite_difference_jacobian(self, q)

    def evaluate(self, q):
        """ Evaluates the map and jacobian simultaneously. The default
            implementation simply calls both forward and Getjacobian()
            separately but overriding this method can make the evaluation 
            more efficient """
        x = self.forward(q)
        J = self.jacobian(q)
        return [x, J]

    def pullback_gradient(self, q, g):
        """ If g is the gradient of a function c(x) defined on the range space
        of this map so that g = d/dx c(x)), then the gradient of the "pullback"
        function c(phi(q)) is d/dq c(phi(q)) = J'g. This method computes and
        returns this "pullback gradient" J'g.
        WARNING: The return is will be of the same type as g:
            - if g is an array then the function returns an array
            - if g is a signe collumn matrix then it returns a 
                numpy matrix object
        """
        J = self.jacobian(q)
        return np.dot(J.transpose(), g)


class PullbackFunction(DifferentiableMap):

    def __init__(self, phi, f):
        self.phi = phi
        self.f = f

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return self.phi.input_dimension()

    def forward(self, q):
        return self.f(self.phi(q))

    def evaluate(self, q):
        x = self.phi(q)
        [x, f_g] = self.f.evaluate(x)
        g = self.phi.pullback_gradient(q, f_g)
        return [x, g]


class SquaredNorm(DifferentiableMap):
    """ Simple squared norm : f(x) = |x|^2 """

    def __init__(self, x_0):
        self.x_0 = x_0

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return self.x_0.size

    def forward(self, x):
        delta_x = x - self.x_0
        return 0.5 * np.dot(delta_x, delta_x)

    def jacobian(self, x):
        delta_x = x - self.x_0
        return np.matrix(delta_x)


class IdentityMap(DifferentiableMap):
    """Simple identity map : f(x) = x"""

    def __init__(self, n):
        self.dim = n

    def output_dimension(self):
        return self.dim

    def input_dimension(self):
        return self.dim

    def forward(self, x):
        return x

    def jacobian(self, q):
        return np.matrix(np.eye(self.dim))


class AffineMap(DifferentiableMap):
    """Simple map of the form: f(x) = ax + b"""

    def __init__(self, a, b):
        self.a_ = np.matrix(a)  # Make sure that a is matrix
        self.b_ = np.matrix(b).transpose()

    def output_dimension(self):
        return self.b_.shape[0]

    def input_dimension(self):
        return self.b_.shape[0]

    def forward(self, x):
        x_tmp = x.reshape(self.b_.shape)
        y = self.a_ * x_tmp + self.b_
        return y.reshape(self.output_dimension())

    def jacobian(self, x):
        return self.a_


def finite_difference_jacobian(f, q):
    """ Takes an object f that has a forward method returning
    a numpy array when querried. """
    dt = 1e-4
    dt_half = dt / 2.
    J = np.zeros((
        f.output_dimension(), f.input_dimension()))
    for j in range(q.size):
        q_up = copy.deepcopy(q)
        q_up[j] += dt_half
        x_up = f.forward(q_up)
        q_down = copy.deepcopy(q)
        q_down[j] -= dt_half
        x_down = f.forward(q_down)
        J[:, j] = (x_up - x_down) / dt
    return np.matrix(J)
