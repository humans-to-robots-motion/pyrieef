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

    def pullback_jacobian(self, q, J_f):
        """ If J is the jacobian of a function f(x), J_f = d/dx f(x)
            then the jacobian of the "pullback" of f defined on the
            range space of a map g (this map), f(g(q)) is
                    d/dq f(g(q)) = J_f(g(q)) J_g
            This method computes and
            returns this "pullback gradient" J_f (g(q)) J_g(q).
        WARNING: J_f is assumed to be a jacobian np.matrix object
        """
        return J_f * self.jacobian(q)


class Compose(DifferentiableMap):

    def __init__(self, f, g):
        """ Make sure the composition makes sense
            This function should be called pullback if we approxiate
            higher order (hessian) derivaties by pullback, here it's
            still computing the true 1st order derivative of the
            composition.

            f after g : f(g(x))

            """
        assert g.output_dimension() == f.input_dimension()
        self._f = f
        self._g = g

    def output_dimension(self):
        return self._f.output_dimension()

    def input_dimension(self):
        return self._g.input_dimension()

    def forward(self, q):
        return self._f(self._g(q))

    def jacobian(self, q):
        [y, J] = self.evaluate(q)
        return J

    def evaluate(self, q):
        """  d/dq f(g(q)) """
        x = self._g(q)
        [y, J_f] = self._f.evaluate(x)
        J = self._g.pullback_jacobian(q, J_f)
        return [y, J]


class QuadricFunction(DifferentiableMap):
    """ Here we implement a quadric funciton of the form:
        f(x) = x^T A x + bx + c """

    def __init__(self, a, b, c):
        assert a.shape[0] == a.shape[1]
        assert b.size == a.shape[1]
        self._a = np.matrix(a)
        self._b = np.matrix(b.reshape(b.size, 1))
        self._c = c
        self._symmetric = np.allclose(self._a, self._a.T, atol=1e-8)
        self._posdef = np.all(np.linalg.eigvals(self._a) > 0)

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return self._b.size

    def forward(self, x):
        x_tmp = np.matrix(x.reshape(self._b.size, 1))
        v = (0.5 *
             x_tmp.transpose() * self._a * x_tmp +
             self._b.transpose() * x_tmp +
             self._c)
        return v

    def jacobian(self, x):
        """ when the matrix is positive this can be simplified
            see matrix cookbook """
        x_tmp = np.matrix(x.reshape(self._b.size, 1))
        if self._symmetric and self._posdef:
            a_term = self._a.transpose() * x_tmp
        else:
            a_term = 0.5 * (self._a + self._a.transpose()) * x_tmp
        return (a_term + self._b).transpose()


class SquaredNorm(DifferentiableMap):
    """ Simple squared norm : f(x)= | x | ^2 """

    def __init__(self, x_0):
        self.x_0 = x_0

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return self.x_0.size

    def forward(self, x):
        delta_x = np.array(x).reshape(x.size) - self.x_0
        print "delta_x.shape", delta_x
        return 0.5 * np.dot(delta_x, delta_x)

    def jacobian(self, x):
        delta_x = x - self.x_0
        return np.matrix(delta_x)


class IdentityMap(DifferentiableMap):
    """Simple identity map : f(x)=x"""

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
    """Simple map of the form: f(x)=ax + b"""

    def __init__(self, a, b):
        self._a = np.matrix(a)  # Make sure that a is matrix
        self._b = np.matrix(b.reshape(b.size, 1))

    def output_dimension(self):
        return self._b.shape[0]

    def input_dimension(self):
        return self._a.shape[1]

    def forward(self, x):
        x_tmp = x.reshape(self.input_dimension(), 1)
        tmp = self._a * x_tmp
        y = tmp + self._b
        return y.reshape(self.output_dimension())

    def jacobian(self, x):
        return self._a


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
