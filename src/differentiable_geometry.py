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
    def Forward(self, q):
        raise NotImplementedError()

    # Method called wehn call object
    def __call__(self, q):
        return self.Forward(q)

    # by default the method returns the finite difference Jacobian.
    # WARNING the object returned by this function is a numpy matrix.
    def Jacobian(self, q):
        return GetFiniteDifferenceJacobian(self, q)

    # Evaluate the map and Jacobian simultaneously. The default
    # implementation simply calls both Forward and GetJacobian()
    # separately but overriding this method can make the evaluation more
    # efficient
    def Evaluate(self, q):
        x = self.Forward(q)
        J = self.Jacobian(q)
        return [x, J]

    # If g is the gradient of a function c(x) defined on the range space
    # of this map so that g = d/dx c(x)), then the gradient of the "pullback"
    # function c(phi(q)) is d/dq c(phi(q)) = J'g. This method computes and
    # returns this "pullback gradient" J'g.
    # WARNING: The return is will be of the same type as g:
    # - if g is an array then the function returns an array
    # - if g is a signe collumn matrix then it returns a numpy matrix object
    def PullbackGradient(self, q, g):
        J = self.Jacobian(q)
        return np.dot(J.transpose(), g)


class PullbackFunction(DifferentiableMap):

    def __init__(self, phi, f):
        self.phi = phi
        self.f = f

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return self.phi.input_dimension()

    def Forward(self, q):
        return self.f(self.phi(q))

    def Evaluate(self, q):
        x = self.phi(q)
        [x, f_g] = self.f.Evaluate(x)
        g = self.phi.PullbackGradient(q, f_g)
        return [x, g]

# Simple squared norm.


class SquaredNorm(DifferentiableMap):

    def __init__(self, x_0):
        self.x_0 = x_0

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return self.x_0.size

    def Forward(self, x):
        delta_x = x - self.x_0
        return 0.5 * np.dot(delta_x, delta_x)

    def Jacobian(self, x):
        delta_x = x - self.x_0
        return np.matrix(delta_x)


# Simple squared norm.
class IdentityMap(DifferentiableMap):

    def __init__(self, n):
        self.dim = n

    def output_dimension(self):
        return self.dim

    def input_dimension(self):
        return self.dim

    def Forward(self, x):
        return x

    def Jacobian(self, q):
        return np.matrix(np.eye(self.dim))


# Takes an object f that has a Foward method returning
# a numpy array when querried.
def GetFiniteDifferenceJacobian(f, q):
    dt = 1e-4
    dt_half = dt / 2.
    J = np.zeros((
        f.output_dimension(), f.input_dimension()))
    for j in range(q.size):
        q_up = copy.deepcopy(q)
        q_up[j] += dt_half
        x_up = f.Forward(q_up)
        q_down = copy.deepcopy(q)
        q_down[j] -= dt_half
        x_down = f.Forward(q_down)
        J[:, j] = (x_up - x_down) / dt
    return np.matrix(J)
