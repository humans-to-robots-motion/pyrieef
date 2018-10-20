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

import numpy as np
import sys
import math
from .workspace import *
from .utils import *
from .differentiable_geometry import *
from scipy.special import lambertw
from abc import abstractmethod

# --------
# this is the scaling factor
# - gamma | x - r | ... Does not work...
# TODO look into this.


def alpha3_f(eta, r, gamma, x):
    return eta * np.exp(-gamma * (x - r))

# y = f(x) is the distance value it gives


def beta3_f(eta, r, gamma, x):
    return x - alpha_f(eta, r, gamma, x)


def beta3_inv_f(eta, r, gamma, y):
    l = lambertw(eta * gamma * np.exp(gamma * r - gamma * y)).real
    return l / gamma + y

# --------


def alpha2_f(eta, r, gamma, x):
    return eta / x

# y = f(x) is the distance value
# it gives


def beta2_f(eta, r, gamma, x):
    return x - alpha2_f(eta, r, gamma, x)


def beta2_inv_f(eta, r, gamma, y):
    return (y + np.sqrt(y ** 2 + 4. * eta)) * 0.5

# --------
# this is the scaling factor
# - gamma | x - r | ...


def alpha_f(eta, r, gamma, x):
    return eta * np.exp(-gamma * x + r)

# y = f(x) is the distance value it gives


def beta_f(eta, r, gamma, x):
    return x - alpha_f(eta, r, gamma, x)

# def beta_inv_f(eta, r, gamma, y):
#     return lambertw(gamma * eta * np.exp(-gamma * y)).real / gamma + y


def beta_inv_f(eta, r, gamma, y):
    l = lambertw(eta * gamma * np.exp(r - gamma * y)).real
    return l / gamma + y


class PlaneDiffeomoprhism(DifferentiableMap):

    def output_dimension(self):
        return 2

    def input_dimension(self):
        return 2

    @abstractmethod
    def inverse(self, y):
        raise NotImplementedError()


class AnalyticPlaneDiffeomoprhism(PlaneDiffeomoprhism):

    @abstractmethod
    def object(self):
        raise NotImplementedError()

# The polar coordinates r and phi can be converted to the Cartesian coordinates
# x and y by using the trigonometric functions sine and cosine


class PolarCoordinateSystem(AnalyticPlaneDiffeomoprhism):

    def __init__(self):
        self.circle = Circle()
        self.circle.radius = .1
        self.circle.origin = np.array([.0, .0])
        self.eta = self.circle.radius  # for the exp

    # Access the internal object.
    def object(self):
        return self.circle

    # Converts from Euclidean to Polar
    # p[0] : x
    # p[1] : y
    def forward(self, p):
        p_0 = p - self.circle.origin
        r = np.linalg.norm(p_0)
        phi = math.atan2(p_0[1], p_0[0])
        return np.array([r, phi])

    # Converts from Polar to Euclidean
    # p[0] : r
    # p[1] : phi
    def inverse(self, p):
        x = p[0] * math.cos(p[1]) + self.circle.origin[0]
        y = p[0] * math.sin(p[1]) + self.circle.origin[1]
        return np.array([x, y])


class ElectricCircle(AnalyticPlaneDiffeomoprhism):

    def __init__(self):
        self.circle = Circle()
        self.circle.radius = .1
        self.circle.origin = np.array([.1, -.1])
        self.eta = self.circle.radius  # for the exp

    # Access the internal object.
    def object(self):
        return self.circle

    # squishes points inside the circle
    def forward(self, x):
        # print "origin : ", self.origin
        x_center = x - self.circle.origin
        d_1 = np.linalg.norm(x_center)
        y = np.array([0., 0.])
        y[0] = math.pow(d_1, self.eta)
        y[1] = math.atan2(x_center[1], x_center[0])
        return y

    # maps them back outside of the circle
    def inverse(self, y):
        # print "origin : ", self.origin
        x = np.array([0., 0.])
        x_center = np.array([0., 0.])
        d_1 = math.pow(y[0], 1 / self.eta)
        x_center[0] = math.cos(y[1]) * d_1
        x_center[1] = math.sin(y[1]) * d_1
        x = x_center + self.circle.origin
        return x


class AnalyticEllipse(AnalyticPlaneDiffeomoprhism):

    def __init__(self):
        self.ellipse = Ellipse()
        self.radius = .1
        self.eta = self.radius  # for the exp
        # self.eta = .01 # for the 1/x
        self.gamma = 1.
        self.origin = np.array([.1, -.1])

    # Access the internal object.
    def object(self):
        return self.ellipse

    # To recover the distance scaling one should
    # pass the alpha and beta inverse functions.
    def set_alpha(self, a, b):
        self.alpha_ = a
        self.beta_inv_ = b

    # squishes points inside the circle
    def Deformationforward(self, x):
        x_center = x - self.origin
        d_1 = np.linalg.norm(x_center)
        alpha = self.alpha_(self.eta, self.radius, self.gamma, d_1)
        return alpha * normalize(x_center)

    # maps them back outside of the circle
    def Deformationinverse(self, y):
        # print "origin : ", self.origin
        y_center = y - self.origin
        d_2 = np.linalg.norm(y_center)
        d_1 = self.beta_inv_(self.eta, self.radius, self.gamma, d_2)
        alpha = d_1 - d_2
        return alpha * normalize(y_center)

    # squishes points inside the circle
    def forward(self, x):
        y = x - self.Deformationforward(x)
        return y

    # maps them back outside of the circle
    def inverse(self, y):
        x = y + self.Deformationinverse(y)
        return x


class AnalyticCircle(AnalyticPlaneDiffeomoprhism):

    def __init__(self):
        self.circle = Circle()
        self.circle.radius = .1
        self.circle.origin = np.array([.1, -.1])
        self.eta = self.circle.radius  # for the exp
        # self.eta = .01 # for the 1/x
        self.gamma = 1.
        self.set_alpha(alpha_f, beta_inv_f)

    # Access the internal object.
    def object(self):
        return self.circle

    # To recover the distance scaling one should
    # pass the alpha and beta inverse functions.
    def set_alpha(self, a, b):
        self.alpha_ = a
        self.beta_inv_ = b

    # squishes points inside the circle
    def Deformationforward(self, x):
        # print "origin : ", self.origin
        x_center = x - self.circle.origin
        # Retrieve radius when using beta exponential
        # d_1 = np.linalg.norm(x_center) - self.radius
        d_1 = np.linalg.norm(x_center)
        alpha = self.alpha_(self.eta, self.circle.radius, self.gamma, d_1)
        # d_2 = np.linalg.norm(y - self.origin)
        # print "d_1 (1) : " , d_1
        # # print "d_2 (1) : " , d_2
        # print "alpha (11) : " , alpha
        # # print "alpha (12) : " , (d_1 - d_2)
        # print "radius :", self.radius
        # print "origin : ", self.origin
        # print "1 : ", x_center
        # print "2 : ", x
        # print "norm : ", normalize(x_center)
        return alpha * normalize(x_center)

    # maps them back outside of the circle
    def Deformationinverse(self, y):
        # print "origin : ", self.origin
        y_center = y - self.circle.origin
        d_2 = np.linalg.norm(y_center)
        d_1 = self.beta_inv_(self.eta, self.circle.radius, self.gamma, d_2)
        alpha = d_1 - d_2
        # print "d_1 (2) : " , d_1
        # print "d_2 (2) : " , d_2
        # print "alpha (2) : ", alpha
        return alpha * normalize(y_center)

    # squishes points inside the circle
    def forward(self, x):
        y = x - self.Deformationforward(x)
        return y

    # maps them back outside of the circle
    def inverse(self, y):
        x = y + self.Deformationinverse(y)
        return x


class AnalyticMultiCircle(AnalyticPlaneDiffeomoprhism):

    def __init__(self, circles):
        self.circles_ = circles
        self.gamma = 20.

    def object(self):
        self.circles_

    def Additiveforward(self, x):
        dx = np.array([0., 0.])
        for i, obj in enumerate(self.circles_):
            dx += obj.Deformationforward(x)
        return x - dx

    def Additiveinverse(self, y):
        dy = np.array([0., 0.])
        for i, obj in enumerate(self.circles_):
            dy += obj.Deformationinverse(y)
        return y + dy

    # This activation function is implemented through
    # a softmax function.
    def GetActivation(self, i, x):
        part = 0.
        for circle in self.circles_:
            d = circle.object().dist_from_border(x)
            part += np.exp(-self.gamma * d)
        d = self.circles_[i].object().dist_from_border(x)
        return np.exp(-self.gamma * d) / part

    def OneCircle(self, i, x):
        x_center = x - self.circles_[i].circle.origin
        d_1 = np.linalg.norm(x_center)
        activation = self.GetActivation(i, x)
        alpha = (activation * self.circles_[i].eta *
                 np.exp(self.circles_[i].gamma *
                        (-d_1 + self.circles_[i].circle.radius)))
        return alpha * normalize(x_center)

    def forward(self, x):
        dx = np.array([0., 0.])
        for i, obj in enumerate(self.circles_):
            dx += self.OneCircle(i, x)
            activation = self.GetActivation(i, x)
            # print ("activation[", i ,"] : ", activation, " , ",
            #     self.circles_[i].origin, " dx :", dx)
        return x - dx


def InterpolationGeodescis(obj, x_1, x_2):
    line = []
    line_inter = []
    line.append(x_1)
    p_init = obj.forward(np.array(x_1))
    p_goal = obj.forward(np.array(x_2))
    for s in np.linspace(0., 1., 100):
        p = (1. - s) * p_init + s * p_goal
        x_new = obj.inverse(p)
        line.append(x_new)
        line_inter.append(p)
        if np.linalg.norm(x_new - x_2) <= 0.001:
            break
    return [np.array(line), np.array(line_inter)]


def NaturalGradientGeodescis(obj, x_1, x_2):
    x_init = np.matrix(x_1).T
    x_goal = np.matrix(x_2).T
    x_tmp = x_init
    eta = 0.01
    line = []
    for i in range(1000):
        # Compute tensor.
        J = obj.jacobian(np.array(x_tmp.T)[0])
        g = J.T * J
        # Implement the attractor derivative here directly
        # suposes that it's of the form |phi(q) - phi(q_goal)|^2
        # hence the addition of the J^T
        ridge = 0.
        d_x = np.linalg.inv(g + ridge * np.eye(2)) * J.T * (
            x_goal - x_tmp)
        x_new = x_tmp + eta * normalize(d_x)
        line.append(np.array([x_new.item(0), x_new.item(1)]))
        if np.linalg.norm(x_new - x_goal) <= eta:
            line.append([x_goal.item(0), x_goal.item(1)])
            print(("End at : ", i))
            break
        x_tmp = x_new
    return np.array(line)
