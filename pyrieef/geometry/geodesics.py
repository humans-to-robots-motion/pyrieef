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
from .charge_simulation import *
from .utils import *


def ComputeTensor(simulation, p, delta=0.001):
    # print "p : ", p
    p_0 = p.item(0)
    p_1 = p.item(1)
    # print "p_0 : ", p_0
    # print "p_1 : ", p_1
    gamma = 10.  # 30.
    rho = simulation.PotentialCausedByObject(np.array([p_0, p_1]))
    rho_x = simulation.PotentialCausedByObject(np.array([p_0 + delta, p_1]))
    rho_y = simulation.PotentialCausedByObject(np.array([p_0, p_1 + delta]))
    J = np.matrix(np.zeros((3, 2)))
    J[0, 0] = gamma * (rho_x - rho) / delta
    J[0, 1] = gamma * (rho_y - rho) / delta
    J[1, 0] = 1.
    J[2, 1] = 1.
    # print "J : "
    # print J
    return J.T * J


def ComputeNaturalGradient(simulation, x_1, x_2):
    # print "x_init : "
    # print x_1
    # print "x_goal : "
    # print x_2
    x_init = np.matrix(x_1).T
    x_goal = np.matrix(x_2).T
    x_tmp = x_init
    eta = 0.005
    line = []
    for i in range(1000):
        g = ComputeTensor(simulation, x_tmp)
        d_x = np.linalg.inv(g) * (x_goal - x_tmp)
        x_new = x_tmp + eta * normalize(d_x)
        line.append(np.array([x_new.item(0), x_new.item(1)]))
        if np.linalg.norm(x_new - x_goal) <= eta:
            line.append([x_goal.item(0), x_goal.item(1)])
            # print "End at : ", i
            break
        x_tmp = x_new
    return np.array(line)


def ComputeInterpolationGeodescis(simulation, x_1, x_2):
    x_init = np.matrix(x_1).T
    x_goal = np.matrix(x_2).T
    rho_init = simulation.PotentialCausedByObject(x_init)
    rho_goal = simulation.PotentialCausedByObject(x_goal)
    x_tmp = x_init
    line = []
    eta = 0.005
    for alpha in np.linspace(0., 1., 1000):
        p_new = ((1. - alpha) * np.array(
            [x_init.item(0), x_init.item(1), rho_init]) +
            alpha * np.array(
            [x_goal.item(0), x_goal.item(1), rho_goal]))
        print(p_new)
        x_new = np.matrix([p_new.item(0), p_new.item(1)]).T
        line.append(np.array([x_new.item(0), x_new.item(1)]))
        if np.linalg.norm(x_new - x_goal) <= eta:
            line.append([x_goal.item(0), x_goal.item(1)])
            break
        x_tmp = x_new
    return np.array(line)


def ComputeInitialVelocityGeodescis(simulation, x, x_dot):
    x_init = np.matrix(x).T
    x_init_dot = np.matrix(x_dot).T
    x_tmp = x_init
    d_x = x_init_dot
    line = []
    eta = 0.01
    for i in range(200):
        g = ComputeTensor(simulation, x_tmp)
        x_new = x_tmp + eta * normalize(np.linalg.inv(g) * d_x)
        line.append(np.array([x_new.item(0), x_new.item(1)]))
        d_x = x_new - x_tmp
        x_tmp = x_new
    return np.array(line)


def ComputeGeodesic(simulation, x_1, x_2):
    return ComputeNaturalGradient(simulation, x_1, x_2)
