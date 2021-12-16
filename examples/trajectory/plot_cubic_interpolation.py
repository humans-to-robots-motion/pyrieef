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
#                                        Jim Mainprice on Thursday Dec 16 2021

import demos_common_imports
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from pyrieef.geometry.differentiable_geometry import DifferentiableMap


class CubicInterpolator(DifferentiableMap):

    """
    Interpolate waypoints using Cubic Catmull–Rom splines

        Details:
            https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    """

    def __init__(self, T, dt=1.):
        self._T = T
        self._dt = dt

        self._A = np.empty((4, 4))
        self._A[0, :] = [0, 1, 0, 0]
        self._A[1, :] = [-.5, 0, .5, 0]
        self._A[2, :] = [1, -2.5, 2, -.5]
        self._A[3, :] = [-.5, 1.5, -1.5, .5]

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return self._T + 4

    def forward(self, x):

        # retrieve time along spline
        t = x[0]

        # s is over the [0, 1] interval
        ds = math.fmod(t / self._dt, self._T)
        si = math.floor(ds)
        ds -= si

        # calculatre spline coefficents
        a = self._A @ x[si+1:si+5]

        # return spline value at ds
        return a[3] * (ds ** 3) + a[2] * (ds ** 2) + a[1] * ds + a[0]


T = 10
dt = 1.
epsilon = 1e-4

interpolator = CubicInterpolator(T, dt)

# setup time indices
time_index_1 = np.linspace(-dt, (T + 1) * dt, T + 3)
time_index_2 = np.linspace(epsilon, T * dt - epsilon, 100)

# test with sinusoid function
waypoints = np.sin(time_index_1)

# calculate all interpolated quatities
interpolated_configurations = []
interpolated_velocities = []
interpolated_derivative = [None] * len(waypoints)
for k in range(len(waypoints)):
    interpolated_derivative[k] = []

for t in time_index_2:
    x = np.append(t, waypoints)
    interpolated_configurations.append(interpolator(x))

    dx = interpolator.gradient(x)
    interpolated_velocities.append(dx[0])
    for k in range(len(waypoints)):
        interpolated_derivative[k].append(dx[k + 1])

# plot everything using subplots
fig, axs = plt.subplots(1 + len(waypoints), 1)
axs[0].plot(time_index_2, interpolated_configurations)
axs[0].plot(time_index_1, waypoints, 'o')
axs[0].plot(time_index_1, waypoints, '--')
axs[0].legend(['Position'])
for k in range(len(waypoints)):
    axs[k + 1].plot(time_index_2, interpolated_derivative[k])
    axs[k + 1].legend(['Waypoint' + str(k)])
plt.show()
