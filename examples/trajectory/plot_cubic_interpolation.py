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

    """ Interpolate waypoints using a cubic spline  """

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

        t = x[0]
        waypoints = x[1:]

        ds = math.fmod(t / self._dt, self._T)
        si = math.floor(ds)
        ds -= si

        p = [0, 0, 0, 0]
        p[0] = waypoints[si]
        p[1] = waypoints[si + 1]
        p[2] = waypoints[si + 2]
        p[3] = waypoints[si + 3]

        coef =  self._A @ p

        dspow2 = ds ** 2
        dspow3 = dspow2 * ds

        return coef[3] * dspow3 + coef[2] * dspow2 + coef[1] * ds + coef[0]

        # TODO this formula is wrong...
        # return p[1] + \
        #     0.5 * (p[2] - p[0]) * ds + \
        #      (1. * p[0] - -2.5 * p[1] + 4. * p[2] - p[3]) * dspow2 + \
        #      (-p[0] + 3. * p[1] - 3 * p[2] + p[3]) * dspow3


T = 10
dt = 1.
epsilon = 1e-4

time_index_1 = np.linspace(-dt, (T + 1) * dt, T + 3)
time_index_2 = np.linspace(epsilon, T * dt - epsilon, 100)
interpolator = CubicInterpolator(T, dt)

waypoints = np.sin(time_index_1)

interpolated_configurations = []
interpolated_velocities = []
interpolated_accelerations = []

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

fig, axs = plt.subplots(1 + len(waypoints), 1)

axs[0].plot(time_index_2, interpolated_configurations)
axs[0].plot(time_index_1, np.sin(time_index_1), 'o')
axs[0].plot(time_index_1, np.sin(time_index_1), '--')
axs[0].legend(['Position'])

# plt.subplot(512)
# plt.plot(time_index_2, interpolated_velocities)
# plt.legend(['Velocity'])

for k in range(len(waypoints)):
    axs[k + 1].plot(time_index_2, interpolated_derivative[k])
    axs[k + 1].legend(['Waypoint' + str(k)])

plt.show()
