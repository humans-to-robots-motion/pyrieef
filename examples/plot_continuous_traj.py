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

import demos_common_imports
import time
import numpy as np
import matplotlib.pyplot as plt
from pyrieef.motion.objective import ConstantAccelerationTrajectory

T = 10
dt = 1.
epsilon = 1e-6
trajectory = ConstantAccelerationTrajectory(T, 1, dt)
time_index_1 = np.linspace(0., T * dt, T + 1)
time_index_2 = np.linspace(0., T * dt, 1000)

for i, t in enumerate(time_index_1):
    trajectory.configuration(i)[:] = np.sin(t)

interpolated_configurations = []
interpolated_velocities = []
interpolated_accelerations = []
for t in time_index_2:
    detla = epsilon if t > 0 else 0.
    q0 = trajectory.config_at_time(t - detla)
    q1 = trajectory.config_at_time(t)
    q2 = trajectory.config_at_time(t + epsilon)
    interpolated_configurations.append(q1)
    interpolated_velocities.append((q2 - q1) / epsilon)
    interpolated_accelerations.append((q2 - 2 * q1 + q0) / (epsilon ** 2))

plt.figure()
plt.subplot(311)
# Warning the accleration for the last config is ill defined.
# Note that for the first segment the interpolation is
# not defined.
plt.plot(time_index_2[:-1], interpolated_accelerations[:-1])
plt.legend(['Acceleration'])

plt.subplot(312)
plt.plot(time_index_2, interpolated_velocities)
plt.legend(['Velocity'])

plt.subplot(313)
plt.plot(time_index_1, trajectory.list_configurations())
plt.plot(time_index_1, trajectory.list_configurations(), '*')
plt.plot(time_index_2, interpolated_configurations)
plt.legend(['Position'])


plt.show()
