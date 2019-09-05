
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
from pyrieef.motion.trajectory import CubicSplineTrajectory
from scipy.interpolate import interp1d

T = 10
dt = 1.
epsilon = 1e-6

time_index_1 = np.linspace(0., T * dt, T + 1)
time_index_2 = np.linspace(epsilon, T * dt - epsilon, 1000)

f = interp1d(time_index_1, np.sin(time_index_1), kind='cubic')
trajectory = CubicSplineTrajectory(T, 1, dt)
for i, t in enumerate(time_index_1):
    trajectory.configuration(i)[:] = np.sin(t)
print(trajectory.list_configurations())
trajectory.initialize_spline()

interpolated_configurations = []
interpolated_velocities = []
interpolated_accelerations = []
for t in time_index_2:
    interpolated_configurations.append(trajectory(t))
    interpolated_velocities.append(trajectory.velocity(t))
    interpolated_accelerations.append(trajectory.acceleration(t))

plt.figure()
plt.subplot(311)
# Warning the accleration for the last config is ill defined.
# Note that for the first segment the interpolation is not defined.
plt.plot(time_index_2[1:-1], interpolated_accelerations[1:-1])
plt.legend(['Acceleration'])

plt.subplot(312)
plt.plot(time_index_2, interpolated_velocities)
plt.legend(['Velocity'])

plt.subplot(313)
plt.plot(time_index_2, interpolated_configurations)
plt.plot(time_index_1, np.sin(time_index_1), 'o')
plt.plot(time_index_1, np.sin(time_index_1), '--')
plt.legend(['Position'])


plt.show()
