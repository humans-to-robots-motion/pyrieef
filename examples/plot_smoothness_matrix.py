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
from pyrieef.motion.objective import *
import matplotlib.pyplot as plt

SAMPLES = True

dim = 1
trajectory = linear_interpolation_trajectory(
    q_init=np.zeros(dim),
    q_goal=np.ones(dim),
    T=100
)
objective = MotionOptimization2DCostMap(
    T=trajectory.T(),
    n=trajectory.n(),
    q_init=trajectory.initial_configuration(),
    q_goal=trajectory.final_configuration()
)
objective.create_clique_network()
objective.add_smoothness_terms(2)
objective.create_objective()

H1 = objective.objective.hessian(trajectory.active_segment())
np.set_printoptions(suppress=True, linewidth=200, precision=0,
                    formatter={'float_kind': '{:2.0f}'.format})
print(H1.shape)
print(H1 / 200000)

H2 = objective.create_smoothness_metric() / 10000
# H2 = np.loadtxt("tmp.txt")
np.set_printoptions(suppress=True, linewidth=200, precision=0,
                    formatter={'float_kind': '{:2.0f}'.format})

if SAMPLES:
    # Here plot te sample
    std_dev = 1.
    nb_samples = 20
    # WARNING: This coefficent has to be large
    H1[0, 0] = 1000
    cov = np.linalg.inv(H1)
    mean = np.array([0] * cov.shape[0])
    samples = np.random.multivariate_normal(mean, cov, nb_samples).T
    samples = std_dev * samples
    plt.plot(samples)
else:
    # Here plot the matrix collumns
    print(H1)
    A_inv = np.linalg.inv(H1)
    print(np.max(A_inv))
    A_inv /= np.max(A_inv)
    plt.plot(A_inv)

plt.show()
