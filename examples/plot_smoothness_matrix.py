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

dim = 1
trajectory = linear_interpolation_trajectory(
    q_init=np.zeros(dim),
    q_goal=np.ones(dim),
    T=20
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
                    formatter={'float_kind': '{:8.0f}'.format})
print(H1[dim * 10:, dim * 10:])

H2 = objective.create_smoothness_metric()
np.set_printoptions(suppress=True, linewidth=200, precision=0,
                    formatter={'float_kind': '{:8.0f}'.format})


A_inv = np.linalg.inv(H2)
print(np.max(A_inv))
A_inv /= np.max(A_inv)
plt.plot(A_inv)
plt.show()
