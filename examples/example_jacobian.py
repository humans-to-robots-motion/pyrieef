
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
#                                       Jim Mainprice on Thursday June 13 2019

import pybullet_utils.bullet_client as bc
import pybullet
import numpy as np
import time

# connect to pybullet
# new direct client, GUI for graphic
p = bc.BulletClient(connection_mode=pybullet.GUI)

# load robot
robot = p.loadURDF("../data/r2_robot.urdf")
njoints = p.getNumJoints(robot)
print("number joints: " + str(njoints))

# end-effector idx
eff_idx = njoints - 1

# calc jacobian
# J[0] pos, J[1] angular
zero_vec = [0.] * njoints

np.random.seed(0)
configurations = np.random.uniform(low=-3.14, high=3.14, size=(100, 3))

t_0 = time.time()
for q in configurations:
    J = p.calculateJacobian(
        robot, eff_idx, [0., 0., 0.], q.tolist(), zero_vec, zero_vec)
t = time.time() - t_0

print("pos jacobian:")
print(np.array(J[0]))

print("Done. ({} sec., {:0.0} Hz)".format(t, len(configurations) / t))
for i in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)
