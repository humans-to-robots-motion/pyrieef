#!/usr/bin/env python

# Copyright (c) 2019, University of Stuttgart
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

# os.sys.path.insert is only needed when pybullet is not installed
# but running from github repo instead
import os
import pybullet
import pybullet_data

# choose connection method: GUI, DIRECT, SHARED_MEMORY
pybullet.connect(pybullet.GUI)
pybullet.loadURDF(os.path.join(
    pybullet_data.getDataPath(), "plane.urdf"), 0, 0, -1)
# load URDF, given a relative or absolute file+path
obj = pybullet.loadURDF(os.path.join(pybullet_data.getDataPath(),
                                     "r2d2.urdf"))

posX = 0
posY = 3
posZ = 2

# query the number of joints of the object
numJoints = pybullet.getNumJoints(obj)

print(numJoints)

# set the gravity acceleration
pybullet.setGravity(0, 0, -9.8)

# step the simulation for 5 seconds
# t_end = time.time() + 5
while True:
    pybullet.stepSimulation()
    # posAndOrn = pybullet.getBasePositionAndOrientation(obj)
    # print(posAndOrn)

print("finished")
# remove all objects
pybullet.resetSimulation()

# disconnect from the physics server
pybullet.disconnect()
