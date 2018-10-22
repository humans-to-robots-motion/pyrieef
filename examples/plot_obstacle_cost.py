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

from demos_common_imports import *
import numpy as np
from numpy.testing import assert_allclose
from pyrieef.motion.cost_terms import *
from pyrieef.geometry.workspace import *
import matplotlib.pyplot as plt


def plot(obst_cost, label):
    x = np.linspace(0., 1., 100)
    y = np.zeros(100)
    z = np.zeros(100)
    for i in range(len(z)):
        z[i] = obst_cost(np.array([x[i], y[i]]))
    plt.plot(x, z, label=label)

workspace = Workspace()
workspace.obstacles.append(Circle(np.array([0., 0.]), 0.1))
sdf = SignedDistanceWorkspaceMap(workspace)
fig = plt.figure(figsize=(7, 6.5))
plot(CostGridPotential2D(sdf, 10., .1, 1.), "10")
plot(CostGridPotential2D(sdf, 20., .1, 1.), "20")
plot(CostGridPotential2D(sdf, 30., .1, 1.), "30")
plot(CostGridPotential2D(sdf, 40., .1, 1.), "40")
plt.legend()
plt.show()
