#!/usr/bin/env python

# Copyright (c) 2015 Max Planck Institute
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
# Jim Mainprice on Sunday June 17 2017

from demos_common_imports import *
import numpy as np
import matplotlib.pyplot as plt
from pyrieef.geometry.workspace import *
from pyrieef.geometry.charge_simulation import *
from pyrieef.geometry.pixel_map import *
from pyrieef.geometry.geodesics import *
import itertools

workspace = Workspace()

segment = Segment()
segment.origin = np.array([0.2, -.4])
segment.length = 1.0
segment.orientation = 2.0
segment.nb_points = 50
workspace.obstacles.append(segment)

segment = Segment()
segment.origin = np.array([0.0, -.5])
segment.length = 1.0
segment.orientation = 0.0
segment.nb_points = 50
workspace.obstacles.append(segment)

points = workspace.all_points()
X = np.array(points)[:, 0]
Y = np.array(points)[:, 1]

print "Charge simulation..."
simulation = ChargeSimulation()
simulation.charged_points_ = points
simulation.Run()

extends = Extends(workspace.box.dim[0] / 2.)
grid = PixelMap(0.01, extends)
matrix = np.zeros((grid.nb_cells_x, grid.nb_cells_y))
for i in range(grid.nb_cells_x):
    for j in range(grid.nb_cells_y):
        p = grid.grid_to_world(np.array([i, j]))
        # TODO why is it this way... (j before i)
        # these are matrix coordinates...
        matrix[j, i] = simulation.PotentialCausedByObject(p)
plt.imshow(matrix, origin='lower', extent=workspace.box.box_extends())
plt.scatter(X, Y)
plt.ylabel('some points')
plt.axis('equal')
plt.axis(workspace.box.box_extends())

print "Compute geodesics..."
x_goal = np.array([0.4, -0.4])
nx, ny = (5, 5)
x = np.linspace(-.4, -.3, nx)
y = np.linspace(-.4, -.2, ny)
for i, j in itertools.product(range(nx), range(ny)):
    x_init = np.array([x[i], y[j]])
    # line = ComputeInterpolationGeodescis(simulation, x_init, x_goal)
    # line = ComputeGeodesic(simulation, x_init, x_goal)
    line = ComputeInitialVelocityGeodescis(simulation, x_init, x_goal - x_init)
    X = np.array(line)[:, 0]
    Y = np.array(line)[:, 1]
    plt.plot(X, Y, color="r", linewidth=2.0)
    plt.plot(x_init[0], x_init[1], 'ro')
plt.plot(x_goal[0], x_goal[1], 'ko')
print "Done."
plt.show()
