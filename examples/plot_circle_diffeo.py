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
from pyrieef.geometry.diffeomorphisms import *
from pyrieef.geometry.utils import *
import itertools

# phi = ElectricCircle()
# phi.origin = np.array([.1, .0])

phi = AnalyticCircle()
phi.set_alpha(alpha_f, beta_inv_f)
phi.circle.origin = np.array([.1, .0])

workspace = Workspace()
workspace.obstacles.append(phi.object())
points = workspace.all_points()
X = np.array(points)[:, 0]
Y = np.array(points)[:, 1]

extends = Extends(workspace.box.dim[0] / 2.)
grid = PixelMap(0.01, extends)
matrix = np.zeros((grid.nb_cells_x, grid.nb_cells_y))
for i in range(grid.nb_cells_x):
    for j in range(grid.nb_cells_y):
        p = grid.grid_to_world(np.array([i, j]))
        # TODO why is it this way... (j before i)
        # these are matrix coordinates...
        # matrix[j, i] = simulation.PotentialCausedByObject(p)
# plt.imshow(matrix, origin='lower',extent=workspace.box.Extends())
# plt.scatter( X, Y )
plt.plot(X, Y, "k", linewidth=2.0)
# plt.ylabel('some points')
plt.axis('equal')
plt.axis(workspace.box.box_extends())

x_goal = np.array([0.4, 0.4])
nx, ny = (3, 3)
x = np.linspace(-.2, -.1, nx)
y = np.linspace(-.5, -.1, ny)
for i, j in itertools.product(range(nx), range(ny)):
    x_init = np.array([x[i], y[j]])
    print "x_init : ", x_init

    [line, line_inter] = InterpolationGeodescis(phi, x_init, x_goal)
    plot_line(line, 'b', .01)
    # plot_line(line_inter, 'g')

    line = NaturalGradientGeodescis(phi, x_init, x_goal)
    plot_line(line, 'r', .01)

plt.plot(workspace.obstacles[0].origin[0],
         workspace.obstacles[0].origin[1], 'kx')
plt.plot(x_goal[0], x_goal[1], 'ko')
plt.show()
