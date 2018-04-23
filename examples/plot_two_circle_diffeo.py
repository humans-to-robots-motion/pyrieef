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
#                                           Jim Mainprice on Sunday June 17 2017

import demos_common_imports
import numpy as np
import matplotlib.pyplot as plt
from workspace import *
from charge_simulation import *
from pixel_map import *
from geodesics import *
from diffeomorphisms import *
from utils import *
import itertools

# circle = ElectricCircle()
# circle.origin = np.array([.1, .0])

circle_1 = AnalyticCircle()
circle_1.set_alpha(alpha_f, beta_inv_f)
circle_1.circle.origin = np.array([.1, .0])
circle_1.circle.radius = 0.1
circle_1.eta = circle_1.circle.radius

circle_2 = AnalyticCircle()
circle_2.set_alpha(alpha_f, beta_inv_f)
circle_2.circle.origin = np.array([.1, .25])
circle_2.circle.radius = 0.05
circle_2.eta = circle_2.circle.radius

circle_3 = AnalyticCircle()
circle_3.set_alpha(alpha_f, beta_inv_f)
circle_3.circle.origin = np.array([.2, .25])
circle_3.circle.radius = 0.05
circle_3.eta = circle_3.circle.radius

circle_4 = AnalyticCircle()
circle_4.set_alpha(alpha_f, beta_inv_f)
circle_4.circle.origin = np.array([.0, .25])
circle_4.circle.radius = 0.05
circle_4.eta = circle_4.circle.radius

analytical_circles = []
analytical_circles.append(circle_1)
analytical_circles.append(circle_2)
analytical_circles.append(circle_3)
analytical_circles.append(circle_4)

workspace = Workspace()
for circle in analytical_circles:
    workspace.obstacles.append(circle.object())
for obstacles in workspace.obstacles:
    points = obstacles.SampledPoints()
    X = np.array(points)[:,0]
    Y = np.array(points)[:,1]
    plt.plot( X, Y, "b", linewidth=2.0 )

extends = Extends(workspace.box.dim[0]/2.)
grid = PixelMap(0.01, extends)
matrix = np.zeros((grid.nb_cells_x, grid.nb_cells_y))
for i in range(grid.nb_cells_x):
    for j in range(grid.nb_cells_y):
        p = grid.grid_to_world(np.array([i, j]))
plt.axis('equal')
plt.axis(workspace.box.Extends())

x_goal = np.array([0.4, 0.4])
nx, ny = (3, 3)
x = np.linspace(-.2, -.1, nx)
y = np.linspace(-.5, -.1, ny)
# nx, ny = (3, 1)
# x = np.linspace(-.2, -.1, nx)
# y = np.linspace(-.2, -.2, ny)
# nx, ny = (1, 1)
# x = [-0.2] 
# y = [-0.2]

circles = AnalyticMultiCircle(analytical_circles)

for i, j in itertools.product(range(nx), range(ny)):
    x_init = np.array([x[i], y[j]])
    print "x_init : ", x_init

    # Does not have an inverse.
    # [line, line_inter] = InterpolationMultiGeodescis(
    #     circles, x_init, x_goal)

    line = NaturalGradientGeodescis(
        circles, x_init, x_goal)

    plot_line(line, 'r')
    # plot_line(line_inter, 'g')

for obst in workspace.obstacles:
    plt.plot( obst.origin[0], obst.origin[1], 'kx' )
plt.plot( x_goal[0], x_goal[1], 'ko' )
plt.show()  

