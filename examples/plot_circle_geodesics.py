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
import matplotlib.pyplot as plt
from pyrieef.geometry.workspace import *
from pyrieef.geometry.charge_simulation import *
from pyrieef.geometry.pixel_map import *
from pyrieef.geometry.geodesics import *
from pyrieef.geometry.diffeomorphisms import *
from pyrieef.geometry.utils import *
from pyrieef.motion.trajectory import *
from pyrieef.motion.cost_terms import *
from pyrieef.motion.geodesic import GeodesicObjective2D
from pyrieef.optimization import algorithms
import itertools


VERBOSE = True
BOXES = False
TRAJ_LENGTH = 60
ALPHA = 10.
MARGIN = .20
OFFSET = 0.1


def optimize_geodesic(workspace, phi, q_init, q_goal):
    trajectory = linear_interpolation_trajectory(
        q_init, q_goal, T=TRAJ_LENGTH)
    objective = GeodesicObjective2D(
        T=trajectory.T(),
        n=trajectory.n(),
        q_init=trajectory.initial_configuration(),
        q_goal=trajectory.final_configuration(),
        embedding=None)
    sdf = SignedDistanceWorkspaceMap(workspace)
    cost = CostGridPotential2D(sdf, ALPHA, MARGIN, OFFSET)
    objective.embedding = phi
    objective.obstacle_potential = cost
    objective.workspace = workspace
    objective.create_clique_network()
    algorithms.newton_optimize_trajectory(
        objective.objective, trajectory, verbose=VERBOSE, maxiter=100)
    return trajectory.list_configurations()


phi = AnalyticCircle()
phi.set_alpha(alpha_f, beta_inv_f)
phi.circle.origin = np.array([.1, .0])

workspace = Workspace()
workspace.obstacles.append(phi.object())
points = workspace.all_points()
X = np.array(points)[:, 0]
Y = np.array(points)[:, 1]

extent = Extent(workspace.box.dim[0] / 2.)
grid = PixelMap(0.01, extent)
matrix = np.zeros((grid.nb_cells_x, grid.nb_cells_y))
for i in range(grid.nb_cells_x):
    for j in range(grid.nb_cells_y):
        p = grid.grid_to_world(np.array([i, j]))
plt.plot(X, Y, "k", linewidth=2.0)
plt.axis('equal')
plt.axis(workspace.box.box_extent())

x_goal = np.array([0.4, 0.4])
nx, ny = (3, 3)
x = np.linspace(-.2, -.1, nx)
y = np.linspace(-.5, -.1, ny)
for i, j in itertools.product(list(range(nx)), list(range(ny))):
    x_init = np.array([x[i], y[j]])
    print("x_init : ", x_init)

    [line, line_inter] = InterpolationGeodescis(phi, x_init, x_goal)
    plot_line(line, 'b', .01)

    line = optimize_geodesic(workspace, phi, x_init, x_goal)
    plot_line(line, 'r', .01)

plt.plot(workspace.obstacles[0].origin[0],
         workspace.obstacles[0].origin[1], 'kx')
plt.plot(x_goal[0], x_goal[1], 'ko')
plt.show()
