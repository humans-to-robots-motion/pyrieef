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
from pyrieef.graph.shortest_path import *
from pyrieef.geometry.workspace import *
from pyrieef.geometry.differentiable_geometry import *
from pyrieef.motion.cost_terms import *
import pyrieef.rendering.workspace_renderer as render
import time

show_result = True
radius = .1
nb_points = 40
nb_rbfs = 5
average_cost = False
alpha = .0001
sigma = 200

workspace = Workspace()

# Sample cost terrain
# w = np.random.random(64)
w = np.ones(nb_rbfs**2)
rbf = [None] * nb_rbfs**2
points = workspace.box.meshgrid_points(nb_rbfs).T
print(points.shape)
for i, x0 in enumerate(points):
    print("x0 : {} {}".format(i, x0.T))
    rbf[i] = RadialBasisFunction(x0, sigma * np.eye(2))
phi = SumOfTerms(rbf)
X, Y = workspace.box.meshgrid(nb_points)
costmap = two_dimension_function_evaluation(X, Y, phi)
# costmap = phi(workspace.box.stacked_meshgrid(nb_points))
print("costmap.shape : ", costmap.shape)

# Plan path
converter = CostmapToSparseGraph(costmap, average_cost)
converter.integral_cost = True
graph = converter.convert()
if average_cost:
    assert check_symmetric(graph)
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(1)
time_0 = time.time()
for i in range(100):
    s_w = sample_collision_free(workspace)
    t_w = sample_collision_free(workspace)
    s = pixel_map.world_to_grid(s_w)
    t = pixel_map.world_to_grid(t_w)
    try:
        print("planning...")
        path = converter.dijkstra_on_map(
            alpha * costmap, s[0], s[1], t[0], t[1])
    except:
        continue
    print("took t : {} sec.".format(time.time() - time_0))

    if show_result:
        trajectory = [None] * len(path)
        for i, p in enumerate(path):
            trajectory[i] = pixel_map.grid_to_world(np.array(p))
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True)
        viewer.draw_ws_img(costmap, interpolate="none")
        viewer.draw_ws_obstacles()
        viewer.draw_ws_line(trajectory)
        viewer.draw_ws_point(s_w)
        viewer.draw_ws_point(t_w)
        viewer.show_once()

print(path)
