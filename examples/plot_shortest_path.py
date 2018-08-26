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
from graph.shortest_path import *
import numpy as np
from numpy.testing import assert_allclose
from geometry.workspace import *
from motion.cost_terms import *
import rendering.workspace_renderer as render
from utils import timer
import time

workspace = Workspace()
radius = .1
nb_points = 24
workspace.obstacles.append(Circle(np.array([0.1, 0.1]), radius))
workspace.obstacles.append(Circle(np.array([-.1, 0.1]), radius))
phi = SimplePotential2D(SignedDistanceWorkspaceMap(workspace))
costmap = phi(workspace.box.stacked_meshgrid(nb_points))
print costmap
average_cost = True
converter = CostmapToSparseGraph(costmap, average_cost)
graph = converter.convert()
if average_cost:
    assert check_symmetric(graph)

viewer = render.WorkspaceDrawer(workspace)
time_0 = time.time()
predecessors = shortest_paths(graph)

np.random.seed(1)
for i in range(100):
    s_i = int(23 * np.random.random())
    s_j = int(23 * np.random.random())
    t_i = int(23 * np.random.random())
    t_j = int(23 * np.random.random())
    print "querry : ({}, {}) ({},{})".format(s_i, s_j, t_i, t_j)
    path = converter.shortest_path(predecessors, s_i, s_j, t_i, t_j)
    pixel_map = workspace.pixel_map(nb_points)
    trajectory = [None] * len(path)
    for i, p in enumerate(path):
        trajectory[i] = pixel_map.grid_to_world(np.array(p))

    viewer.init()
    viewer.draw_ws_background(phi, nb_points)
    viewer.draw_ws_obstacles()
    viewer.draw_ws_line(trajectory)
    viewer.show_once()
print "took t : {} sec.".format(time.time() - time_0)
print path




