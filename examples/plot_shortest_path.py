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
from pyrieef.motion.cost_terms import *
import pyrieef.rendering.workspace_planar as render
from utils import timer
import time

show_result = True
radius = .1
nb_points = 40
average_cost = False
integral_cost = True
binary_cost = True

workspace = Workspace()
workspace.obstacles.append(Circle(np.array([0.1, 0.1]), radius))
workspace.obstacles.append(Circle(np.array([-.1, 0.1]), radius))

if binary_cost:
    costmap = occupancy_map(nb_points, workspace)
    costmap = (~costmap.astype(bool)).astype(int)  # invert occupnacy
else:
    phi = CostGridPotential2D(
        SignedDistanceWorkspaceMap(workspace), 10., .1, 10.)
    costmap = phi(workspace.box.stacked_meshgrid(nb_points)).T

print(costmap)

converter = CostmapToSparseGraph(costmap, average_cost)
converter.integral_cost = integral_cost
graph = converter.convert()
if average_cost:
    assert check_symmetric(graph)
# predecessors = shortest_paths(graph)
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(1)
time_0 = time.time()
for i in range(100):
    s_w = sample_collision_free(workspace)
    t_w = sample_collision_free(workspace)
    s = pixel_map.world_to_grid(s_w)
    t = pixel_map.world_to_grid(t_w)
    # print "querry : ({}, {}) ({},{})".format(s[0], s[1], t[0], t[1])
    # path = converter.shortest_path(predecessors, s[0], s[1], t[0], t[1])
    # path = converter.dijkstra(graph, s[0], s[1], t[0], t[1])
    try:
        print("planning...")
        path = converter.dijkstra_on_map(costmap, s[0], s[1], t[0], t[1])
    except:
        continue
    print("took t : {} sec.".format(time.time() - time_0))

    if show_result:
        trajectory = [None] * len(path)
        for i, p in enumerate(path):
            trajectory[i] = pixel_map.grid_to_world(np.array(p))
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True)
        # viewer.set_drawing_axis(0)
        # viewer.draw_ws_background(phi, nb_points, interpolate="none")
        viewer.draw_ws_img(costmap)
        viewer.draw_ws_obstacles()
        viewer.draw_ws_line(trajectory)
        viewer.draw_ws_point(s_w)
        viewer.draw_ws_point(t_w)
        viewer.show_once()

print(path)
