#!/usr/bin/env python

# Copyright (c) 2022
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
#                                        Jim Mainprice on Friday March 11 2022

import demos_common_imports
from pyrieef.geometry.workspace import *
from pyrieef.geometry.workspace import occupancy_map
from pyrieef.rendering.workspace_planar import WorkspaceDrawer

# np.random.seed(1)

boxes_height = .10  # percentage
boxes_width = .15   # percentage
environment = EnvBox(dim=np.array([2., 2.]))
workspace = Workspace(environment)
diagonal = workspace.box.diag()
max_h = diagonal * boxes_height
min_h = diagonal * boxes_height * .5
max_w = diagonal * boxes_width
min_w = diagonal * boxes_width * .5
workspace.obstacles = []

while len(workspace.obstacles) < 5:
    origin = workspace.box.sample_uniform()
    h = (max_h - min_h) * np.random.rand() + min_h
    w = (max_w - min_w) * np.random.rand() + min_w
    if workspace.min_dist(origin)[0] < np.linalg.norm([h, w]):
        continue
    dimensions = np.array([w, h])
    theta = np.pi * np.random.rand() - np.pi
    orientation = rotation_matrix_2d_radian(theta)
    workspace.obstacles.append(OrientedBox(origin, dimensions, orientation))


# Compute Occupancy map
matrix = occupancy_map(150, workspace)

# Compute Signed distance field
# meshgrid = workspace.box.stacked_meshgrid(150)
# matrix = SignedDistanceWorkspaceMap(workspace)(meshgrid).T

# Setup viewer
viewer = WorkspaceDrawer(workspace, wait_for_keyboard=True)
viewer.draw_ws_img(matrix)
viewer.draw_ws_obstacles()
viewer.show_once()
