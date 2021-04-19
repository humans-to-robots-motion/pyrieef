#!/usr/bin/env python

# Copyright (c) 2020, University of Stuttgart
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
#                                      Jim Mainprice on Monday January 27 2020

import demos_common_imports
from pyrieef.geometry.workspace import *
from pyrieef.geometry.rotations import *
from pyrieef.kinematics.robot import *
from pyrieef.rendering.workspace_renderer import WorkspaceDrawer
from pyrieef.rendering.workspace_renderer import WorkspaceOpenGl

robot = create_robot_from_file(scale=.02)
# robot = create_robot_with_even_keypoints(scale=.03)
workspace = Workspace()
workspace.obstacles.append(Box(
    origin=np.array([-.3, 0]), dim=np.array([.4, .02])))
workspace.obstacles.append(Box(
    origin=np.array([.3, 0]), dim=np.array([.4, .02])))
sdf = SignedDistanceWorkspaceMap(workspace)
# viewer = WorkspaceDrawer(workspace, wait_for_keyboard=True)
viewer = WorkspaceOpenGl(workspace, wait_for_keyboard=True)
q = np.array([.0, -.2, .2])
viewer.draw_ws_obstacles()
viewer.draw_ws_polygon(robot.shape, q[:2], q[2])
for name, i in robot.keypoint_names.items():
    p = robot.keypoint_map(i)(q)
    viewer.draw_ws_point(p, color='b', shape='o')
viewer.background_matrix_eval = True
viewer.draw_ws_background(sdf, nb_points=100)
viewer.show_once()
