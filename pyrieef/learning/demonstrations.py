#!/usr/bin/env python

# Copyright (c) 2018 University of Stuttgart
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
#                                        Jim Mainprice on Sunday Aug 27 2018

import common_imports
from graph.shortest_path import *
from learning.dataset import *
from learning.utils import *
from geometry.workspace import *
from motion.cost_terms import *
import rendering.workspace_renderer as render
from tqdm import tqdm
import time


def collision_check_interpolation(workspace, p_init, p_goal):
    """ Check interior interpolation for collision """
    delta = workspace.box.diag() / 100.
    nb_points = int(np.linalg.norm(p_init - p_goal) / delta)
    for i in range(1, nb_points):
        alpha = float(i) / float(nb_points)
        p = (1. - alpha) * p_init + alpha * p_goal
        if workspace.in_collision(p):
            return True
    return False


def load_circles_workspace(ws, box):
    workspace = Workspace(box)
    for i in range(ws[0].shape[0]):
        center = ws[0][i]
        radius = ws[1][i][0]
        if radius > 0:
            workspace.add_circle(center, radius)
    return workspace


def compute_demonstration(
        workspace, converter, nb_points, show_result, average_cost):

    phi = CostGridPotential2D(SignedDistanceWorkspaceMap(workspace),
                              10., .03, 10.)
    costmap = phi(workspace.box.stacked_meshgrid(nb_points)).transpose()
    resample = True
    while resample:
        s_w = sample_collision_free(workspace)
        t_w = sample_collision_free(workspace)
        if not collision_check_interpolation(workspace, s_w, t_w):
            continue
        resample = False
        pixel_map = workspace.pixel_map(nb_points)
        s = pixel_map.world_to_grid(s_w)
        t = pixel_map.world_to_grid(t_w)
        try:
            path = converter.dijkstra_on_map(costmap, s[0], s[1], t[0], t[1])
        except:
            resample = True

    trajectory = [None] * len(path)
    for i, p in enumerate(path):
        trajectory[i] = pixel_map.grid_to_world(np.array(p))

    if show_result:
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=False)
        viewer.draw_ws_background(phi, nb_points)
        viewer.draw_ws_obstacles()
        viewer.draw_ws_line(trajectory)
        viewer.draw_ws_point(s_w)
        viewer.draw_ws_point(t_w)
        viewer.show_once()
        time.sleep(.4)

if __name__ == '__main__':

    data_ws = dict_to_object(
        load_dictionary_from_file(filename='workspaces_1k_small.hdf5'))
    print " -- size : ", data_ws.size
    print " -- lims : ", data_ws.lims
    print " -- datasets.shape : ", data_ws.datasets.shape
    print " -- data_ws.shape : ", data_ws.datasets.shape

    x_min = data_ws.lims[0, 0]
    x_max = data_ws.lims[0, 1]
    y_min = data_ws.lims[1, 0]
    y_max = data_ws.lims[1, 1]
    box = box_from_limits(x_min, x_max, y_min, y_max)

    show_result = False
    average_cost = False
    nb_points = 24
    converter = CostmapToSparseGraph(
        np.ones((nb_points, nb_points)), average_cost)
    converter.convert()
    np.random.seed(1)

    for k, ws in enumerate(tqdm(data_ws.datasets)):
        compute_demonstration(load_circles_workspace(ws, box),
                              converter,
                              nb_points=nb_points,
                              show_result=show_result,
                              average_cost=average_cost)
