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


import sys
print(sys.version_info)
if sys.version_info >= (3, 0):
    from .common_imports import *
else:
    import common_imports
from graph.shortest_path import *
from learning.dataset import *
from learning.random_environment import *
from utils.misc import *
from geometry.workspace import *
from motion.cost_terms import *
from motion.trajectory import *
from motion.objective import *
from tqdm import tqdm
import time
from utils.options import *
from utils.collision_checking import *


PATHS_PER_ENVIROMENT = 10
AVERAGE_COST = True
POINTS = 24
DRAW = False
VERBOSE = False
ALPHA = 10.
MARGIN = .20
OFFSET = 0.1
TRAJ_LENGTH = 20
DEFAULT_WS_FILE = '1k_small.hdf5'


def obsatcle_potential(workspace):
    sdf = SignedDistanceWorkspaceMap(workspace)
    return CostGridPotential2D(sdf, ALPHA, MARGIN, OFFSET)


def cost_grid(workspace):
    return obsatcle_potential(workspace)(
        workspace.box.stacked_meshgrid(POINTS)).T


def grid_to_world_path(workspace, path):
    """ grid to world path """
    pixel_map = workspace.pixel_map(POINTS)
    path_w = [None] * len(path)
    for i, p in enumerate(path):
        path_w[i] = pixel_map.grid_to_world(np.array(p))
    return path


def sample_path(
        workspace,
        graph,
        nb_points,
        no_linear_interpolation):
    """ finds a path that does not collide with enviroment
    but that is significantly difficult to perform """
    cost = cost_grid(workspace)
    pixel_map = workspace.pixel_map(nb_points)
    half_diag = workspace.box.diag() / 2.
    path = None
    for _ in range(100):
        s_w = sample_collision_free(workspace, MARGIN / 2)
        t_w = sample_collision_free(workspace, MARGIN / 2)
        if no_linear_interpolation:
            if np.linalg.norm(s_w - t_w) < half_diag:
                continue
            if not collision_check_linear_interpolation(workspace, s_w, t_w):
                continue
        s = pixel_map.world_to_grid(s_w)
        t = pixel_map.world_to_grid(t_w)
        try:
            path = graph.dijkstra_on_map(cost, s[0], s[1], t[0], t[1])
        except:
            print("Warning : error in dijkstra")

    return path


def generate_paths(nb_points):

    workspaces = load_workspaces_from_file(
        filename="workspaces_" + DEFAULT_WS_FILE)

    grid = np.ones((nb_points, nb_points))
    graph = CostmapToSparseGraph(grid, AVERAGE_COST)
    graph.convert()

    paths = [None] * len(workspaces)
    for k, workspace in enumerate(tqdm(workspaces)):
        paths[k] = [None] * PATHS_PER_ENVIROMENT
        if VERBOSE:
            print(("Compute demo ", k))
        nb_tries = 0
        for i in range(PATHS_PER_ENVIROMENT):
            while paths[k][i] is None:
                nb_tries += 1
                hard = nb_tries < 20
                try:
                    paths[k][i] = sample_path(
                        workspace,
                        graph,
                        nb_points=nb_points,
                        no_linear_interpolation=hard)
                except ValueError as e:
                    trajectories[k] = None
                    if verbose:
                        print("Warning : ", e)
        if DRAW:
            import rendering.workspace_renderer as render
            viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True)
            viewer.set_workspace(workspace)
            viewer.draw_ws_background(obsatcle_potential(workspace), nb_points)
            viewer.draw_ws_obstacles()
            for path in paths[k]:
                viewer.draw_ws_line(path, color="r")
                viewer.draw_ws_point(path[0])
                viewer.draw_ws_point(path[-1])
            viewer.show_once()
            time.sleep(.4)
    return trajectories


if __name__ == '__main__':

    np.random.seed(0)
    parser = optparse.OptionParser("usage: %prog [options] arg1 arg2")
    parser.add_option('--nb_points', type="int", default=24)
    add_boolean_options(parser, ['verbose', 'show_result', 'average_cost'])
    (options, args) = parser.parse_args()
    VERBOSE = options.verbose
    DRAW = options.show_result
    show_demo_id = -1
    nb_points = options.nb_points
    print((" -- options : ", options))
    paths = generate_paths(nb_points)
    save_trajectories_to_file(paths, filename='paths_' + DEFAULT_WS_FILE)
