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

from __future__ import print_function
import common_imports
from graph.shortest_path import *
from learning.dataset import *
from learning.utils import *
from geometry.workspace import *
from motion.cost_terms import *
from motion.trajectory import *
from motion.objective import *
import rendering.workspace_renderer as render
from tqdm import tqdm
import time
from utils.options import *
from utils.collision_checking import *


def load_circles_workspace(ws, box):
    workspace = Workspace(box)
    for i in range(ws[0].shape[0]):
        center = ws[0][i]
        radius = ws[1][i][0]
        if radius > 0:
            workspace.add_circle(center, radius)
    return workspace


def save_trajectories_to_file(trajectories):
    max_length = 0
    for t in trajectories:
        if len(t) > max_length:
            max_length = len(t)
    trajectories_data = [-1000. * np.ones((max_length, 2))] * len(trajectories)
    for i, t in enumerate(trajectories):
        for k, q in enumerate(t):
            trajectories_data[i][k] = q

    data = {}
    data["datasets"] = np.stack(trajectories_data)
    write_dictionary_to_file(data, filename='trajectories_1k_small.hdf5')


def optimize(path, workspace, costmap, verbose=False):
    T = len(path) - 1
    trajectory = Trajectory(T, 2)
    for i, p in enumerate(path):
        trajectory.configuration(i)[:] = path[i]
    sdf = SignedDistanceWorkspaceMap(workspace)
    optimizer = MotionOptimization2DCostMap(
        T=T,
        n=2,
        q_init=trajectory.initial_configuration(),
        q_goal=trajectory.final_configuration(),
        extends=workspace.box.extends(),
        signed_distance_field=sdf)
    optimizer.costmap = CostGridPotential2D(sdf, 20., .03, 1.)
    optimizer.verbose = verbose
    optimizer.set_scalars(
        obstacle_scalar=1.,
        init_potential_scalar=0.,
        term_potential_scalar=10000000.,
        smoothness_scalar=1.)
    optimizer.create_clique_network()
    optimizer.add_all_terms()
    optimizer.create_objective()
    t_start = time.time()
    [dist, traj, gradient, deltas] = optimizer.optimize(
        trajectory.configuration(0), 100, trajectory,
        optimizer="newton")
    if verbose:
        print("time : {}".format(time.time() - t_start))
    return trajectory


def compute_demonstration(
        workspace, converter, nb_points, show_result, average_cost, verbose):
    phi = CostGridPotential2D(
        SignedDistanceWorkspaceMap(workspace), 20., .03, 1.)
    costmap = phi(workspace.box.stacked_meshgrid(nb_points)).transpose()
    resample = True
    while resample:
        s_w = sample_collision_free(workspace)
        t_w = sample_collision_free(workspace)
        if not collision_check_linear_interpolation(workspace, s_w, t_w):
            continue
        resample = False
        pixel_map = workspace.pixel_map(nb_points)
        s = pixel_map.world_to_grid(s_w)
        t = pixel_map.world_to_grid(t_w)
        try:
            path = converter.dijkstra_on_map(costmap, s[0], s[1], t[0], t[1])
        except:
            resample = True

    traj = [None] * len(path)
    trajectory = ContinuousTrajectory(len(path) - 1, 2)
    for i, p in enumerate(path):
        traj[i] = pixel_map.grid_to_world(np.array(p))
        trajectory.configuration(i)[:] = traj[i]

    nb_config = 20
    interpolated_traj = [None] * nb_config
    for i, s in enumerate(np.linspace(0, 1, nb_config)):
        interpolated_traj[i] = trajectory.configuration_at_parameter(s)

    optimized_trajectory = optimize(interpolated_traj, workspace, verbose)
    if collision_check_trajectory(workspace, optimized_trajectory):
        print("Warning: has collision !!!")

    result = [None] * nb_config
    for i in range(len(result)):
        result[i] = optimized_trajectory.configuration(i)

    if show_result:
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True)
        viewer.draw_ws_background(phi, nb_points)
        viewer.draw_ws_obstacles()
        viewer.draw_ws_line(interpolated_traj, color="r")
        viewer.draw_ws_line(traj, color="b")
        viewer.draw_ws_line(result, color="g")
        viewer.draw_ws_point(s_w)
        viewer.draw_ws_point(t_w)
        viewer.show_once()
        time.sleep(.4)

    return path

if __name__ == '__main__':

    parser = optparse.OptionParser("usage: %prog [options] arg1 arg2")
    add_boolean_options(parser, ['verbose', 'show_result', 'average_cost'])
    (options, args) = parser.parse_args()
    verbose = options.verbose
    show_result = options.show_result
    show_demo_id = -1
    average_cost = options.average_cost
    nb_points = 24

    print(" -- options : ", options)

    data_ws = dict_to_object(
        load_dictionary_from_file(filename='workspaces_1k_small.hdf5'))
    print(" -- size : ", data_ws.size)
    print(" -- lims : ", data_ws.lims.flatten())
    print(" -- datasets.shape : ", data_ws.datasets.shape)
    print(" -- data_ws.shape : ", data_ws.datasets.shape)
    x_min = data_ws.lims[0, 0]
    x_max = data_ws.lims[0, 1]
    y_min = data_ws.lims[1, 0]
    y_max = data_ws.lims[1, 1]
    box = box_from_limits(x_min, x_max, y_min, y_max)

    converter = CostmapToSparseGraph(
        np.ones((nb_points, nb_points)), average_cost)
    converter.convert()
    np.random.seed(1)

    trajectories = [None] * len(data_ws.datasets)
    for k, ws in enumerate(tqdm(data_ws.datasets)):
        if verbose:
            print("Compute demo ", k)
        trajectories[k] = compute_demonstration(
            load_circles_workspace(ws, box),
            converter,
            nb_points=nb_points,
            # show_result=show_result,
            show_result=(show_demo_id == k or show_result),
            average_cost=average_cost,
            verbose=verbose)
        if show_demo_id == k and not show_result:
            break
    save_trajectories_to_file(trajectories)
