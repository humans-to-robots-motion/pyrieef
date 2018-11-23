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
from learning.random_paths import *
from utils.misc import *
from geometry.workspace import *
from motion.cost_terms import *
from motion.trajectory import *
from motion.objective import *
from tqdm import tqdm
import time
from utils.options import *
from utils.collision_checking import *


MAX_ITERATIONS = 100
MAX_GRADIENT_NORM = 10000.
ALPHA = 10.
MARGIN = .20
OFFSET = 0.1
TRAJ_LENGTH = 20
DEFAULT_WS_FILE = '1k_small.hdf5'


def obsatcle_potential(workspace):
    sdf = SignedDistanceWorkspaceMap(workspace)
    return CostGridPotential2D(sdf, ALPHA, MARGIN, OFFSET)


def optimize(path, workspace, costmap, verbose=False):
    T = len(path) - 1
    trajectory = Trajectory(T, 2)
    for i, p in enumerate(path):
        trajectory.configuration(i)[:] = path[i]

    optimizer = MotionOptimization2DCostMap(
        T=T,
        n=2,
        q_init=trajectory.initial_configuration(),
        q_goal=trajectory.final_configuration(),
        box=workspace.box,
        signed_distance_field=SignedDistanceWorkspaceMap(workspace))
    optimizer.obstacle_potential = obsatcle_potential(workspace)
    optimizer.verbose = verbose
    optimizer.set_scalars(
        obstacle_scalar=1.,
        init_potential_scalar=0.,
        term_potential_scalar=10000000.,
        acceleration_scalar=30.,
        velocity_scalar=5.)
    optimizer.create_clique_network()
    optimizer.add_smoothness_terms(1)
    optimizer.add_smoothness_terms(2)
    optimizer.add_obstacle_terms()
    optimizer.add_obstacle_barrier()
    optimizer.add_box_limits()
    optimizer.add_attractor(trajectory)
    optimizer.create_objective()
    t_start = time.time()
    # print(optimizer.verbose)
    [dist, traj, gradient, deltas] = optimizer.optimize(
        trajectory.configuration(0), MAX_ITERATIONS, trajectory,
        optimizer="newton")
    if np.linalg.norm(deltas) > MAX_GRADIENT_NORM:
        return None
    if verbose:
        print(("time : {}".format(time.time() - t_start)))
    return trajectory


def compute_demonstration(
        workspace, graph, nb_points,
        show_result, average_cost, verbose, no_linear_interpolation):
    pixel_map = workspace.pixel_map(nb_points)
    path = sample_path(workspace, graph, nb_points, no_linear_interpolation)
    if path is None:
        return None
    traj = [None] * len(path)
    trajectory = ContinuousTrajectory(len(path) - 1, 2)
    for i, p in enumerate(path):
        traj[i] = pixel_map.grid_to_world(np.array(p))
        trajectory.configuration(i)[:] = traj[i]

    interpolated_traj = [None] * TRAJ_LENGTH
    for i, s in enumerate(np.linspace(0, 1, TRAJ_LENGTH)):
        interpolated_traj[i] = trajectory.configuration_at_parameter(s)

    optimized_trajectory = optimize(
        interpolated_traj, workspace, None, verbose)
    if optimized_trajectory is None:
        print("Warning: gradient too big !!!")
        return None
    collision = collision_check_trajectory(workspace, optimized_trajectory)
    if collision and verbose:
        print("Warning: has collision !!!")
        return None

    result = [None] * TRAJ_LENGTH
    for i in range(len(result)):
        result[i] = optimized_trajectory.configuration(i)

    return optimized_trajectory


def generate_one_demonstration(nb_points, demo_id):
    grid = np.ones((nb_points, nb_points))
    graph = CostmapToSparseGraph(grid, False)
    graph.convert()
    workspaces = load_workspaces_from_file(
        filename="workspaces_" + DEFAULT_WS_FILE)
    print(("Compute demo ", demo_id))

    hard = True
    trajectory = None
    while trajectory is None:
        if not hard:
            print("try easy")
        try:
            trajectory = compute_demonstration(
                workspaces[demo_id], graph, nb_points=nb_points,
                show_result=False, average_cost=False, verbose=True,
                no_linear_interpolation=hard)
        except ValueError as e:
            trajectory = None
            print("Warning : ", e)
        hard = False
    return trajectory


def generate_demonstrations(nb_points):
    grid = np.ones((nb_points, nb_points))
    graph = CostmapToSparseGraph(grid, options.average_cost)
    graph.convert()
    workspaces = load_workspaces_from_file(
        filename="workspaces_" + DEFAULT_WS_FILE)
    trajectories = [None] * len(workspaces)
    for k, workspace in enumerate(tqdm(workspaces)):
        if verbose:
            print(("Compute demo ", k))
        nb_tries = 0
        while trajectories[k] is None:
            nb_tries += 1
            hard = nb_tries < 20
            try:
                trajectories[k] = compute_demonstration(
                    workspace,
                    graph,
                    nb_points=nb_points,
                    # show_result=show_result,
                    show_result=(show_demo_id == k or options.show_result),
                    average_cost=options.average_cost,
                    verbose=verbose,
                    no_linear_interpolation=hard)
            except ValueError as e:
                trajectories[k] = None
                if verbose:
                    print("Warning : ", e)
        if show_demo_id == k and not options.show_result:
            break
    return trajectories


if __name__ == '__main__':

    np.random.seed(0)
    parser = optparse.OptionParser("usage: %prog [options] arg1 arg2")
    parser.add_option('--nb_points', type="int", default=24)
    add_boolean_options(parser, ['verbose', 'show_result', 'average_cost'])
    (options, args) = parser.parse_args()
    verbose = options.verbose
    show_demo_id = -1
    nb_points = options.nb_points
    print((" -- options : ", options))
    trajectories = generate_demonstrations(nb_points)
    save_trajectories_to_file(
        trajectories,
        filename='trajectories_' + DEFAULT_WS_FILE)
