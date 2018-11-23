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

import __init__
from learning.random_environment import *
from learning.random_paths import *
import learning.demonstrations as demos
from graph.shortest_path import *
from geometry.workspace import sample_workspace
import time
import sys


def test_random_enviroments():
    sys.argn = 0
    sys.argv = []
    options = RandomEnvironmentOptions().get_options()
    options.numdatasets = 5
    datasets, workspaces = random_environments(options)
    assert len(workspaces["datasets"]) == options.numdatasets
    assert len(datasets["datasets"]) == options.numdatasets


def test_standard_dataset():
    options = RandomEnvironmentOptions("costdata2d_55k_28").get_options()
    options.numdatasets = 5
    datasets, workspaces = random_environments(options)
    assert len(workspaces["datasets"]) == options.numdatasets
    assert len(datasets["datasets"]) == options.numdatasets


def test_demonstrations():
    nb_demonstrations = 3
    nb_points = 24
    average_cost = False
    converter = CostmapToSparseGraph(
        np.ones((nb_points, nb_points)), average_cost)
    converter.convert()
    trajectories = [None] * nb_demonstrations
    t_start = time.time()
    demos.MAX_ITERATIONS = 2
    for k in range(nb_demonstrations):
        trajectories[k] = demos.compute_demonstration(
            sample_workspace(nb_circles=3), converter, nb_points=nb_points,
            show_result=False, average_cost=average_cost, verbose=True,
            no_linear_interpolation=True)
        # TODO fix this test, it takes too long for now.
        if trajectories[k] is not None:
            assert trajectories[k].n() == 2
            assert trajectories[k].T() == demos.TRAJ_LENGTH - 1
    print("time : {} sec.".format(time.time() - t_start))


if __name__ == "__main__":
    test_random_enviroments()
    test_standard_dataset()
    test_demonstrations()
