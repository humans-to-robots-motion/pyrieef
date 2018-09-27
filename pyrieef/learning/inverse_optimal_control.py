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

from __future__ import print_function
import common_imports
from motion.objective import *
from motion.trajectory import *
import numpy as np


class InverseOptimalControl:

    """ Abstract class for IOC problems """

    def __init__(self, nb_demonstrations):
        self._cost_function = None
        self._solutions = [None] * nb_demonstrations
        self._demonstrations = [None] * nb_demonstrations

    @abstractmethod
    def solution(self, env_id):
        raise NotImplementedError()

    @abstractmethod
    def on_step(self):
        raise NotImplementedError()


class Learch(InverseOptimalControl):

    def __init__(self, nb_demonstrations):
        InverseOptimalControl.__init__(nb_demonstrations)

    @abstractmethod
    def planning(self, env_id):
        raise NotImplementedError()

    @abstractmethod
    def supervised_learning(self):
        raise NotImplementedError()

    def one_step(self, iteration):

        # 1) step off the cost manifold
        self.planning()

        # 2) project solutions
        self.supervised_learning()


class Learch2D(Learch):

    def __init__(self, dataset):
        Learch.__init__(len(dataset))
        self._nb_points = 24
        self._goodness_scalar = .2
        self._goodness_stddev = .2

    def initialize_data(dataset):
        self._goodness_fields = [None] * len(dataset)
        for k, ws in enumerate(dataset):
            self._workspaces[k] = ws.workspace
            self._demonstrations[k] = ws.demonstrations[0]
            self._solutions[k] = linear_interpolation_trajectory(
                self._demonstrations[k].initial_configuration(),
                self._demonstrations[k].final_configuration())
            self._occupancy_maps[k] = occupancy_map(
                self._nb_points, self._workspaces[k])
            self._goodness_maps[k] = goodness_map(
                self._demonstrations[k],
                self._nb_points, self._workspaces[k].box.extent(),
                self._goodness_scalar,
                self._goodness_stddev)

    def planning(self):
        for workspace in self._workspaces[k]:
        return 0

    def supervised_learning(self):
        return 0


def goodness_map(trajectory, nb_points, box,
                 goodness_scalar,
                 goodness_stddev):
    pixelmap = pixelmap_from_box()
    occpancy_map = np.zeros((nb_points, nb_points))
    for waypoint in trajectory.list_configurations():
        occpancy_map[tuple(pixelmap.world_to_matrix(waypoint))] = 1
    return goodness_scalar * np.exp(-0.5 * (
        edt(occpancy_map) / goodness_stddev)**2)
