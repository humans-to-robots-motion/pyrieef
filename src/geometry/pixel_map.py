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
#                                           Jim Mainprice on Sunday May 17 2015
import numpy as np
from differentiable_geometry import *
from scipy.interpolate import RectBivariateSpline


class Extends:

    def __init__(self, sides=.5):
        self.x_min = -sides
        self.x_max = sides
        self.y_min = -sides
        self.y_max = sides

    def x(self):
        return self.x_max - self.x_min

    def y(self):
        return self.y_max - self.y_min

    def sample_uniform(self):
        """ Sample uniformly point in extend"""
        pt = np.random.random(2)  # \in [0, 1]^2
        pt[0] = self.x() * pt[0] + self.x_min
        pt[1] = self.y() * pt[1] + self.y_min
        return pt


class PixelMap:
    """
    Implements an axis aligned regular pixel-grid map. It follows the
    convention  that we use on the C++ version of the class. which means
    that the min and max extends are the origin of the world coordinates.
    """

    def __init__(self, resolution, extends=Extends()):
        self.extends = extends
        self.resolution = resolution
        self.origin_minus = np.array([self.extends.x_min, self.extends.y_min])
        self.origin = self.origin_minus + 0.5 * self.resolution
        self.nb_cells_x = int(self.extends.x() / self.resolution)
        self.nb_cells_y = int(self.extends.y() / self.resolution)

    def world_to_matrix(self, x):
        """
        matrix coordinates allow to visualy compare with world on a flat screen
        these are the coordinates used for representing 2D environments
        """
        grid_coord = self.world_to_grid(x)
        return np.array([grid_coord[1], grid_coord[0]])

    def world_to_grid(self, x):
        """ grid coordinates of a point in world coordinates."""
        return np.floor((x - self.origin_minus) / self.resolution).astype(int)

    def grid_to_world(self, p):
        """world coorindates of the center of a grid cell"""
        return self.resolution * p.astype(float) + self.origin


class RegressedPixelGridSpline(DifferentiableMap):
    """
    Lightweight wrapper around the Analytical grid to implement the
    NDimZerothOrderFunction interface. Upon construction can decide whether the
    to use the finite-differenced Hessian or to replace it with the identity
    matrix.
    """

    def __init__(self, matrix, resolution, extends=Extends()):
        self._extends = extends
        x = np.arange(self._extends.x_min, self._extends.x_max, resolution)
        y = np.arange(self._extends.y_min, self._extends.y_max, resolution)
        self._interp_spline = RectBivariateSpline(x, y, matrix)

    def output_dimension(self): return 1

    def input_dimension(self): return 2

    def extends(self): return self._extends

    def forward(self, p):
        assert p.size == 2
        return self._interp_spline(p[0], p[1])

    def jacobian(self, p):
        assert p.size == 2
        J = np.matrix([[0., 0.]])
        J[0, 0] = self._interp_spline(p[0], p[1], dx=1)
        J[0, 1] = self._interp_spline(p[0], p[1], dy=1)
        return J
