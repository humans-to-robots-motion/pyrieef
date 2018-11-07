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

import numpy as np
from .differentiable_geometry import *
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
from itertools import product


def edt(image):
    return ndimage.distance_transform_edt(image == 0)


def sdf(image):
    dist1 = ndimage.distance_transform_edt(image == 0)
    dist2 = ndimage.distance_transform_edt(image == 1)
    return dist1 - dist2


class Extent:

    def __init__(self, sides=.5):
        self.x_min = -sides
        self.x_max = sides
        self.y_min = -sides
        self.y_max = sides

    def x(self):
        return self.x_max - self.x_min

    def y(self):
        return self.y_max - self.y_min


def sample_uniform(extent):
    """ Sample uniformly point in extend"""
    pt = np.random.random(2)  # in [0, 1]^2
    lower_corner = np.array([extent.x_min, extent.y_min])
    dim = np.array([extent.x(), extent.y()])
    return np.multiply(dim, pt) + lower_corner


class PixelMap:
    """
    Implements an axis aligned regular pixel-grid map. It follows the
    convention  that we use on the C++ version of the class. which means
    that the min and max extent are the origin of the world coordinates.
    """

    def __init__(self, resolution, extent=Extent()):
        self.extent = extent
        self.resolution = resolution
        self.origin_minus = np.array([self.extent.x_min, self.extent.y_min])
        self.origin = self.origin_minus + 0.5 * self.resolution
        self.nb_cells_x = int(self.extent.x() / self.resolution)
        self.nb_cells_y = int(self.extent.y() / self.resolution)

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

    def __init__(self, matrix, resolution, extent=Extent()):
        self._extends = extent
        x = np.arange(self._extends.x_min, self._extends.x_max, resolution)
        y = np.arange(self._extends.y_min, self._extends.y_max, resolution)
        self._interp_spline = RectBivariateSpline(x, y, matrix)

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return 2

    def extent(self):
        return self._extends

    def forward(self, p):
        # assert p.size == 2
        return self._interp_spline(p[0], p[1])

    def jacobian(self, p):
        assert p.size == 2
        J = np.matrix([[0., 0.]])
        J[0, 0] = self._interp_spline(p[0], p[1], dx=1)
        J[0, 1] = self._interp_spline(p[0], p[1], dy=1)
        return J


def costmap_from_matrix(extent, matrix):
    """ Creates a costmap wich is continuously defined given a matrix """
    assert matrix.shape[0] == matrix.shape[1]
    assert extent.x() == extent.y()
    resolution = extent.x() / matrix.shape[0]
    return RegressedPixelGridSpline(matrix, resolution, extent)


def two_dimension_function_evaluation(X, Y, phi):
    """
    Evaluates a function at X Y test points given by meshgrid

        x = y = np.linspace(min, max, n)
        X, Y = np.meshgrid(x, y)

    Parameters
    ----------
        X : numpy array (n, n)
        Y : numpy array (n, n)
        phi : function
    """
    assert X.shape[0] == X.shape[1] == Y.shape[0] == Y.shape[1]
    Z = np.zeros(X.shape)
    for i, j in product(range(X.shape[0]), range(X.shape[0])):
        Z[i, j] = phi(np.array([X[i, j], Y[i, j]]))
    return Z
