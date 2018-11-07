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

from __init__ import *
from numpy.testing import assert_allclose
from geometry.pixel_map import *
from itertools import product


def test_pixelmap_random():
    resolution = .2
    pixel_map = PixelMap(resolution)
    print("pm -- resolution : {}".format(pixel_map.resolution))
    print("pm -- origin : {}".format(pixel_map.origin))
    print("pm -- nb_cells_x : {}".format(pixel_map.nb_cells_x))
    print("pm -- nb_cells_y : {}".format(pixel_map.nb_cells_y))
    for i in range(100):
        p_g1 = pixel_map.world_to_grid(sample_uniform(pixel_map.extent))
        p_w1 = pixel_map.grid_to_world(p_g1)
        p_g2 = pixel_map.world_to_grid(p_w1)
        p_w2 = pixel_map.grid_to_world(p_g2)
        assert_allclose(p_w1, p_w2)
    print("Random pixel map OK !")


def test_pixelmap_meshgrid():
    resolution = .2
    pixel_map = PixelMap(resolution)
    print("pm -- resolution : {}".format(pixel_map.resolution))
    print("pm -- origin : {}".format(pixel_map.origin))
    print("pm -- nb_cells_x : {}".format(pixel_map.nb_cells_x))
    print("pm -- nb_cells_y : {}".format(pixel_map.nb_cells_y))
    extent = pixel_map.extent
    nb_points = int(extent.x_max - extent.x_min) / resolution
    print("nb_points : ", nb_points)
    x_min = extent.x_min + 0.5 * resolution
    x_max = extent.x_max - 0.5 * resolution
    y_min = extent.y_min + 0.5 * resolution
    y_max = extent.y_max - 0.5 * resolution
    x = np.linspace(x_min, x_max, nb_points)
    y = np.linspace(y_min, y_max, nb_points)
    print(x)
    print(y)
    X, Y = np.meshgrid(x, y)
    for i, j in product(list(range(x.size)), list(range(y.size))):
        p_w1 = np.array([X[i, j], Y[i, j]])
        p_g1 = pixel_map.world_to_grid(p_w1)
        p_w2 = pixel_map.grid_to_world(p_g1)
        assert_allclose(p_w1, p_w2)

    print("Test two_dimensional_function_eval")
    f = ExpTestFunction()
    x = y = np.linspace(-1., 1., 100)
    X, Y = np.meshgrid(x, y)
    Z1 = two_dimension_function_evaluation(X, Y, f)
    Z2 = f(np.stack([X, Y]))
    assert_allclose(Z1, Z2)


def test_regressed_grid():

    l = 0.5

    # Regularly-spaced, coarse grid
    ds1 = 0.1
    x = np.arange(-l, l, ds1)
    y = np.arange(-l, l, ds1)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(2 * X)**2 - (Y / 2)**2)

    # spline with grid data
    interp_spline = RectBivariateSpline(x, y, Z.T)

    # same but with DifferentiableMap structure
    f = RegressedPixelGridSpline(Z.T, ds1, Extent(l))

    # Regularly-spaced, fine grid

    ds2 = 0.05
    x2 = np.arange(-l, l, ds2)
    y2 = np.arange(-l, l, ds2)
    Z2 = interp_spline(x2, y2)
    g2_x = interp_spline(x2, y2, dx=1)  # Gradient x
    g2_y = interp_spline(x2, y2, dy=1)  # Gradient y

    # Function interpolation
    g1_x = np.zeros((x2.size, y2.size))
    g1_y = np.zeros((x2.size, y2.size))
    z1 = np.zeros((x2.size, y2.size))
    print("g1 : ", g1_x.shape)
    for i, x in enumerate(x2):
        for j, y in enumerate(y2):
            p = np.array([x, y])
            z1[i, j] = f.forward(p)
            grad = f.gradient(p)
            g1_x[i, j] = grad[0]  # Gradient x
            g1_y[i, j] = grad[1]  # Gradient y

    print(g1_x.shape)

    assert check_is_close(Z2, z1, 1e-10)
    assert check_is_close(g2_x, g1_x, 1e-10)
    assert check_is_close(g2_y, g1_y, 1e-10)


if __name__ == "__main__":
    test_pixelmap_random()
    test_pixelmap_meshgrid()
    test_regressed_grid()
