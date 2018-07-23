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
#                                        Jim Mainprice on Sunday June 17 2017


import test_common_imports
from geometry.workspace import *


def test_ellipse():

    ellipse = Ellipse()
    ellipse.a = 0.1
    ellipse.b = 0.2

    dist = ellipse.dist_from_border(np.array([0.3, 0.0]))
    print "dist = ", dist
    assert np.fabs(dist - 0.2) < 1.e-06

    dist = ellipse.dist_from_border(np.array([0.0, 0.3]))
    print "dist = ", dist
    assert np.fabs(dist - 0.1) < 1.e-06


def sample_circles(nb_circles):
    centers = np.random.rand(nb_circles, 2)
    radii = np.random.rand(nb_circles)
    return centers, radii


def test_sdf_jacobians():
    verbose = False
    centers, radii = sample_circles(nb_circles=10)
    circles = []
    for p, r in zip(centers, radii):
        # print("p : {}, r : {}".format(p, r))
        circles.append(Circle(p, r))
    for c in circles:
        signed_distance_field = SignedDistance2DMap(c)
        assert check_jacobian_against_finite_difference(
            signed_distance_field, verbose)


def test_sdf_workspace():
    workspace = Workspace()
    centers, radii = sample_circles(nb_circles=10)
    workspace.obstacles.append(Circle(centers[0], radii[0]))
    workspace.obstacles.append(Circle(centers[1], radii[1]))
    signed_distance_field = SignedDistanceWorkspaceMap(workspace)
    assert check_jacobian_against_finite_difference(signed_distance_field)


test_ellipse()
test_sdf_jacobians()
test_sdf_workspace()
