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
from test_differentiable_geometry import *
from geometry.diffeomorphisms import *


def check_beta(alpha, beta, beta_inv):
    success = False
    for i in range(1000):
        eta = .001
        r = .2
        gamma = 3.
        dx = np.random.rand(1)[0]
        # print dx
        dy = beta(eta, r, gamma, dx)
        dx_new = beta_inv(eta, r, gamma, dy)
        if np.fabs(dx - dx_new) > 1.e-12:
            print(("test beta (", i, ")"))
            print((" -- value : ", dx))
            print((" -- forward : ", dy))
            print((" -- inverse : ", dx_new))
            print("Error.")
            success = False
            break

        # print "x = ", x, " , y = ", y , " , x_new = ", x_new
        # print "Ok."
        success = True
    return success


def check_diffeo_inverse(diffeomorphism, test_points):
    success = False
    for i, x in enumerate(test_points):
        y = diffeomorphism.forward(x)
        x_new = diffeomorphism.inverse(y)
        dist = np.linalg.norm(x - x_new)
        if dist > 1.e-12:
            print(("test (", i, ")"))
            print((" -- norm : ", np.linalg.norm(x)))
            print((" -- point : ", x))
            print((" -- forward : ", y))
            print((" -- inverse : ", x_new))
            print("Error.")
            success = False
            break
        success = True
    return success


def check_obstacle_inverse(obstacle):
    success = True
    o = obstacle.object()
    print(("center : ", o.origin))
    # print(("radius : ", o.radius))
    # np.random.seed(0)
    test_points = []
    if success:
        success = False
        for i in range(1000):
            x = 5. * np.random.rand(2) + o.origin
            if o.is_inside(x):
                continue
            test_points.append(x)
    return check_diffeo_inverse(obstacle, test_points)


def test_inverse_functions():

    assert check_beta(alpha_f, beta_f, beta_inv_f)
    assert check_beta(alpha2_f, beta2_f, beta2_inv_f)

    # TODO look why this one fails...
    # assert check_beta(alpha3_f, beta3_f, beta3_inv_f)

    # print("Test PolarCoordinateSystem")
    # obstacle = PolarCoordinateSystem()
    # assert check_jacobian_against_finite_difference(obstacle)
    # assert check_inverse(obstacle)

    # print("Test ElectricCircle")
    # obstacle = ElectricCircle()
    # assert check_jacobian_against_finite_difference(obstacle)
    # assert check_inverse(obstacle)

    # obstacle = AnalyticEllipse()
    # obstacle.set_alpha(alpha_f, beta_inv_f)
    # if check_inverse(obstacle):
    #     print "Analytic Ellipse OK !!!"
    # else:
    #     print "Analytic Ellipse Error !!!"

    print("Test AnalyticCircle")
    obstacle = AnalyticCircle()
    obstacle.set_alpha(alpha_f, beta_inv_f)
    assert check_jacobian_against_finite_difference(obstacle)
    assert check_obstacle_inverse(obstacle)

    print("Test AnalyticConvexPolygon")
    obstacle = AnalyticConvexPolygon(
        polygon=ellipse_polygon(.2, .1, [.0, .0], [.1, .0], 0.))
    obstacle.set_alpha(alpha_f, beta_inv_f)
    assert check_jacobian_against_finite_difference(obstacle)
    assert check_obstacle_inverse(obstacle)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # TODO !!!

    # test_points = np.random.rand(1000, 5)
    # softmax = SoftmaxDiffeomorphism()
    # check_diffeo_inverse(softmax, test_points)

    # circles = []
    # circles.append(AnalyticCircle(origin=[.1, .0], radius=0.1))
    # circles.append(AnalyticCircle(origin=[.1, .25], radius=0.05))
    # circles.append(AnalyticCircle(origin=[.2, .25], radius=0.05))
    # circles.append(AnalyticCircle(origin=[.0, .25], radius=0.05))

    # print("Test AnalyticMultiCircle")
    # obstacle = AnalyticMultiCircle(circles)

    # assert check_jacobian_against_finite_difference(obstacle)
    # assert check_inverse(obstacle)

    print("Done.")


if __name__ == "__main__":
    test_inverse_functions()
