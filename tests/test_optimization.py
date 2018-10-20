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
from scipy import optimize
import scipy
print((scipy.__version__))
import numpy as np
from geometry.differentiable_geometry import *
from motion.objective import *
from numpy.testing import assert_allclose


def test_optimization_module():
    print("**********************************************")
    print("TEST BFGS")
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    res = optimize.minimize(
        optimize.rosen, x0, method='BFGS',
        jac=optimize.rosen_der,
        options={'gtol': 1e-6, 'disp': True})
    print(res)
    assert_allclose(res.jac, np.array(
        [9.93918700e-07,   4.21980188e-07, 2.23775033e-07,
         -6.10304485e-07,   1.34057054e-07]))
    return res.fun


def test_optimization_trust():
    print("**********************************************")
    print("TEST Newton trust region ")
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    res = optimize.minimize(
        optimize.rosen, x0, method='trust-ncg',
        jac=optimize.rosen_der,
        hess=optimize.rosen_hess,
        options={'gtol': 1e-8, 'disp': True})
    print(res.x)
    print(optimize.rosen(x0).shape)
    print(optimize.rosen_der(x0).shape)
    print(optimize.rosen_hess(x0).shape)
    return res.fun


def test_quadric():
    np.random.seed(0)
    dim = 3
    # Symetric positive definite case
    k = np.matrix(np.random.rand(dim, dim))
    f = QuadricFunction(                # g = x'Ax + b'x + c
        k.T * k,              # A
        np.random.rand(dim),            # b
        1.)                             # c

    gtol = 1e-6
    x0 = [1.3, 0.7, 0.8]
    res = optimize.minimize(
        f,
        x0,
        method='BFGS',
        jac=f.gradient,
        options={'gtol': gtol, 'disp': True})
    print("- res.jac : {}".format(res.jac.shape))
    print("-   zeros : {}".format(np.zeros(dim).shape))
    assert_allclose(res.jac, np.zeros(dim), atol=gtol)


def test_motion_optimimization_2d():
    print("Checkint Motion Optimization")
    trajectory = linear_interpolation_trajectory(
        q_init=np.zeros(2), q_goal=np.ones(2), T=20)
    objective = MotionOptimization2DCostMap(
        T=trajectory.T(),
        q_init=trajectory.initial_configuration(),
        q_goal=trajectory.final_configuration())
    objective.create_clique_network()
    objective.add_init_and_terminal_terms()
    objective.add_smoothness_terms(1)
    objective.create_objective()
    gtol = 1e-7
    assert check_jacobian_against_finite_difference(
        objective.objective, verbose=False)
    res = optimize.minimize(
        objective.objective,
        trajectory.active_segment(),
        method='BFGS',
        jac=objective.objective.gradient,
        options={'gtol': gtol, 'disp': True})
    # objective.optimize(q_init=np.zeros(2), trajectory=trajectory)
    print(trajectory.x().shape)
    print(res.x.shape)
    print(res)
    print(trajectory.x())
    # print "- res.jac : {}".format(res.jac.shape)
    print("max : ", max(res.jac))
    print("jac : ", res.jac)
    assert_allclose(res.jac, np.zeros(res.jac.size), atol=1e-1)

if __name__ == "__main__":
    test_optimization_module()
    test_optimization_trust()
    test_quadric()
    test_motion_optimimization_2d()
