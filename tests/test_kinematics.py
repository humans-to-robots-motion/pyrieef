#!/usr/bin/env python

# Copyright (c) 2020, University of Stuttgart
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
#                                       Jim Mainprice on Monday January 27 2020

import __init__
from kinematics.homogeneous_transform import *
from kinematics.robot import *
from kinematics.planar_arm import *
from geometry.differentiable_geometry import *
from numpy.testing import assert_allclose


def test_planar_rotation():

    kinematic_map = PlanarRotation(np.ones(2))
    assert check_jacobian_against_finite_difference(kinematic_map)

    kinematic_map = PlanarRotation(np.array([.23, 12.]))
    assert check_jacobian_against_finite_difference(kinematic_map)


def test_homogeneous_transform():

    kinematic_map = HomogeneousTransform2D()
    assert_allclose(kinematic_map(np.zeros(3)), np.zeros(2))

    kinematic_map = HomogeneousTransform2D(np.ones(2))
    assert_allclose(kinematic_map(np.zeros(3)), np.ones(2))

    kinematic_map = HomogeneousTransform2D(np.ones(2))
    p1 = kinematic_map(np.array([1., 1., 0.]))
    p2 = np.array([2., 2.])
    assert_allclose(p1, p2)

    kinematic_map = HomogeneousTransform2D(np.ones(2))
    p1 = kinematic_map(np.array([1., 1., 0.785398]))
    p2 = np.array([1., 2.41421])
    assert_allclose(p1, p2, 1e-4)


def test_homogeneous_jacobian():

    kinematic_map = HomogeneousTransform2D(np.random.rand(2))

    print("----------------------")
    print("Check identity (J implementation) : ")
    for i in range(4):
        assert check_jacobian_against_finite_difference(kinematic_map)


def test_freeflyer():
    robot = Freeflyer()
    assert_allclose(robot.shape[0], [0, 0])
    assert_allclose(robot.shape[1], [0, 1])
    assert_allclose(robot.shape[2], [1, 1])
    assert_allclose(robot.shape[3], [1, 0])

    robot = Freeflyer(scale=.2)
    assert_allclose(robot.shape[0], [0, 0])
    assert_allclose(robot.shape[1], [0, .2])
    assert_allclose(robot.shape[2], [.2, .2])
    assert_allclose(robot.shape[3], [.2, 0])

    robot = create_robot_from_file()
    assert robot.name == "freeflyer"

    robot = create_freeflyer_from_segments()
    assert robot.name == "freeflyer"


def test_isometries():

    p = np.random.rand(2)
    affine2d = Isometry2D(.4, p)
    assert_allclose(
        affine2d.inverse().matrix(),
        np.linalg.inv(affine2d.matrix()))

    R = rand_rotation_3d_matrix()
    p = np.random.rand(3)
    affine3d = Isometry3D(R, p)
    assert_allclose(
        affine3d.inverse().matrix(),
        np.linalg.inv(affine3d.matrix()))

    R_1 = rand_rotation_3d_matrix()
    p_1 = np.random.rand(3)
    T_1 = Isometry3D(R_1, p_1)

    R_2 = rand_rotation_3d_matrix()
    p_2 = np.random.rand(3)
    T_2 = Isometry3D(R_2, p_2)

    T_3 = T_1 * T_2
    assert_allclose(
        T_3.matrix(),
        np.dot(T_1.matrix(), T_2.matrix()))


def test_planar_robot():

    robot = TwoLinkArm()
    robot.link_lengths[0] = 1.2
    robot.link_lengths[1] = 2.3

    for q in np.random.random((100, 2)):
        q_bound = np.pi
        q[0] = 2. * q_bound * q[0] - q_bound
        q[1] = 2. * q_bound * q[1] - q_bound
        robot.set_and_update(q)
        x1 = robot.wrist
        x2 = planar_arm_fk_pos(q, robot.link_lengths)
        assert_allclose(x1, x2)


def test_planar_robot_jacobian():
    verbose = False
    link_lengths = [1.2, 2.3]
    phi = TwoLinkArmAnalyticalForwardKinematics(link_lengths)
    for q in np.random.random((100, 2)):
        q_bound = np.pi
        q[0] = 2. * q_bound * q[0] - q_bound
        q[1] = 2. * q_bound * q[1] - q_bound
        J = phi.jacobian(q)
        J_diff = finite_difference_jacobian(phi, q)
        if verbose:
            print("J : ")
            print(J)
            print("J_diff : ")
            print(J_diff)
        assert_allclose(J, J_diff)


if __name__ == "__main__":
    # test_planar_rotation()
    # test_homogeneous_transform()
    # test_homogeneous_jacobian()
    # test_freeflyer()
    # test_isometries()
    test_planar_robot_jacobian()
