#!/usr/bin/env python

# Copyright (c) 2021
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
#                                    Jim Mainprice on Friday September 23 2021

import numpy as np
from math import pi
from math import cos, sin
from geometry.differentiable_geometry import *


def radian(q):
    return np.array([q_i * pi / 180 for q_i in q])


def transform_2d(theta, l):
    return np.array([[cos(theta), -sin(theta), l],
                     [sin(theta), cos(theta), 0],
                     [0, 0, 1]])


class TwoLinkArm:

    def __init__(self, q0=[0, 0]):
        self.shoulder = np.array([0, 0])
        self.link_lengths = [0, 1]
        self.set_and_update(q0)

    def set_and_update(self, q):
        self.q = q
        self.forward_kinematics()

    def forward_kinematics(self):
        T1 = transform_2d(self.q[0], 0)
        T2a = transform_2d(0, self.link_lengths[0])
        T2b = transform_2d(self.q[1], 0)
        T3 = transform_2d(0, self.link_lengths[1])
        T_e = T1 @ T2a @ T2b
        T_w = T_e @ T3
        self.elbow = T_e[:2, 2]
        self.wrist = T_w[:2, 2]

    def plot(self):
        plt.plot([self.shoulder[0], self.elbow[0]],
                 [self.shoulder[1], self.elbow[1]],
                 'r-')
        plt.plot([self.elbow[0], self.wrist[0]],
                 [self.elbow[1], self.wrist[1]],
                 'r-')
        plt.plot(self.shoulder[0], self.shoulder[1], 'ko')
        plt.plot(self.elbow[0], self.elbow[1], 'ko')
        plt.plot(self.wrist[0], self.wrist[1], 'ko')


def planar_arm_fk_pos(
        q=[0, 0],
        link_lengths=[0, 1]):
    """ 
    Analytical solution to the forward kinematics problem
    """
    assert(len(link_lengths) == 2)
    l1 = link_lengths[0]
    l2 = link_lengths[1]
    return np.array([
        l1 * cos(q[0]) + l2 * cos(q[0] + q[1]),
        l1 * sin(q[0]) + l2 * sin(q[0] + q[1])])


class TwoLinkArmAnalyticalForwardKinematics(DifferentiableMap):

    def __init__(self, link_lengths=[0, 1]):
        """
        phi : q -> R^2

        Analytical solution to the forward kinematics problem
        """
        self._link_lengths = link_lengths

    def output_dimension(self):
        return 2

    def input_dimension(self):
        return 2

    def forward(self, q):
        return planar_arm_fk_pos(q, self._link_lengths)

    def jacobian(self, q):
        l1 = self._link_lengths[0]
        l2 = self._link_lengths[1]
        q12 = q[0] + q[1]
        l2_sin_q12 = l2 * sin(q12)
        l2_cos_q12 = l2 * cos(q12)
        return np.array([
            [-l1 * sin(q[0]) - l2_sin_q12, - l2_sin_q12],
            [l1 * cos(q[0]) + l2_cos_q12, l2_cos_q12]])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    arm = TwoLinkArm(radian([60, 0]))
    arm.forward_kinematics()
    print("wrist : ", arm.wrist)
    arm.plot()
    plt.axis("equal")
    plt.show()
