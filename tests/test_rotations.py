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
#                                        Jim Mainprice on Tue August 24 2021

import __init__
from geometry.rotations import *
import numpy as np


def test_rotation_matrices():

    for theta1 in np.random.uniform(low=-3.14, high=3.14, size=100):
        R = rotation_matrix_2d_radian(theta1)
        theta2 = angle_from_matrix_2d(R)
        # print("theta : {} , R : \n {} ".format(theta1, R))
        # print("theta1 : ", theta1)
        # print("theta2 : ", theta2)
        assert np.isclose(theta1, theta2)

test_rotation_matrices()
