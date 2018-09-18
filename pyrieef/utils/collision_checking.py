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

from geometry.workspace import *
from motion.trajectory import *
import numpy as np


def collision_check_trajectory(workspace, trajectory):
    """ Check trajectory for collision """
    delta = workspace.box.diag() / 100.
    interpolated_traj = trajectory.continuous_trajectory()
    length = interpolated_traj.length()
    for s in np.linspace(0, 1, num=int(length / delta) + 1):
        if workspace.in_collision(
                interpolated_traj.configuration_at_parameter(s)):
            return True
    return False


def collision_check_linear_interpolation(workspace, p_init, p_goal):
    """ Check interior interpolation for collision """
    delta = workspace.box.diag() / 100.
    length = np.linalg.norm(p_init - p_goal)
    for s in np.linspace(0, 1, num=int(length / delta) + 1):
        p_interp = (1. - s) * p_init + s * p_goal
        if workspace.in_collision(p_interp):
            return True
    return False
