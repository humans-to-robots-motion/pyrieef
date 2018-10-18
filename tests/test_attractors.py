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
from geometry.attractors import *
from geometry.differentiable_geometry import *
from geometry.diffeomorphisms import *


# Makes sure the attractor.evaluates to 0 at the goal point.
def eval_attractor(phi):
    x_goal = np.random.rand(2)
    attractor = MakeAttractor(phi, x_goal)
    v = attractor.forward(x_goal)
    assert np.fabs(v) < 1.e-12
    return v


def test_attractor_identity():
    phi = IdentityMap(2)
    v = eval_attractor(phi)
    print("v (identity) : ", v)


def test_attractor_polar():
    phi = PolarCoordinateSystem()
    v = eval_attractor(phi)
    print("v (polar) : ", v)


if __name__ == "__main__":
    test_attractor_identity()
    test_attractor_polar()
