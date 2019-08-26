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
from geometry.utils import *


def test_line_parameters():

    p1 = np.array([2, 3])
    p2 = np.array([6, 4])
    a_res = 1 / 4
    b_res = 5 / 2

    a, b = line_parameters(p1, p2)
    print("a : {}, b : {}".format(a, b))
    assert np.fabs(a - a_res) < 1e-6
    assert np.fabs(b - b_res) < 1e-6

    a, b = line_parameters(p2, p1)
    print("a : {}, b : {}".format(a, b))
    assert np.fabs(a - a_res) < 1e-6
    assert np.fabs(b - b_res) < 1e-6


def test_line_line_intersection():

    p1 = np.array([0, 0])
    p2 = np.array([1, 1])
    p3 = np.array([0, 1])
    p4 = np.array([1, 0])
    pr = np.array([0.5, 0.5])

    p = line_line_intersection(p1, p2, p3, p4)
    print("p : ", p)
    assert np.linalg.norm(p - pr) < 1e-6

    p = line_line_intersection_det(p1, p2, p3, p4)
    print("p : ", p)
    assert np.linalg.norm(p - pr) < 1e-6

    p1 = np.array([0, 0])
    p2 = np.array([3, 3])
    p3 = np.array([0, 3])
    p4 = np.array([3, 0])
    pr = np.array([1.5, 1.5])

    p = line_line_intersection(p1, p2, p3, p4)
    print("p : ", p)
    assert np.linalg.norm(p - pr) < 1e-6

    p = line_line_intersection_det(p1, p2, p3, p4)
    print("p : ", p)
    assert np.linalg.norm(p - pr) < 1e-6

    p1 = np.array([0, 0])
    p2 = np.array([3, 3])
    p3 = np.array([0, 3])
    p4 = np.array([.1, 2.9])
    pr = np.array([1.5, 1.5])

    p = line_line_intersection(p1, p2, p3, p4)
    print("p : ", p)
    assert np.linalg.norm(p - pr) < 1e-6

    p = line_line_intersection_det(p1, p2, p3, p4)
    print("p : ", p)
    assert np.linalg.norm(p - pr) < 1e-6



if __name__ == "__main__":
    test_line_parameters()
    test_line_line_intersection()
