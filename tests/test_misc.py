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
#                                     Jim Mainprice on Thursday October 28 2021
import numpy as np
import math
import timeit


def test_1():
    math.sqrt(.7774)


def test_2():
    np.sqrt(.7774)


def test_3():
    np.random.seed(0)
    x = np.random.rand(100000)
    res = np.vectorize(math.sqrt)(x)

def test_4():
    np.random.seed(0)
    x = np.random.rand(100000)
    res = np.sqrt(x)


print("time in sec : ", timeit.timeit(stmt=test_1, number=100000))
print("time in sec : ", timeit.timeit(stmt=test_2, number=100000))
print("time in sec : ", timeit.timeit(stmt=test_3, number=1))
print("time in sec : ", timeit.timeit(stmt=test_4, number=1))
