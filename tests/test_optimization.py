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
#                                        Jim Mainprice on Sunday June 17 2018

from __init__ import *
from scipy import optimize
import scipy
print(scipy.__version__)
import numpy as np


def test_optimization_module():
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    res = optimize.minimize(optimize.rosen, x0,
                            method='BFGS',
                            jac=optimize.rosen_der,
                            options={'gtol': 1e-6, 'disp': True})
    print res
    assert_allclose(res.jac, np.array(
        [9.93918700e-07,   4.21980188e-07, 2.23775033e-07,
         -6.10304485e-07,   1.34057054e-07]))
    return res.fun


if __name__ == "__main__":
    test_optimization_module()
