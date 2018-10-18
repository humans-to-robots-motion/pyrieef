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

from scipy import optimize
import numpy as np
import time


def newton_optimize_trajectory(
        objective,
        trajectory,
        verbose=False,
        maxiter=15):
    t_start = time.time()
    res = optimize.minimize(
        x0=trajectory.active_segment(),
        method='Newton-CG',
        fun=objective.forward,
        jac=objective.gradient,
        hess=objective.hessian,
        options={'maxiter': maxiter, 'disp': verbose}
    )
    if verbose:
        print(("optimization done in {} sec.".format(time.time() - t_start)))
        print(("gradient norm : ", np.linalg.norm(res.jac)))
    trajectory.active_segment()[:] = res.x
    return res
