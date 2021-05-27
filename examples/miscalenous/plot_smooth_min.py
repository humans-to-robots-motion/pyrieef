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

from demos_common_imports import *
import numpy as np
import matplotlib.pyplot as plt
from pyrieef.geometry.differentiable_geometry import *

alpha = np.linspace(-1., -20., 10)
for a in alpha:
    n = 100
    f = LogSumExp(2, a)
    x_p = np.linspace(-1., 1., n)
    x_n = -x_p
    y = np.array([0.] * n)
    for i in range(n):
        y[i] = f(np.array([x_p[i], x_n[i]]))
    plt.plot(x_p, y, label="alpha : {}".format(int(a)))
print("Done.")
plt.legend(
    bbox_to_anchor=(0.85, 1),
    loc='upper left',
    borderaxespad=0., fontsize="6")
plt.show()