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

import demos_common_imports
import matplotlib.pyplot as plt
import numpy
from pyrieef.geometry.differentiable_geometry import *

def evaluate(s, x):
    return [s(np.array([x_i])) for x_i in x]


s_1 = Compose(Sigmoid(1), Scale(IdentityMap(1), 1))
s_2 = Compose(Sigmoid(1), Scale(IdentityMap(1), 2))
s_3 = Compose(Sigmoid(1), Scale(IdentityMap(1), 3))
s_4 = Compose(Sigmoid(1), Scale(IdentityMap(1), 10))

plt.figure()
x = np.linspace(-1., 1., 100)
plt.plot(x, evaluate(s_1, x), label="1")
plt.plot(x, evaluate(s_2, x), label="2")
plt.plot(x, evaluate(s_3, x), label="3")
plt.plot(x, evaluate(s_4, x), label="4")
plt.legend(
    bbox_to_anchor=(0.85, 1),
    loc='upper left',
    borderaxespad=0., fontsize="6")
plt.show()
