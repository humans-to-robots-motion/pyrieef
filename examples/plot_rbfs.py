#!/usr/bin/env python

# Copyright (c) 2020, University of Stuttgart
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
#                                        Jim Mainprice on Sunday June 22 2020

import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d

PLOT3D = False

mu1 = [1, 0]                # mean of rbf 1 (red)   -> (w[0])
mu2 = [-1, 0]               # mean of rbf 2 (blue)  -> (w[1])
w = np.array([10, -10])     # weight vector
stddev = 1

ti = np.linspace(-2.0, 2.0, 100)
xx, yy = np.meshgrid(ti, ti)
rbf1 = Rbf(mu1[0], mu1[1], 1, epsilon=stddev, function='gaussian')
rbf2 = Rbf(mu2[0], mu2[1], 1, epsilon=stddev, function='gaussian')
phi = np.stack([rbf1(xx, yy), rbf2(xx, yy)])      # create featuremap tensor
X, Y, Z = xx, yy, np.tensordot(w, phi, axes=1)    # calculate costfield

if not PLOT3D:
    plt.xlabel('x')
    plt.ylabel('y')
    plt.pcolor(X, Y, Z, cmap=cm.jet)
    plt.plot(mu1[0], mu1[1], 'o', ms=10, color='r')
    plt.plot(mu2[0], mu2[1], 'o', ms=10, color='b')
    plt.colorbar()
else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        X, Y, Z,
        cmap=cm.jet,
        linewidth=0, antialiased=False)
plt.show()
