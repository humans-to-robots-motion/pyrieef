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
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyrieef.geometry.differentiable_geometry import *


class ExpF(DifferentiableMap):

    def output_dimension(self): return 1

    def input_dimension(self): return 2

    def forward(self, p):
        assert p.size == 2
        return np.exp(-(2 * p[0])**2 - (p[1] / 2)**2)

# Regularly-spaced, coarse grid
dx, dy = 0.4, 0.4
xmax, ymax = 2, 4
x = np.arange(-xmax, xmax, dx)
y = np.arange(-ymax, ymax, dy)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(2 * X)**2 - (Y / 2)**2)
print(Z.shape)

interp_spline = RectBivariateSpline(x, y, Z.transpose())

# Regularly-spaced, fine grid
dx2, dy2 = 0.16, 0.16
x2 = np.arange(-xmax, xmax, dx2)
y2 = np.arange(-ymax, ymax, dy2)
Z2 = interp_spline(x2, y2)
g2_x = interp_spline(x2, y2, dx=1)  # Gradient x
g2_y = interp_spline(x2, y2, dy=1)  # Gradient y

f = ExpF()
g1_x = np.zeros((x2.size, y2.size))
g1_y = np.zeros((x2.size, y2.size))
z1 = np.zeros((x2.size, y2.size))
print("g1 : ", g1_x.shape)
for i, x in enumerate(x2):
    for j, y in enumerate(y2):
        p = np.array([x, y])
        z1[i, j] = f.forward(p)
        grad = f.gradient(p)
        g1_x[i, j] = grad[0]  # Gradient x
        g1_y[i, j] = grad[1]  # Gradient y


print("g1_x : \n", g1_x)
print("g2_x : \n", g2_x)

plot3d = False
if plot3d:
    X2, Y2 = np.meshgrid(x2, y2)
    fig, ax = plt.subplots(nrows=1, ncols=3, subplot_kw={'projection': '3d'})
    ax[0].plot_wireframe(X, Y, Z, color='k')
    ax[1].plot_wireframe(X2, Y2, Z2.transpose(), color='k')
    ax[2].plot_wireframe(X2, Y2, z1.transpose(), color='k')
    for axes in ax:
        axes.set_zlim(-0.2, 1)
        axes.set_axis_off()
else:
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.3)
    extent = (-3, 4, -4, 3)

    def plot_matrix(idx, mat, title):
        plt.subplot(3, 3, idx)
        plt.title(title)
        im = plt.imshow(mat, extent=extent)
        plt.colorbar(im, fraction=0.046, pad=0.04)

    plot_matrix(1, Z, "Coarse Values")
    plot_matrix(2, z1.transpose(), "Values ")
    plot_matrix(3, Z2.transpose(), "Interpolated Values")

    plot_matrix(5, g1_x.transpose(), "X Gradient")
    plot_matrix(6, g2_x.transpose(), "Interpolated X Gradient")

    plot_matrix(8, g1_y.transpose(), "Y Gradient")
    plot_matrix(9, g2_y.transpose(), "Interpolated Y Gradient")

fig.tight_layout()
plt.show()
