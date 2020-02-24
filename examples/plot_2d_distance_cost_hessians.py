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
#                                        Jim Mainprice on Wed February 12 2019

from demos_common_imports import *
import numpy as np
from pyrieef.geometry.workspace import *
from pyrieef.geometry.interpolation import *
from pyrieef.rendering.workspace_renderer import WorkspaceDrawer
from pyrieef.motion.cost_terms import ObstaclePotential2D
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import itertools


def eigsorted(A):
    """ Returns sorted eigen values and eigen vectors """
    vals, vecs = np.linalg.eigh(A)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def confidence_ellipse(mean, cov, ax, n_std=3.0, **kwargs):
    """ Plots a covariance ellipse centered at mean """
    vals, vecs = eigsorted(cov)
    if np.any(np.isnan(vals)):
        return
    if np.any(np.isnan(vecs)):
        return
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2. * n_std * np.sqrt(vals)
    ellipse = Ellipse(mean, width=w, height=h, angle=theta, **kwargs)
    return ax.add_patch(ellipse)


# Creates a workspace with just one circle
workspace = Workspace()
workspace.obstacles = [Circle(origin=[.0, .0], radius=0.1)]
renderer = WorkspaceDrawer(workspace)
sdf = SignedDistanceWorkspaceMap(workspace)
cost = ObstaclePotential2D(sdf, 1., 10.)

# Querries the hessian of the cost
nb_points = 10
X, Y = workspace.box.meshgrid(nb_points)
Sigma = []
for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
    p = np.array([X[i, j], Y[i, j]])
    H = cost.hessian(p)
    Sigma.append([p, np.linalg.inv(H)])

renderer.set_drawing_axis(i)
renderer.background_matrix_eval = False
renderer.draw_ws_background(Compose(RangeSubspaceMap(3, [0]), cost),
                            color_style=plt.cm.Blues)
for cov in Sigma:
    confidence_ellipse(cov[0], cov[1], renderer._ax,
                       n_std=.03, edgecolor='red', facecolor="none")
renderer.show()
