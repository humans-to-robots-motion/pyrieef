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
# Jim Mainprice on Sunday June 17 2018

import demos_common_imports
from motion.motion_optimization import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

workspace = Workspace()
workspace.obstacles.append(Circle(np.array([0.2, .15]), .1))
workspace.obstacles.append(Circle(np.array([-.1, .15]), .1))
signed_distance_field = SignedDistanceWorkspaceMap(workspace)
extends = workspace.box.extends()
motion_optimization = MotionOptimization2DCostMap(
    T=20, n=2, extends=extends, signed_distance_field=signed_distance_field)

trajectory = Trajectory(motion_optimization.T)
x_init = -0.2 * np.ones(2)
x_goal = .3 * np.ones(2)


plt.figure(figsize=(7, 6.5))
plt.axis('equal')
plt.axis(workspace.box.box_extends())
colorst = [cm.gist_ncar(i) for i in np.linspace(
    0, 0.9, len(workspace.obstacles))]
for i, o in enumerate(workspace.obstacles):
    plt.plot(o.origin[0], o.origin[1], 'kx')
    points = o.sampled_points()
    X = np.array(points)[:, 0]
    Y = np.array(points)[:, 1]
    plt.plot(X, Y, color=colorst[i], linewidth=2.0)
    print "colorst[" + str(i) + "] : ", colorst[i]
plt.plot(x_init[0], x_init[1], 'ro')
plt.plot(x_goal[0], x_goal[1], 'bo')

nb_points = 100
xs = np.linspace(extends.x_min, extends.x_max, nb_points)
ys = np.linspace(extends.y_min, extends.y_max, nb_points)
X, Y = np.meshgrid(xs, ys)
Z = signed_distance_field(np.stack([X, Y]))
color_style = plt.cm.hot
color_style = plt.cm.bone
color_style = plt.cm.magma
im = plt.imshow(Z,
                extent=workspace.box.box_extends(),
                origin='lower',
                interpolation='bilinear',
                cmap=color_style)
plt.colorbar(im, fraction=0.05, pad=0.02)
cs = plt.contour(X, Y, Z, 16, cmap=color_style)
# plt.colorbar(cs, fraction=0.05, pad=0.02)
plot_3d = False
if plot_3d:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=color_style, linewidth=0, antialiased=False)

# Plot trajectory
trajectory = linear_interpolation_trajectory(
    x_init, x_goal, motion_optimization.T)
for i in range(100):
    [dist, trajectory] = motion_optimization.optimize(x_init, 1, trajectory)
for k in range(motion_optimization.T + 1):
    q = trajectory.configuration(k)
    plt.plot(q[0], q[1], 'ro')
    # plt.show(block=False)
    # plt.draw()
    # plt.pause(0.0001)
plt.show()
