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
from pyrieef.motion.objective import *
import time


def plot_results(workspace, x_init, x_goal, trajectory, optimizer):

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D
    plt.figure(figsize=(7, 6.5))
    plt.axis('equal')
    plt.axis(workspace.box.box_extent())
    colorst = [cm.gist_ncar(i) for i in np.linspace(
        0, 0.9, len(workspace.obstacles))]
    for i, o in enumerate(workspace.obstacles):
        plt.plot(o.origin[0], o.origin[1], 'kx')
        points = o.sampled_points()
        X = np.array(points)[:, 0]
        Y = np.array(points)[:, 1]
        plt.plot(X, Y, color=colorst[i], linewidth=2.0)
        print("colorst[" + str(i) + "] : ", colorst[i])
    plt.plot(x_init[0], x_init[1], 'ro')
    plt.plot(x_goal[0], x_goal[1], 'bo')

    nb_points = 100
    X, Y = workspace.box.meshgrid(nb_points)
    Z = optimizer.sdf(np.stack([X, Y]))
    color_style = plt.cm.hot
    color_style = plt.cm.bone
    color_style = plt.cm.magma
    im = plt.imshow(Z,
                    extent=workspace.box.box_extent(),
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
        ax.plot_surface(X, Y, Z, cmap=color_style,
                        linewidth=0, antialiased=False)

    for k in range(optimizer.T + 1):
        q = trajectory.configuration(k)
        plt.plot(q[0], q[1], 'ro')
        # plt.show(block=False)
        # plt.draw()
        # plt.pause(0.0001)
    plt.show()


def trajectory_optimization():

    use_matplotlib = False

    workspace = Workspace()
    workspace.obstacles.append(Circle(np.array([0.2, .0]), .1))
    workspace.obstacles.append(Circle(np.array([-.2, .0]), .1))
    signed_distance_field = SignedDistanceWorkspaceMap(workspace)
    extent = workspace.box.extent()
    optimizer = MotionOptimization2DCostMap(
        T=20, n=2, box=workspace.box,
        signed_distance_field=signed_distance_field)
    optimizer.set_eta(.1)
    optimizer.set_scalars(
        term_potential_scalar=10000.,
        smoothness_scalar=400.
    )
    optimizer.create_clique_network()
    optimizer.add_all_terms()
    optimizer.create_objective()

    q_init = np.array([-.3, -.3])
    q_goal = np.array([+.4, .1])
    trajectory = linear_interpolation_trajectory(q_init, q_goal, 20)

    if use_matplotlib:
        t_start = time .time()
        [dist, trajectory, gradient, deltas] = optimizer.optimize(
            q_init=q_init, nb_steps=100, trajectory=trajectory)
        print("optimization took : {} sec.".format(time.time() - t_start))
        # Plot trajectory
        plot_results(workspace, x_init, x_goal, trajectory, optimizer)
    else:
        from pyrieef.rendering import workspace_planar
        from pyrieef.rendering import opengl
        viewer = workspace_planar.WorkspaceOpenGl(workspace)
        viewer.draw_ws_background(optimizer.obstacle_potential_from_sdf())
        # viewer.draw_ws_obstacles()

        for i in range(1000):
            [dist, trajectory, gradient, deltas] = optimizer.optimize(
                q_init=q_init, nb_steps=1, trajectory=trajectory)
            if i % 1 == 0:
                dx = -100. * deltas + trajectory.active_segment()
                g_traj = Trajectory(q_init=q_init, x=dx)
                for k in range(optimizer.T + 1):
                    q = trajectory.configuration(k)
                    viewer.draw_ws_circle(.01, q)
                    viewer.draw_ws_line(q, g_traj.configuration(k))
                viewer.show()
                time.sleep(0.02)
        input("Press Enter to continue...")

if __name__ == "__main__":
    trajectory_optimization()
    # import cProfile
    # cProfile.run('trajectory_optimization()')
