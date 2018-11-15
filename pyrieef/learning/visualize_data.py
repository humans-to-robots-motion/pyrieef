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

import common_imports
import matplotlib.pyplot as plt

from dataset import *
from utils.options import *
from utils.misc import *
import rendering.workspace_renderer as render
import time

# if running in the
# The default python provided in (Ana)Conda is not a framework build. However,
# the Conda developers have made it easy to install a framework build in both
# the main environment and in Conda envs. To use this install python.app conda
# install python.app and use pythonw rather than python


def draw_one_data_point(fig, lim,
                        occ, sdf, cost,
                        numb_rows=1, row=0, workspace=None):

    # x_min, x_max, y_min, y_max
    extent = np.array([lim[0][0], lim[0][1], lim[1][0], lim[1][1]])

    # print(numb_rows)
    ax0 = fig.add_subplot(numb_rows, 3, 1 + 3 * row)
    image_0 = plt.imshow(occ.T, extent=extent, origin='lower')
    ax1 = fig.add_subplot(numb_rows, 3, 2 + 3 * row)
    image_1 = plt.imshow(sdf.T, extent=extent, origin='lower')
    ax2 = fig.add_subplot(numb_rows, 3, 3 + 3 * row)
    image_2 = plt.imshow(cost.T, extent=extent, origin='lower')

    if workspace is not None:
        for circle in workspace.obstacles:
            ax0.plot(circle.origin[0], circle.origin[1], "rx")

    draw_fontsize = 5

    ax0.tick_params(labelsize=draw_fontsize)
    ax1.tick_params(labelsize=draw_fontsize)
    ax2.tick_params(labelsize=draw_fontsize)

    if row == 0:
        ax0.set_title('Occupancy', fontsize=draw_fontsize)
        ax1.set_title('Signed Distance Field', fontsize=draw_fontsize)
        ax2.set_title('Chomp Cost', fontsize=draw_fontsize)


def draw_all_costmaps(basename='1k_small.hdf5'):

    data = dict_to_object(
        load_dictionary_from_file(filename='costdata2d_' + str(basename)))
    print("Data is now loaded !!!")
    print((" -- size : ", data.size))
    print((" -- lims : ", data.lims))
    print((" -- datasets.shape : ", data.datasets.shape))
    workspaces = load_workspaces_from_file(filename='workspaces_' + basename)

    print(" -- displaying with matplotlib...")
    dataset_id = 0
    for data1, data2, data3, data4 in zip(*[iter(data.datasets)] * 4):
        fig = plt.figure(figsize=(5, 6))
        # fig = plt.figure(figsize=(8, 9))

        ws1 = workspaces[dataset_id + 0]
        ws2 = workspaces[dataset_id + 1]
        ws3 = workspaces[dataset_id + 2]
        ws4 = workspaces[dataset_id + 3]

        draw_one_data_point(
            fig, data.lims, data1[0], data1[1], data1[2], 4, 0, ws1)
        draw_one_data_point(
            fig, data.lims, data2[0], data2[1], data2[2], 4, 1, ws2)
        draw_one_data_point(
            fig, data.lims, data3[0], data3[1], data3[2], 4, 2, ws3)
        draw_one_data_point(
            fig, data.lims, data4[0], data4[1], data4[2], 4, 3, ws4)

        dataset_id += 4
        # centers =
        plt.show(block=False)
        plt.draw()
        plt.pause(0.0001)

        # raw_input("Press [enter] to continue.")
        plt.close(fig)


def draw_grids(data):
    """ This function draws two images next to each other. """
    fig = plt.figure(figsize=(5, 2))
    draw_one_data_point(fig, data[0], data[1], data[2])
    plt.show(block=False)
    plt.draw()
    plt.pause(0.0001)

    # raw_input("Press [enter] to continue.")
    plt.close(fig)


def draw_all_workspaces(basename, scale, multicol=True):
    dataset = load_workspace_dataset(basename)
    rows = 1
    cols = 1
    t_sleep = 0.4
    if multicol:
        rows = 3
        cols = 4
        t_sleep = 3.
    for workspaces in zip(*[iter(dataset)] * (rows * cols)):
        viewer = render.WorkspaceDrawer(
            workspaces[0].workspace,
            wait_for_keyboard=True, rows=rows, cols=cols, scale=scale)
        for k, ws in enumerate(workspaces):
            viewer.set_drawing_axis(k)
            viewer.set_workspace(ws.workspace)
            viewer.draw_ws_img(ws.costmap)
            viewer.draw_ws_obstacles()
            if ws.demonstrations:
                for trajectory in ws.demonstrations:
                    configurations = trajectory.list_configurations()
                    viewer.draw_ws_line(configurations, color="r")
                    viewer.draw_ws_point(configurations[0], color="k")
                    dt = 0.1
                    for t in range(1, trajectory.T() + 1):
                        # the acceleration at t = 0 is 0
                        # we can insert it in the data or not..
                        a_t = trajectory.acceleration(t, dt)
                        v_t = trajectory.velocity(t, dt)
                        q_t = trajectory.configuration(t)
                        # Integration in time
                        # it is normal that the acceleration does not
                        # correlate with the direction of motion because
                        # the velocity should, not acceleration
                        q_t_1 = v_t * dt + a_t * (dt**2)
                        q_t_0 = v_t * dt
                        scale_direction = 4.
                        # viewer.draw_ws_line_fill(
                        #     [q_t, q_t + scale_direction * q_t_0], color="k")
                        viewer.draw_ws_point(q_t + q_t_1, color="k", shape='o')

        viewer.show_once()
        # time.sleep(t_sleep)


if __name__ == '__main__':

    parser = optparse.OptionParser("usage: %prog [options]")
    add_boolean_options(
        parser,
        ['verbose',          # prints debug information
         'trajectories',     # displays the trajectories
         'costmaps',         # displays the costmaps
         ])
    parser.add_option('--basename', type='string', default='1k_small.hdf5')
    parser.add_option('--scale', type='float', default=1.3)
    options, args = parser.parse_args()

    if options.costmaps:
        draw_all_costmaps(options.basename)
    elif options.trajectories:
        draw_all_workspaces(options.basename, options.scale)
    else:
        print((parser.print_help()))
