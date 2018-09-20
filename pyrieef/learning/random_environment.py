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
from geometry.workspace import *
from math import *
from random import *
import optparse
import os
from dataset import *
from tqdm import tqdm


def samplerandpt(lims):
    """
        Sample a random point within limits
    """
    pt = np.array(lims.shape[0] * [0.])
    for j in range(pt.size):
        pt[j] = lims[j][0] + random() * (lims[j][1] - lims[j][0])
    return pt


def chomp_obstacle_cost(min_dist, epsilon):
    """ 
        Compute the cost function now (From CHOMP paper)
        If min_dist < 0, cost = -min_dist + epsilon/2
        If min_dist >= 0 && min_dist < epsilon, have a different cost 
        If min_dist >= epsilon, cost = 0
    """
    cost = 0.
    if min_dist < 0:
        cost = - min_dist + 0.5 * epsilon
    elif min_dist <= epsilon:
        cost = (1. / (2 * epsilon)) * ((min_dist - epsilon) ** 2)
    return cost


def grids(workspace, grid_to_world, epsilon):
    """
        Creates a boolean matrix of occupancies
        To convert it to int or floats, use the following
        matrix.astype(int)
        matrix.astype(float)
    """
    # print "grid_to_world.shape : ", grid_to_world.shape
    occupancy = np.zeros((grid_to_world.shape[0], grid_to_world.shape[1]))
    sdf = np.zeros((grid_to_world.shape[0], grid_to_world.shape[1]))
    costs = np.zeros((grid_to_world.shape[0], grid_to_world.shape[1]))
    # return [None, None, None]
    for i in range(grid_to_world.shape[0]):
        for j in range(grid_to_world.shape[1]):
            [min_dist, obstacle_id] = workspace.min_dist(grid_to_world[i, j])
            sdf[i, j] = min_dist
            occupancy[i, j] = min_dist <= 0.
            costs[i, j] = chomp_obstacle_cost(min_dist, epsilon)
    return [occupancy, sdf, costs]


def sample_circle_workspace(box,
                            nobjs_max=3,
                            random_max=False,
                            maxnumtries=100):
    """ Samples a workspace made of a maximum of 
        nobjs_max circles that do not intersect 
        todo replace the random environment script to use this function """
    workspace = Workspace(box)
    extends = box.extends()
    lims = np.array([[0., 1.], [0., 1.]])
    lims[0][1] = extends.x_max
    lims[0][0] = extends.x_min
    lims[1][1] = extends.y_max
    lims[1][0] = extends.y_min
    diagonal = box.diag()
    minrad = .10 * diagonal
    maxrad = .15 * diagonal
    nobj = nobjs_max if not random_max else int(ceil(random() * nobjs_max))
    found = False
    for numtries in range(maxnumtries):
        r = minrad + random() * (maxrad - minrad)
        c = samplerandpt(lims)
        [min_dist, obstacle_id] = workspace.min_dist(c)
        if min_dist >= (r + .01 * diagonal):
            workspace.add_circle(c, r)
        if len(workspace.obstacles) >= nobj:
            return workspace
    return None


def random_environments(opt):

    ndim = 2
    lims = np.array([[0., 1.], [0., 1.]])
    # size        = torch.LongStorage({opt.xsize, opt.ysize}) # col x row
    size = np.array([opt.xsize, opt.ysize])
    numdatasets = opt.numdatasets
    maxnobjs = opt.maxnumobjs
    minrad = opt.minobjrad
    maxrad = opt.maxobjrad
    epsilon = opt.epsilon
    padding = 3
    resolution_x = 1. / opt.xsize
    resolution_y = 1. / opt.ysize
    save_workspace = True
    if opt.seed >= 0:
        print  "set random seed ({})".format(opt.seed)
        np.random.seed(opt.seed)

    if resolution_x != resolution_y:
        print "Warning : resolution_x != resolution_y"
    else:
        resolution = resolution_x

    # Create a bunch of datasets
    datasets = [None] * numdatasets
    dataws = [None] * numdatasets
    k = 0

    # Create structure that contains grids and obstacles
    # The box which defines the workspace, is axis aligned
    # and it's origin is at the center
    box = Box()
    box.dim[0] = lims[0][1] - lims[0][0]
    box.dim[1] = lims[1][1] - lims[1][0]
    box.origin[0] = box.dim[0] / 2.
    box.origin[1] = box.dim[1] / 2.
    grid = PixelMap(resolution, box.extends())
    grid_to_world = np.zeros((grid.nb_cells_x, grid.nb_cells_y, 2))
    for i in range(grid.nb_cells_x):
        for j in range(grid.nb_cells_y):
            grid_to_world[i, j] = grid.grid_to_world(np.array([i, j]))

    # Try for this many time to do any one single thing before restarting
    maxnumtries = 100
    print("Num datasets : " + str(numdatasets))
    for k in tqdm(range(numdatasets)):

        # Create empty workspace.
        workspace = Workspace(box)
        numtries = 0  # Initialize num tries
        # nobj = int(ceil(random() * maxnobjs))
        nobj = maxnobjs
        while True:
            r = minrad + random() * (maxrad - minrad)
            c = samplerandpt(lims)
            # If this object is reasonably far away from other objects
            [min_dist, obstacle_id] = workspace.min_dist(c)
            if True or min_dist >= (r + 0.1):
                workspace.add_circle(c, r)
            numtries = numtries + 1  # Increment num tries

            # Go further only if we have not exceeded all tries
            if len(workspace.obstacles) >= nobj or numtries >= maxnumtries:
                # Compute the occupancy grid and the cost
                # Needs states in Nx2 format
                [occ, sdf, cost] = grids(workspace, grid_to_world, epsilon)
                datasets[k] = np.array([occ, sdf, cost])

                if save_workspace:
                    ws_c = -1000. * np.ones((maxnobjs, 2))
                    ws_r = -1000. * np.ones((maxnobjs, 2))
                    for i, o in enumerate(workspace.obstacles):
                        ws_c[i, :] = o.origin
                        ws_r[i, 0] = o.radius
                    dataws[k] = np.array([ws_c, ws_r])

                if opt.display:
                    draw_grids([occ, sdf, cost])

                k = k + 1
                break
        else:
            print('[OBJS] Reached max number of tries. Restarting run...')

    data = {}
    data["lims"] = lims
    data["size"] = size
    data["datasets"] = np.stack(datasets)

    print np.stack(datasets).shape
    print np.stack(dataws).shape
    workspaces = {}
    workspaces["lims"] = lims
    workspaces["size"] = size
    workspaces["datasets"] = np.stack(dataws)

    return data, workspaces


if __name__ == '__main__':

    parser = optparse.OptionParser("usage: %prog [options] arg1 arg2")

    parser.add_option('--numdatasets',
                      default=1000, type="int", dest='numdatasets',
                      help='Number of datasets to generate')
    parser.add_option('--savefilename',
                      default='2dcostdata.t7', type="string", dest='savefilename',
                      help='Filename to save results in (in local directory)')
    parser.add_option('--savematlabfile',
                      default=False, type="int", dest='savematlabfile',
                      help='Save results in .mat format')
    parser.add_option('--xsize',
                      default=24, type="int", dest='xsize',
                      help='Size of the x-dimension (in pixels). X values go from 0-1')
    parser.add_option('--ysize',
                      default=24, type="int", dest='ysize',
                      help='Size of the y-dimension (in pixels). Y values go from 0-1')
    parser.add_option('--maxnumobjs',
                      default=5, type="int", dest='maxnumobjs',
                      help='Maximum number of obst. per scene (ranges from 1-this number)')
    parser.add_option('--minobjrad',
                      default=0.05, type="float", dest='minobjrad',
                      help='Minimum radius of any obstacle (in m)')
    parser.add_option('--maxobjrad',
                      default=0.15, type="float", dest='maxobjrad',
                      help='Maximum radius of any obstacle (in m)')
    parser.add_option('--epsilon',
                      default=0.1, type="float", dest='epsilon',
                      help='Distance from obstacle at which obstacle cost zeroes out (in m)')
    parser.add_option('--display',
                      default=False, type="int", dest='display',
                      help='If set, displays the obstacle costs/occ grids in 2D')
    parser.add_option('--seed',
                      default=0, type="int", dest='seed',
                      help='Random number seed. -ve values mean random seed')

    (options, args) = parser.parse_args()
    # if len(args) != 2:
    #     parser.error("incorrect number of arguments")

    datasets, workspaces = random_environments(options)
    write_dictionary_to_file(datasets, filename='costdata2d_1k_small.hdf5')
    write_dictionary_to_file(workspaces, filename='workspaces_1k_small.hdf5')
