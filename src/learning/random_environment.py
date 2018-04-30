import common_imports
from workspace import *
from math import *
from random import *
import optparse
import matplotlib.pyplot as plt

def DrawGrids(occ, cost):
    fig = plt.figure(figsize=(4, 2))
    fig.add_subplot(1, 2, 1)
    image_0 = plt.imshow(occ)
    fig.add_subplot(1, 2, 2)
    image_1 = plt.imshow(cost)
    plt.show(block=False)
    plt.draw()
    plt.pause(0.0001)
    plt.close(fig)

# Sample a random point within limits
def samplerandpt(lims):
    pt = np.array(lims.shape[0]*[0.])
    for j in range(pt.size):
        pt[j] = lims[j][0] + random() * (lims[j][1] - lims[j][0])
    return pt

# Compute the cost function now (From CHOMP paper)
# If min_dist < 0, cost = -min_dist + epsilon/2
# If min_dist >= 0 && min_dist < epsilon, have a different cost 
# If min_dist >= epsilon, cost = 0
def ChompObstacleCost(min_dist, epsilon):
    cost = 0.
    if min_dist < 0 : 
        costs = - min_dist + 0.5 * epsilon
    elif min_dist <= epsilon :
        costs = (1./(2*epsilon))*((min_dist - epsilon) ** 2)
    return cost

# Creates a boolean matrix of occupancies
# To convert it to int or floats, use the following
# matrix.astype(int)
# matrix.astype(float)
def Grids(workspace, resolution, epsilon):
    grid = PixelMap(resolution, Extends(workspace.box.dim[0]/2.))
    occupancy   = np.zeros((grid.nb_cells_x, grid.nb_cells_y))
    costs       = np.zeros((grid.nb_cells_x, grid.nb_cells_y))
    for i in range(grid.nb_cells_x):
        for j in range(grid.nb_cells_y):
            pt = grid.grid_to_world(np.array([i, j]))
            min_dist = workspace.MinDist(pt)
            occupancy[i, j] = min_dist <= 0.
            costs[i, j] = min_dist
            # costs[i, j] = ChompObstacleCost(min_dist, epsilon)
    return [occupancy, costs]

def RandomEnvironments(opt):

    ndim        = 2
    lims        = np.array([[0., 1.], [0., 1.]])
    # size        = torch.LongStorage({opt.xsize, opt.ysize}) # col x row
    numdatasets = opt.numdatasets
    maxnobjs    = opt.maxnumobjs
    minrad      = opt.minobjrad
    maxrad      = opt.maxobjrad
    epsilon     = opt.epsilon
    padding     = 3
    resolution_x = 1./opt.xsize
    resolution_y = 1./opt.ysize

    if resolution_x != resolution_y:
        print "Warning : resolution_x != resolution_y"
    else:
        resolution = resolution_x

    # Create a bunch of datasets
    maxnobjs = 3
    datasets = [None] * numdatasets
    k = 1
    

    # Drawing
    # plt.ion() # enables interactive mode
    # fig = plt.figure(figsize=(1, 2))
    # fig.add_subplot(1, 2, 1)
    # image_0 = plt.imshow(np.zeros((opt.xsize, opt.ysize)))
    # fig.add_subplot(1, 2, 2)
    # image_1 = plt.imshow(np.zeros((opt.xsize, opt.ysize))) 
    # plt.show(block=False)
    

    # Try for this many time to do any one single thing before restarting
    maxnumtries = 100
    while k < numdatasets:

        # Create structure that contains grids and obstacles
        workspace=Workspace(Box(
            np.array([lims[0][0], lims[1][0]]), 
            np.array([lims[0][1]-lims[0][0], lims[1][1]-lims[1][0]])))

        if k%10 == 0 : print('Dataset: ' + str(k) + '/' + str(numdatasets))
        numtries = 0 # Initialize num tries
        nobj = int(ceil(random() * maxnobjs))
        while len(workspace.obstacles) < nobj and numtries < maxnumtries :
            r = minrad + random() * (maxrad - minrad)
            c = samplerandpt(lims)
            # If this object is reasonably far away from other objects
            min_dist = workspace.MinDist(c)
            # print "numtries : {}, nobj : {},\
            #  c : {}, r : {}, min_dist : {}".format(
            #     numtries, nobj, c, r, min_dist)
            if min_dist >= (r + 0.1):
                workspace.AddCircle(c, r)
            numtries = numtries+1  # Increment num tries

            # Go further only if we have not exceeded all tries
            if numtries < maxnumtries: 
                # Compute the occupancy grid and the cost
                # Needs states in Nx2 format
                [occ, cost] = Grids(workspace, resolution, epsilon) 
                datasets[k] = [occ, cost]

                # print "cost : "
                # print  cost

                print "nobj : {}, obstacles : {} ".format(
                    nobj, len(workspace.obstacles))

                DrawGrids(occ, cost)
                # raw_input("Press [enter] to continue.")
                k = k+1
                break
        else:
            print('[OBJS] Reached max number of tries. Restarting run...')

    plt.show()

if __name__ == '__main__':

    parser = optparse.OptionParser("usage: %prog [options] arg1 arg2")

    parser.add_option('--numdatasets', 
        default=100, type="int", dest='numdatasets',
        help='Number of datasets to generate')
    parser.add_option('--savefilename', 
        default='2dcostdata.t7', type="string", dest='savefilename',
        help='Filename to save results in (in local directory)')
    parser.add_option('--savematlabfile', 
        default=False, type="int", dest='savematlabfile',
        help='Save results in .mat format')
    parser.add_option('--xsize',
        default=100, type="int", dest='xsize',
        help='Size of the x-dimension (in pixels). X values go from 0-1')
    parser.add_option('--ysize', 
        default=100, type="int", dest='ysize',
        help='Size of the y-dimension (in pixels). Y values go from 0-1')
    parser.add_option('--maxnumobjs', 
        default=4, type="int", dest='maxnumobjs',
        help='Maximum number of obst. per scene (ranges from 1-this number)')
    parser.add_option('--minobjrad',
        default=0.1, type="float", dest='minobjrad',
        help='Minimum radius of any obstacle (in m)')
    parser.add_option('--maxobjrad', 
        default=0.25, type="float", dest='maxobjrad', 
        help='Maximum radius of any obstacle (in m)')
    parser.add_option('--epsilon', 
        default=0.1, type="float", dest='epsilon',
        help='Distance from obstacle at which obstacle cost zeroes out (in m)')
    parser.add_option('--display', 
        default=False, type="int", dest='display',
        help='If set, displays the obstacle costs/occ grids in 2D')
    parser.add_option('--seed', 
        default=-1, type="int", dest='seed',
        help='Random number seed. -ve values mean random seed')

    (options, args) = parser.parse_args()
    # if len(args) != 2:
    #     parser.error("incorrect number of arguments")

    RandomEnvironments(options)
