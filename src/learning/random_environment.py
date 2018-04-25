import common_imports
from workspace import *
from math import *
from math.random import *
import optparse

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
    occupancy = np.zeros((grid.nb_cells_x, grid.nb_cells_y))
    for i in range(grid.nb_cells_x):
        for j in range(grid.nb_cells_y):
            pt = grid.grid_to_world(np.array([i, j]))
            min_dist = workspace.MinDist(pt)
            occupancy[i, j] = min_dist =< 0.
            costs[i, j] = ChompObstacleCost(min_dist, epsilon)
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

    # Create a bunch of datasets
    maxnobjs = 3
    datasets = {}
    k = 1
    
    # Try for this many time to do any one single thing before restarting
    maxnumtries = 100
    for k in range(numdatasets):

        # Create obstacles
        workspace=Workspace()

        if k%10 == 0 : print('Dataset: ' + k + '/' + numdatasets)
        numtries = 0; # Initialize num tries
        nobj = ceil(random() * maxnobjs)
        objs = []
        while len(objs) < nobj and numtries < maxnumtries:
            r = minrad + random() * (maxrad - minrad)
            c = samplerandpt(lims)
            # If this object is reasonably far away from other objects
            if (mincircobjdist(c, objs) >= (r + 0.1)):
                workspace.AddCircle(c, r)
            numtries = numtries + 1  # Increment num tries

            # Go further only if we have not exceeded all tries
            if numtries < maxnumtries: 
                # Compute the occupancy grid and the cost
                # Needs states in Nx2 format
                [occ, cost] = Grids(workspace, resolution, epsilon) 
                # Save dataset
                # dataset = {objs = objs, target = cost:view(size):float(), 
                # input = occ:view(size):byte(), mindist = mindist:view(size):float(),
                # minid = minid:view(size):byte()}
                datasets[k] = dataset

                # Display the cost, occ grid, min dist, min id
                # if opt.display and (k%10 == 0) then
                #   local catimgs = torch.cat({cost:view(1,size[1], size[2]):float(), 
                #   	occ:view(1,size[1], size[2]):float(), 
                #   	mindist:view(1,size[1], size[2]):float(), 
                #   	minid:view(1,size[1], size[2]):float()}, 1);
                #   local temp = image.toDisplayTensor{input=catimgs, 
                #   	padding=padding, nrow=4, scaleeach = true} 
                #   image.display{image = temp, win = qtwindow, x = 0, y = 20}
        else:
            print('[OBJS] Reached max number of tries. Restarting run...')


if __name__ == '__main__':

    parser = optparse.OptionParser("usage: %prog [options] arg1 arg2")

    parser.add_option('-numdatasets', 
        default=100, type="int", dest='numdatasets',
        help='Number of datasets to generate')
    parser.add_option('-savefilename', 
        default='2dcostdata.t7', type="string", dest='savefilename',
        help='Filename to save results in (in local directory)')
    parser.add_option('-savematlabfile', 
        default=false, type="bool", dest='savematlabfile',
        help='Save results in .mat format')
    parser.add_option('-xsize',
        default=100, type="int", dest='xsize',
        help='Size of the x-dimension (in pixels). X values go from 0-1')
    parser.add_option('-ysize', 
        default=100, type="int", dest='ysize',
        help='Size of the y-dimension (in pixels). Y values go from 0-1')
    parser.add_option('-maxnumobjs', 
        default=4, type="int", dest='maxnumobjs',
        help='Maximum number of obst. per scene (ranges from 1-this number)')
    parser.add_option('-minobjrad', 
        default=0.1, type="float", dest='minobjrad',
        help='Minimum radius of any obstacle (in m)')
    parser.add_option('-maxobjrad', 
        default=0.25, type="float", dest='maxobjrad', 
        help='Maximum radius of any obstacle (in m)')
    parser.add_option('-epsilon', 
        default=0.1, type="float", dest='epsilon',
        help='Distance from obstacle at which obstacle cost zeroes out (in m)')
    parser.add_option('-display', 
        default=false, type="bool", dest='display',
        help='If set, displays the obstacle costs/occ grids in 2D')
    parser.add_option('-seed', 
        default=-1, type="int", dest='seed',
        help='Random number seed. -ve values mean random seed')

    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("incorrect number of arguments")

    RandomEnvironments(options)
