import common_imports
from workspace import *

# Sample a random point within limits
def samplerandpt(lims):
    pt = np.array(lims:size(1));
    for j in 1,lims:size(1) do
        pt[j] = lims[j][1] + math.random() * (lims[j][2] - lims[j][1]);
    return pt

# Get distance to closest (circular) obstacle & ID of closest obstacle
def mincircobjdist(pt, objs):
    mindist, minid = math.huge, -1
    for k in range(len*(objs)):
    	 # Signed distance 
        dist = math.sqrt((pt - objs[k].c):pow(2):sum()) - objs[k].r;
        if dist < mindist:
          mindist = dist
          minid   = k
        return mindist,minid


# Given a 2D world with obstacles, this returns the cost map and the 
# occupancy grid of that world - the size of the 2D maps is passed in to the 
# function as a torch.LongTensor(2) The cost used is the CHOMP cost function for
# circular obstacles with distance threshold "epsilon"
def computecost(states, objs, epsilon)
    # Get the cost for all the states
    local cost, _, min_dist, min_id = circobstpotential(states, objs, epsilon)

    # States with distance < 0 are considered occupied
    local occ = min_dist:lt(0)
    
    # Return cost and occupancy (also return min dist and min id)
    return cost, occ, min_dist, min_id

# Create a bunch of datasets
maxnobjs = 3
datasets = {}
k = 1;
# Try for this many time to do any one single thing before restarting
maxnumtries = 100; 
while k <= numdatasets:

	# Create obstacles
	workspace=Workspace()

	if k%10 == 0 : print('Dataset: '..k..'/'..numdatasets)
	numtries = 0; # Initialize num tries
	nobj = math.ceil(math.random() * maxnobjs)
	objs = {}
	while size(objs) < nobj and numtries < maxnumtries:
	r = minrad + math.random() * (maxrad - minrad);
	c = samplerandpt(lims);
	# If this object is reasonably far away from other objects
	if (mincircobjdist(c, objs) >= (r + 0.1)): 
	  objs[#objs + 1] = {r = r, c = c};
	numtries = numtries + 1; # Increment num tries

	# Go further only if we have not exceeded all tries
	if numtries < maxnumtries: 
		# Compute the occupancy grid and the cost
		# Needs states in Nx2 format
		cost, occ, mindist, minid = computecost(statesVect, objs, epsilon) 
		# Save dataset
		# dataset = {objs = objs, target = cost:view(size):float(), 
		# 	input = occ:view(size):byte(), mindist = mindist:view(size):float(),
		#                  minid = minid:view(size):byte()}
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

		# Increment counter
		k = k + 1;
	else:
		print('[OBJS] Reached max number of tries. Restarting run...')
