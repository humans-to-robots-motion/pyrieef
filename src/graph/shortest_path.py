from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


def costmap_node_id(cost, i, j):
    return i + j * cost.shape[0]

def eight_connected_neiborghs(i, j):
    coord 

def costmap_to_graph(cost):
    """ Converts a costmap to a compressed sparse graph

        cost : The M x N matrix of costs. cost[i,j] 
               gives the cost of a certain node
        node_map_coord  = (i, j)
        node_graph_id   = i + j * M
    """

    nb_nodes = cost.shape[0] * cost.shape[1]
    graph_dense = np.zeros((nb_nodes, nb_nodes))

    for (i, j), c_ij in np.ndenumerate(cost):

