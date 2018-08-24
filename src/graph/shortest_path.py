from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


class CostmapToSparseGraph:
    def __init__(self, costmap):
        self.costmap = costmap


    def costmap_node_id(self, i, j):
        return i + j * self.costmap.shape[0]

    def neiborghs(i, j):
        coord = [None] * 8
        coord[0] = (i, j+1)
        coord[1] = (i, j-1)
        coord[2] = (i+1, j)
        coord[3] = (i+1, j-1)
        coord[4] = (i+1, j+1)
        coord[5] = (i-1, j)
        coord[6] = (i-1, j-1)
        coord[7] = (i-1, j+1)
        return coord


def costmap_to_graph(cost):
    """ Converts a costmap to a compressed sparse graph

        cost : The M x N matrix of costs. cost[i,j] 
               gives the cost of a certain node
        node_map_coord  = (i, j)
        node_graph_id   = i + j * M
    """

    nb_nodes = self.costmap.shape[0] * self.costmap.shape[1]
    graph_dense = np.zeros((nb_nodes, nb_nodes))

    for (i, j), c_ij in np.ndenumerate(self.costmap):


