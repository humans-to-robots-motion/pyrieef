from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


def costmap_to_graph(cost):
    """ Converts a costmap to a compressed sparse graph 

    """

    nb_nodes = cost.shape[0] * cost.shape[1]
    graph_dense = np.zeros((nb_nodes, nb_nodes))
