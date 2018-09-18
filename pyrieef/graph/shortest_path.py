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

from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph
import numpy as np


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


def shortest_paths(graph_dense):
    graph_sparse = csgraph.csgraph_from_dense(graph_dense)
    # print graph_sparse
    # print graph_sparse.shape
    dist_matrix, predecessors = csgraph.shortest_path(
        graph_sparse,
        directed=False,
        return_predecessors=True)
    # print predecessors
    return predecessors


class CostmapToSparseGraph:
    """Class that convert image to sparse graph representation 
        TODO write a test for and decide weather it should
        be the costmap or the transpose of the costmap
        that should be passed to initialize the class."""

    def __init__(self, costmap, average_cost=False):
        self.costmap = costmap
        self.average_cost = average_cost
        self.init = False
        self._graph_dense = None
        self._edges = []

    def graph_id(self, i, j):
        return i + j * self.costmap.shape[0]

    def costmap_id(self, g_id):
        j = g_id // self.costmap.shape[0]
        i = g_id % self.costmap.shape[0]
        return (i, j)

    def is_in_costmap(self, i, j):
        """ Returns true if the node coord is in the costmap """
        return (
            i >= 0 and i < self.costmap.shape[0] and
            j >= 0 and j < self.costmap.shape[1])

    def graph_edge_cost(self, n1_i, n1_j, n2_i, n2_j):
        """ return the value of an edge in the graph 
            from n1 to n2"""
        n1_id = self.graph_id(n1_i, n1_j)
        n2_id = self.graph_id(n2_i, n2_j)
        return self._graph_dense[n1_id, n2_id]

    def edge_cost(self, c_i, c_j, n_i, n_j):
        cost_c = self.costmap[c_i, c_j]
        cost_n = self.costmap[n_i, n_j]
        if self.average_cost:
            return 0.5 * (cost_c + cost_n)
        return cost_n

    @staticmethod
    def neiborghs(i, j):
        """ returns the costmap coordinates of all neighbor nodes """
        coord = [None] * 8
        coord[0] = (i, j - 1)
        coord[1] = (i, j + 1)
        coord[2] = (i + 1, j)
        coord[3] = (i + 1, j - 1)
        coord[4] = (i + 1, j + 1)
        coord[5] = (i - 1, j)
        coord[6] = (i - 1, j - 1)
        coord[7] = (i - 1, j + 1)
        return coord

    def convert(self):
        """ Converts a costmap to a compressed sparse graph

            cost : The M x N matrix of costs. cost[i,j]
                   gives the cost of a certain node
            node_map_coord  = (i, j)
            node_graph_id   = i + j * M
        """
        nb_nodes = self.costmap.shape[0] * self.costmap.shape[1]
        self._graph_dense = np.zeros((nb_nodes, nb_nodes))
        self._edges = []
        for (c_i, c_j), c_ij in np.ndenumerate(self.costmap):
            c_node = self.graph_id(c_i, c_j)
            for (n_i, n_j) in self.neiborghs(c_i, c_j):
                if self.is_in_costmap(n_i, n_j):
                    # get the neighbor graph id
                    # compute edge cost and store it in graph
                    n_node = self.graph_id(n_i, n_j)
                    self._graph_dense[c_node, n_node] = self.edge_cost(
                        c_i, c_j, n_i, n_j)
                    self._edges.append([c_node, n_node])
        return self._graph_dense

    def update_graph(self, costmap):
        """ updates the graph fast """
        assert costmap.shape == self.costmap.shape
        assert self._graph_dense is not None
        assert self._edges is not []
        self.costmap = costmap
        nb_nodes = self.costmap.shape[0] * self.costmap.shape[1]
        self._graph_dense = np.zeros((nb_nodes, nb_nodes))
        for e in self._edges:
            node_0_i, node_0_j = self.costmap_id(e[0])
            node_1_i, node_1_j = self.costmap_id(e[1])
            self._graph_dense[e[0], e[1]] = self.edge_cost(
                node_0_i, node_0_j,
                node_1_i, node_1_j)

    def shortest_path(self, predecessors, s_i, s_j, t_i, t_j):
        """ Performs a shortest path querry and returns
            the shortes path between some source cell and target cell
            expressed in costmap coordinates

            predecessors : as obdtained by shortest_paths function
        """
        source_id = self.graph_id(s_i, s_j)
        target_id = self.graph_id(t_i, t_j)
        path = []
        path.append((s_i, s_j))
        while True:
            source_id = predecessors[target_id, source_id]
            path.append(self.costmap_id(source_id))
            if source_id == target_id:
                break
        return path

    def breadth_first_search(self, graph_dense, s_i, s_j, t_i, t_j):
        """ Performs a shortest path querry and returns
            the shortes path between some source cell and target cell
            expressed in costmap coordinates. This method is targeted
            for single querry.

            graph_dense : dense graph retpresentation of the costmap
        """
        source_id = self.graph_id(s_i, s_j)
        target_id = self.graph_id(t_i, t_j)
        graph_sparse = csgraph.csgraph_from_dense(graph_dense)
        nodes, predecessors = csgraph.breadth_first_order(
            graph_sparse,
            source_id,
            directed=False,
            return_predecessors=True)
        path = []
        path.append((t_i, t_j))
        while True:
            target_id = predecessors[target_id]
            path.append(self.costmap_id(target_id))
            if source_id == target_id:
                break
        return path

    def dijkstra(self, graph_dense, s_i, s_j, t_i, t_j):
        """
            Performs a graph search for source and target

            graph_dense : dense graph retpresentation of the costmap
            s_i, s_j : source coordinate on the costmap
            t_i, t_j : target coordinate on the costmap
        """
        source_id = self.graph_id(s_i, s_j)
        target_id = self.graph_id(t_i, t_j)
        graph_sparse = csgraph.csgraph_from_dense(graph_dense)
        dist_matrix, predecessors = csgraph.dijkstra(
            graph_sparse,
            directed=not self.average_cost,
            return_predecessors=True,
            indices=source_id,
            limit=np.inf)
        path = []
        path.append((t_i, t_j))
        while True:
            target_id_bkp = target_id
            target_id = predecessors[target_id]
            t_i, t_j = self.costmap_id(target_id)
            s_i, s_j = self.costmap_id(target_id_bkp)
            path.append((t_i, t_j))
            if source_id == target_id:
                break
        return path

    def dijkstra_on_map(self, costmap, s_i, s_j, t_i, t_j):
        """
            Performs a graph search for source and target on costmap
            this is the most efficient implementation for single
            querry graph search on a 2D costmap with scipy

            costmap : matrix of costs, the type of edge cost
                      is given by the class option, either average of
                      node cost or simply node cost which results in a
                      directed graph.
            s_i, s_j : source coordinate on the costmap
            t_i, t_j : target coordinate on the costmap
        """
        self.update_graph(costmap)
        return self.dijkstra(self._graph_dense, s_i, s_j, t_i, t_j)

    def shortest_path_on_map(self, costmap, s_i, s_j, t_i, t_j):
        """
            Performs a graph search for source and target on costmap
            this is the most efficient implementation for single
            querry graph search on a 2D costmap with scipy"""

        self.update_graph(costmap)
        return self.shortest_path(shortest_paths(self._graph_dense),
                                  s_i, s_j, t_i, t_j)
