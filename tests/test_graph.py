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

from __init__ import *
from graph.shortest_path import *
from geometry.workspace import *
from motion.cost_terms import *
from utils import timer
from numpy.testing import assert_allclose


def test_symetrize():
    A_res = np.array([[0, 2, 1],
                      [2, 0, 0],
                      [1, 0, 0]])
    A = np.zeros((3, 3))
    A[0, 1] = 2
    A[0, 2] = 1
    A = symmetrize(A)
    print("A : \n", A)
    print("A_res : \n", A_res)
    assert_allclose(A, A_res, 1e-8)
    assert check_symmetric(A, A_res)


def test_coordinates():
    costmap = np.random.random((100, 100))
    converter = CostmapToSparseGraph(costmap, average_cost=True)
    graph_ids = 100 * 100 * np.random.random(100)
    for g_id in graph_ids.astype(int):
        c_id = converter.costmap_id(g_id)
        assert converter.graph_id(c_id[0], c_id[1]) == g_id


def test_graph_edge_cost():
    nb_points = 10
    costmap = np.random.random((nb_points, nb_points))
    converter = CostmapToSparseGraph(costmap, average_cost=False)
    graph = converter.convert()
    for (n1_i, n1_j), c_ij in np.ndenumerate(costmap):
        for (n2_i, n2_j) in converter.neiborghs(n1_i, n1_j):
            if converter.is_in_costmap(n2_i, n2_j):
                c1 = converter.edge_cost(n1_i, n1_j, n2_i, n2_j)
                c2 = converter.graph_edge_cost(n1_i, n1_j, n2_i, n2_j)
                assert c2 == c2
                assert c2 == costmap[n2_i, n2_j]


def test_costmap_to_graph_symmetry():
    costmap = np.random.random((5, 5))
    converter = CostmapToSparseGraph(costmap, average_cost=True)
    graph = converter.convert()
    np.set_printoptions(suppress=True, linewidth=200, precision=0)
    print(costmap)
    print(graph.shape)
    print(graph)
    assert check_symmetric(graph)


def test_workspace_to_graph():
    workspace = Workspace()
    radius = .1
    nb_points = 24
    workspace.obstacles.append(Circle(np.array([0.1, 0.1]), radius))
    workspace.obstacles.append(Circle(np.array([-.1, 0.1]), radius))
    pixel_map = workspace.pixel_map(nb_points)
    phi = SimplePotential2D(SignedDistanceWorkspaceMap(workspace))
    # WARNING !!!
    # Here we need to transpose the costmap
    # otherwise the grid representation do not match
    costmap = phi(workspace.box.stacked_meshgrid(nb_points)).T
    converter = CostmapToSparseGraph(costmap, average_cost=False)
    graph = converter.convert()
    for (n1_i, n1_j), c_ij in np.ndenumerate(costmap):
        for (n2_i, n2_j) in converter.neiborghs(n1_i, n1_j):
            if converter.is_in_costmap(n2_i, n2_j):
                c1 = converter.edge_cost(n1_i, n1_j, n2_i, n2_j)
                c2 = converter.graph_edge_cost(n1_i, n1_j, n2_i, n2_j)
                p = pixel_map.grid_to_world(np.array([n2_i, n2_j]))
                c3 = phi(p)
                assert c2 == c2
                assert c2 == costmap[n2_i, n2_j]
                assert_allclose(c2, c3)


def test_workspace_to_shortest_path():
    workspace = Workspace()
    radius = .1
    workspace.obstacles.append(Circle(np.array([0.1, 0.1]), radius))
    workspace.obstacles.append(Circle(np.array([-.1, 0.1]), radius))
    phi = SimplePotential2D(SignedDistanceWorkspaceMap(workspace))
    costmap = phi(workspace.box.stacked_meshgrid(24))
    print(costmap)
    converter = CostmapToSparseGraph(costmap, average_cost=True)
    graph = converter.convert()
    assert check_symmetric(graph)
    s_i = 3
    s_j = 3
    t_i = 21
    t_j = 21
    path = converter.shortest_path(shortest_paths(graph), s_i, s_j, t_i, t_j)
    assert len(path) > 0


def test_breadth_first_search():
    workspace = Workspace()
    radius = .1
    workspace.obstacles.append(Circle(np.array([0.1, 0.1]), radius))
    workspace.obstacles.append(Circle(np.array([-.1, 0.1]), radius))
    phi = SimplePotential2D(SignedDistanceWorkspaceMap(workspace))
    costmap = phi(workspace.box.stacked_meshgrid(24))
    converter = CostmapToSparseGraph(costmap, average_cost=True)
    graph = converter.convert()
    assert check_symmetric(graph)
    s_i = 3
    s_j = 3
    t_i = 21
    t_j = 21
    path = converter.breadth_first_search(graph, s_i, s_j, t_i, t_j)
    assert len(path) > 0

if __name__ == "__main__":
    test_symetrize()
    test_coordinates()
    test_graph_edge_cost()
    test_costmap_to_graph_symmetry()
    test_workspace_to_graph()
    test_workspace_to_shortest_path()
    test_breadth_first_search()
