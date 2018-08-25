#!/usr/bin/env python

# Copyright (c) 2015 Max Planck Institute
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
# Jim Mainprice on Sunday June 17 2017

from test_common_imports import *
from graph.shortest_path import *
import numpy as np
from numpy.testing import assert_allclose
from geometry.workspace import *
from motion.cost_terms import *
import rendering.workspace_renderer as render
from utils import timer

def test_symetrize():
    A_res = np.array([[0, 2, 1],
                      [2, 0, 0],
                      [1, 0, 0]])
    A = np.zeros((3, 3))
    A[0, 1] = 2
    A[0, 2] = 1
    A = symmetrize(A)
    print "A : \n", A
    print "A_res : \n", A_res
    assert_allclose(A, A_res, 1e-8)
    assert check_symmetric(A, A_res)


def test_costmap_to_graph():
    costmap = np.random.random((5, 5))
    converter = CostmapToSparseGraph(costmap, average_cost=True)
    graph = converter.convert()
    np.set_printoptions(suppress=True, linewidth=200, precision=0)
    print costmap
    print graph.shape
    print graph
    assert check_symmetric(graph)


def test_workspace_to_graph():
    workspace = Workspace()
    radius = .1
    workspace.obstacles.append(Circle(np.array([0.1, 0.1]), radius))
    workspace.obstacles.append(Circle(np.array([-.1, 0.1]), radius))
    phi = SimplePotential2D(SignedDistanceWorkspaceMap(workspace))
    costmap = phi(workspace.box.stacked_meshgrid(24))
    print costmap
    converter = CostmapToSparseGraph(costmap, average_cost=True)
    graph = converter.convert()
    assert check_symmetric(graph)
    # viewer = render.WorkspaceRender(workspace)
    # viewer.draw_ws_background(phi)
    # viewer.draw_ws_obstacles()
    # rate = timer.Rate(25)
    # while True:
    #     viewer.render()
    #     rate.sleep()


if __name__ == "__main__":
    test_symetrize()
    test_costmap_to_graph()
    test_workspace_to_graph()
