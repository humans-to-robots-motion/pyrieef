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
# from __future__ import absolute_import
# from .__init__ import *
from geometry.workspace import *
from itertools import product
from numpy.testing import assert_allclose


def test_circle():
    circle = Circle()
    sdf = SignedDistance2DMap(circle)

    print("Check Circle SDF (J implementation) : ")
    assert check_jacobian_against_finite_difference(sdf)

    print("Check Circle SDF (H implementation) : ")
    assert check_hessian_against_finite_difference(sdf)

    x = np.random.random(2)

    # Test gradient outside circle
    print("test circle 1")
    circle = Circle(radius=.2)

    g1 = Shape.dist_gradient(circle, x)
    g2 = circle.dist_gradient(x)
    assert_allclose(g1, g2)

    # Test gradient inside circle
    print("test circle 2")
    circle = Circle(radius=2.)
    g1 = Shape.dist_gradient(circle, x)
    g2 = circle.dist_gradient(x)
    assert_allclose(g1, g2)


def test_segment():
    p1 = np.random.random(2)
    p2 = np.random.random(2)
    line = Segment(p1=p1, p2=p2)
    p1_l, p2_l = line.end_points()
    assert_allclose(p1, p1_l)
    assert_allclose(p2, p2_l)

    segments = []
    segments.append(Segment(origin=np.array(
        [-3., 0]), length=1., orientation=0.))
    segments.append(Segment(origin=np.array(
        [-3., 0]), length=1., orientation=1.57))
    segments.append(Segment(origin=np.array(
        [-3., 0]), length=1., orientation=3.14))
    for segment in segments:
        sdf = SignedDistance2DMap(segment)
        print("Check Segment SDF (J implementation) : ")
        assert check_jacobian_against_finite_difference(sdf)
        print("Check Segment SDF (H implementation) : ")
        assert check_hessian_against_finite_difference(sdf)


def test_box():

    box = Box()

    verticies = box.verticies()
    for k, vertex in enumerate(verticies):
        print("vertex {} : {}".format(k, vertex))

    dist = box.dist_from_border(np.array([0.0, 1.0]))
    print("dist = ", dist)
    assert np.fabs(dist - 0.5) < 1.e-06

    dist = box.dist_from_border(np.array([1.0, 0.0]))
    print("dist = ", dist)
    assert np.fabs(dist - 0.5) < 1.e-06

    boxes = []
    boxes.append(Box(origin=np.array([.5, .5]), dim=np.array([1., 1.])))
    boxes.append(Box(origin=np.array([-.5, .5]), dim=np.array([.5, .5])))
    for box in boxes:
        sdf = SignedDistance2DMap(box)
        print("Check Box SDF (J implementation) : ")
        assert check_jacobian_against_finite_difference(sdf)
        print("Check Box SDF (H implementation) : ")
        assert check_hessian_against_finite_difference(sdf)


def test_inside_box():
    for n in [2, 3]:  # TODO make it work for 3D
        box = EnvBox(
            origin=np.random.rand(n),
            dim=np.random.rand(n) + .5 * np.ones(n))
        for i in range(50):
            p = box.sample_uniform()
            assert box.is_inside(p)
            p = np.random.random(box.origin.size)
            assert not box.is_inside(p + box.upper_corner())
            assert not box.is_inside(-1. * p + box.lower_corner())


def test_axis_aligned_box():
    environment = EnvBox()
    points = environment.meshgrid_points(13)
    dimensions = np.array([.5, .5])

    box1 = AxisAlignedBox(dim=dimensions)
    box2 = Box(dim=dimensions)

    f1 = SignedDistance2DMap(box2)
    for _ in range(100):
        p = environment.sample_uniform()
        J = f1.jacobian(p)
        J_diff = finite_difference_jacobian(f1, p)
        assert check_is_close(J, J_diff, 1e-4)

    f2 = SignedDistance2DMap(box1)
    for _ in range(100):
        p = environment.sample_uniform()

        J1 = f1.jacobian(p)
        J2 = f2.jacobian(p)
        assert check_is_close(J1, J2, 1e-4)

        J = f2.jacobian(p)
        J_diff = finite_difference_jacobian(f2, p)
        assert check_is_close(J, J_diff, 1e-4)

        H1 = box1.dist_hessian(p)
        H2 = box2.dist_hessian(p)
        assert_allclose(H1, H2)

    for p in points:
        sdf1 = box1.dist_from_border(p)
        sdf2 = box2.dist_from_border(p)
        assert np.fabs(sdf1 - sdf2) < 1.e-06

    # TODO make code parallelizable
    # grid = EnvBox().stacked_meshgrid()
    # sdf1 = box1.dist_from_border(grid)
    # sdf2 = box2.dist_from_border(grid)
    # assert_allclose(sdf1, sdf2)


def test_ellipse():

    ellipse = Ellipse(a=0.2, b=0.1)

    dist = ellipse.dist_from_border(np.array([0.3, 0.0]))
    print("dist = ", dist)
    assert np.fabs(dist - 0.1) < 1.e-06

    dist = ellipse.dist_from_border(np.array([0.0, 0.3]))
    print("dist = ", dist)
    assert np.fabs(dist - 0.2) < 1.e-06


def test_line_side():
    assert line_side([1, 0], [0, 0], [-1, -1])
    assert line_side([25, 0], [0, 0], [-1, -14])
    assert line_side([25, 20], [0, 20], [-1, 6])
    assert line_side([24, 20], [-1, 20], [-2, 6])

    assert not line_side([1, 0], [0, 0], [1, 1])
    assert not line_side([12, 0], [0, 0], [2, 1])
    assert not line_side([-25, 0], [0, 0], [-1, -14])
    assert not line_side([1, 0.5], [0, 0], [1, 1])

    assert not line_side([0, 5], [1, 10], [10, 20])
    assert line_side([0, 9.1], [1, 10], [10, 20])
    assert not line_side([0, 5], [1, 10], [20, 10])
    assert not line_side([0, 9.1], [1, 10], [20, 10])

    assert line_side([1, 1], [1, 10], [0, 0])
    assert not line_side([1, 10], [1, 1], [0, 0])
    assert not line_side([1, 1], [1, 10], [5, 0])
    assert line_side([1, 10], [1, 1], [5, 0])

    assert line_side([1, -1], [10, -1], [0, 0])
    assert not line_side([10, -1], [1, -1], [0, 0])
    assert not line_side([1, -1], [10, -1], [0, -9])
    assert line_side([10, -1], [1, -1], [0, -9])

    assert line_side([0, 0], [10, 10], [1, 2])
    assert not line_side([10, 10], [0, 0], [1, 2])
    assert not line_side([0, 0], [10, 10], [1, 0])
    assert line_side([10, 10], [0, 0], [1, 0])

    assert not line_side([0, 0], [-10, 10], [1, 2])
    assert not line_side([0, 0], [-10, 10], [1, 2])
    assert line_side([0, 0], [-10, 10], [-1, -2])
    assert not line_side([-10, 10], [0, 0], [-1, -2])


def test_polygon():
    box = Polygon()
    assert not box.is_inside(np.array([2., 0.]))
    assert not box.is_inside(np.array([0., 2.]))
    assert not box.is_inside(np.array([-2., 0.]))
    assert not box.is_inside(np.array([-0., -2.]))
    assert box.is_inside(np.array([.5, .1]))
    assert box.is_inside(np.array([.5, .5]))
    assert box.is_inside(np.array([.1, .1]))
    assert box.is_inside(np.array([.1, .5]))


def test_hexagon():
    h = hexagon(scale=.5)
    assert not h.is_inside(np.array([2., 0.]))
    assert not h.is_inside(np.array([0., 2.]))
    assert not h.is_inside(np.array([-2., 0.]))
    assert not h.is_inside(np.array([-0., -2.]))
    assert h.is_inside(np.array([.1, 0.]))
    assert h.is_inside(np.array([0., .1]))
    assert h.is_inside(np.array([-.1, 0.]))
    assert h.is_inside(np.array([0., -.1]))


def test_hexagon_jac():
    environment = EnvBox()
    polygon = hexagon(scale=.5)
    f = SignedDistance2DMap(polygon)
    for _ in range(100):
        p = environment.sample_uniform()
        J = f.jacobian(p)
        J_diff = finite_difference_jacobian(f, p)
        assert check_is_close(J, J_diff, 1e-4)
        H = f.hessian(p)
        H_diff = finite_difference_hessian(f, p)
        assert check_is_close(H, H_diff, 1e-4)


def test_sdf_derivatives():
    verbose = False
    circles = []
    for center, radius in sample_circles(nb_circles=10):
        circles.append(Circle(center, radius))
    for c in circles:
        signed_distance_field = SignedDistance2DMap(c)
        assert check_jacobian_against_finite_difference(
            signed_distance_field, verbose)
        assert check_hessian_against_finite_difference(
            signed_distance_field)


def test_sdf_workspace():
    workspace = sample_circle_workspaces(nb_circles=10)
    signed_distance_field = SignedDistanceWorkspaceMap(workspace)
    assert check_jacobian_against_finite_difference(signed_distance_field)
    assert check_hessian_against_finite_difference(signed_distance_field)


def test_meshgrid():
    nb_points = 10
    workspace = Workspace()
    pixel_map = workspace.pixel_map(nb_points)
    X, Y = workspace.box.meshgrid(nb_points)
    print("pm -- resolution : {}".format(pixel_map.resolution))
    print("pm -- origin : {}".format(pixel_map.origin))
    for i, j in product(list(range(nb_points)), list(range(nb_points))):
        p_meshgrid = np.array([X[i, j], Y[i, j]])
        p_grid = pixel_map.world_to_grid(p_meshgrid)
        p_world = pixel_map.grid_to_world(p_grid)
        assert_allclose(p_meshgrid, p_world)


def test_pixelmap():
    nb_points = 10
    workspace = Workspace()
    pixel_map = workspace.pixel_map(nb_points)
    X, Y = workspace.box.meshgrid(nb_points)
    print("pm -- resolution : {}".format(pixel_map.resolution))
    print("pm -- origin : {}".format(pixel_map.origin))
    for i, j in product(list(range(nb_points)), list(range(nb_points))):
        p_meshgrid = np.array([X[i, j], Y[i, j]])
        p_grid = pixel_map.world_to_grid(p_meshgrid)
        p_world = pixel_map.grid_to_world(p_grid)
        assert_allclose(p_meshgrid, p_world)


def test_sdf_grid():
    nb_points = 24
    workspace = sample_circle_workspaces(nb_circles=10)
    sdf = SignedDistanceWorkspaceMap(workspace)
    pixel_map = workspace.pixel_map(nb_points)
    # WARNING !!!
    # Here we need to transpose the costmap
    # otherwise the grid representation do not match
    grid = workspace.box.stacked_meshgrid(nb_points)
    sdfmap = sdf(grid).T
    for i, j in product(list(range(nb_points)), list(range(nb_points))):
        p = pixel_map.grid_to_world(np.array([i, j]))
        assert_allclose(sdf(p), sdfmap[i, j])


def test_workspace_to_occupancy_map():
    np.random.seed(0)
    np.set_printoptions(suppress=True, linewidth=200, precision=2)
    nb_points = 10
    workspace = sample_circle_workspaces(nb_circles=5)
    occ = occupancy_map(nb_points, workspace)
    pixel_map = pixelmap_from_box(nb_points, workspace.box)
    for i, j in product(list(range(nb_points)), list(range(nb_points))):
        p = pixel_map.grid_to_world(np.array([i, j]))
        v = float(workspace.min_dist(p)[0] < 0)
        assert_allclose(occ[i, j], v)


def test_signed_disance_field_function():
    workspace = sample_circle_workspaces(nb_circles=3)
    sdf = workspace.signed_distance_field()
    points = np.random.random((10, 2))
    for p in points:
        assert_allclose(sdf(p), workspace.min_dist(p)[0])


if __name__ == "__main__":

    # test_circle()
    # test_segment()
    # test_box()
    # test_line_side()
    # test_polygon()
    # test_axis_aligned_box()
    # test_inside_box()
    # test_ellipse()
    # test_polygon()
    # test_hexagon()
    # test_sdf_derivatives()
    # test_sdf_workspace()
    # test_meshgrid()
    # test_sdf_grid()
    # test_workspace_to_occupancy_map()
    test_signed_disance_field_function()
