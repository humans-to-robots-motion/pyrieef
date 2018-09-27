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

import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.pyplot import cm
import sys
import math
from pixel_map import *
from abc import abstractmethod
from differentiable_geometry import *


class Shape:
    """
        This class of Shape represent two dimensional Shapes that can
        be represented as analytical or other type of functions.
        The implementations should return a set of points on the
        perimeter of the Shapes.
    """

    def __init__(self):
        self.nb_points = 50

    @abstractmethod
    def dist_from_border(self, x):
        raise NotImplementedError()

    @abstractmethod
    def dist_gradient(self, x):
        raise NotImplementedError()

    @abstractmethod
    def dist_hessian(self, x):
        raise NotImplementedError()

    @abstractmethod
    def sampled_points(self):
        raise NotImplementedError()


class Circle(Shape):

    def __init__(self, c=np.array([0., 0.]), r=0.2):
        Shape.__init__(self)
        self.origin = c
        self.radius = r

    def dist_from_border(self, x):
        """
            Signed distance

            TODO make this generic (3D) and parallelizable... Tough.
        """
        x_center = np.zeros(x.shape)
        x_center[0] = x[0] - self.origin[0]
        x_center[1] = x[1] - self.origin[1]
        # Oddly the norm of numpy is slower than the standard library here...
        # d = np.linalg.norm(x_center)
        d = np.sqrt(x_center[0]**2 + x_center[1]**2)
        return d - self.radius

    def dist_gradient(self, x):
        x_center = np.zeros(x.shape)
        x_center[0] = x[0] - self.origin[0]
        x_center[1] = x[1] - self.origin[1]
        d = np.sqrt(x_center[0]**2 + x_center[1]**2)
        return x_center / d

    def dist_hessian(self, x):
        x_center = np.zeros(x.shape)
        x_center[0] = x[0] - self.origin[0]
        x_center[1] = x[1] - self.origin[1]
        d_inv = 1. / np.sqrt(x_center[0]**2 + x_center[1]**2)
        return d_inv * np.eye(x.size) - d_inv**3 * np.outer(x_center, x_center)

    def sampled_points(self):
        points = []
        for theta in np.linspace(0, 2 * math.pi, self.nb_points):
            x = self.origin[0] + self.radius * np.cos(theta)
            y = self.origin[1] + self.radius * np.sin(theta)
            points.append(np.array([x, y]))
        return points


class Ellipse(Shape):
    """
        Define a ellipse shape. This is performed using
        a and b parameters. (a, b) are the size of the great
        nd small radii.
    """

    def __init__(self):
        Shape.__init__(self)
        self.origin = np.array([0., 0.])
        self.a = 0.2
        self.b = 0.2

    def dist_from_border(self, x):
        """
            Iterative method described, Signed distance
            http://www.am.ub.edu/~robert/Documents/ellipse.pdf
        """
        x_abs = math.fabs(x[0])
        y_abs = math.fabs(x[1])
        a_m_b = self.a**2 - self.b**2
        phi = 0.
        for i in range(100):
            phi = math.atan2(a_m_b * math.sin(phi) + y_abs * self.b,
                             x_abs * self.a)
            # print "phi : ", phi
            if phi > math.pi / 2:
                break
        return math.sqrt((x_abs - self.a * math.cos(phi))**2 +
                         (y_abs - self.b * math.sin(phi))**2)

    def sampled_points(self):
        points = []
        for theta in np.linspace(0, 2 * math.pi, self.nb_points):
            x = self.origin[0] + self.a * np.cos(theta)
            y = self.origin[1] + self.b * np.sin(theta)
            points.append(np.array([x, y]))
        return points


class Segment(Shape):
    """ A segment defined with an origin, length and orientaiton 
        TODO define distance"""

    def __init__(self,
                 origin=np.array([0., 0.]),
                 orientation=0.,
                 length=0.8):
        Shape.__init__(self)
        self.origin = origin
        self.orientation = orientation
        self.length = length

    def sampled_points(self):
        points = []
        p_0 = 0.5 * self.length * np.array([
            np.cos(self.orientation), np.sin(self.orientation)])
        p_1 = self.origin + p_0
        p_2 = self.origin + -1. * p_0
        for alpha in np.linspace(0., 1., self.nb_points):
            # Linear interpolation
            points.append((1. - alpha) * p_1 + alpha * p_2)
        return points


class Box(Shape):
    """
        An axis aligned box (hypercube) defined by 
            - origin    : its center
            - dim       : its extent

        TODO 1) define distance
             2) class should work for 2D and 3D boxes
             3) test this function
    """

    def __init__(self,
                 origin=np.array([0., 0.]),
                 dim=np.array([1., 1.])):
        self.origin = origin
        self.dim = dim

    def upper_corner(self):
        return self.origin + .5 * self.dim

    def lower_corner(self):
        return self.origin - .5 * self.dim

    def diag(self):
        return np.linalg.norm(self.dim)

    def is_inside(self, x):
        """ works (tested) for arbitrary dimensions, 2d and 3d,
            might not work if called on meshgrid data """
        assert x.size == self.origin.size
        for x_i, corner_i in zip(x, self.lower_corner()):
            if x_i < corner_i:
                return False
        for x_i, corner_i in zip(x, self.upper_corner()):
            if x_i > corner_i:
                return False
        return True

    def vertices(self):
        """ TODO test this function """
        verticies = [self.origin.copy()] * 4
        verticies[0][0] += .5 * self.dim[0]
        verticies[0][1] -= .5 * self.dim[1]
        verticies[1][0] -= .5 * self.dim[0]
        verticies[1][1] += .5 * self.dim[1]
        verticies[2] -= .5 * self.dim
        verticies[3] += .5 * self.dim
        return verticies

    def dist_from_border(self, x):
        distances = [np.linalg.norm(x - vertex) for vertex in self.verticies()]
        return min(distances) if not self.is_inside(x) else -min(distances)


class SignedDistance2DMap(DifferentiableMap):
    """
        This class of wraps the shape class in a differentiable map
    """

    def __init__(self, shape):
        self._shape = shape

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return 2

    def forward(self, x):
        return self._shape.dist_from_border(x)

    def jacobian(self, x):
        return np.matrix(self._shape.dist_gradient(x)).reshape((1, 2))

    def hessian(self, x):
        return np.matrix(self._shape.dist_hessian(x))


class SignedDistanceWorkspaceMap(DifferentiableMap):
    """
        This class of wraps the workspace class in a
        differentiable map
    """

    def __init__(self, workspace):
        self._workspace = workspace

    def output_dimension(self):
        return 1

    def input_dimension(self):
        return 2

    def forward(self, x):
        return self._workspace.min_dist(x)[0]

    def jacobian(self, x):
        """ Warning: this gradient is ill defined
            it has a kink when two objects are at the same distance """
        return np.matrix(self._workspace.min_dist_gradient(x)).reshape((1, 2))

    def hessian(self, x):
        """ Warning: this hessian is ill defined
            it has a kink when two objects are at the same distance """
        [mindist, minid] = self._workspace.min_dist(x)
        return np.matrix(self._workspace.obstacles[minid].dist_hessian(x))

    def evaluate(self, x):
        """ Warning: this gradient is ill defined
            it has a kink when two objects are at the same distance """
        [mindist, minid] = self._workspace.min_dist(x)
        g_mindist = self._workspace.obstacles[minid].dist_gradient(x)
        J_mindist = np.matrix(g_mindist).reshape((1, 2))
        return [mindist, J_mindist]


def occupancy_map(nb_points, workspace):
    """ Returns an occupancy map in the form of a square matrix
        using the signed distance field associated to a workspace object """
    meshgrid = workspace.box.stacked_meshgrid(nb_points)
    sdf = SignedDistanceWorkspaceMap(workspace)(meshgrid).transpose()
    return (sdf < 0).astype(float)


class EnvBox(Box):
    """ Specializes a box to defined an environment """

    def __init__(self,
                 origin=np.array([0., 0.]),
                 dim=np.array([1., 1.])):
        Box.__init__(self, origin, dim)

    def box_extend(self):
        return np.array([self.origin[0] - self.dim[0] / 2.,
                         self.origin[0] + self.dim[0] / 2.,
                         self.origin[1] - self.dim[1] / 2.,
                         self.origin[1] + self.dim[1] / 2.,
                         ])

    def extent(self):
        box_extend = self.box_extend()
        extent = Extent()
        extent.x_min = box_extend[0]
        extent.x_max = box_extend[1]
        extent.y_min = box_extend[2]
        extent.y_max = box_extend[3]
        return extent

    def meshgrid(self, nb_points=100):
        """ This mesh grid definition matches the one in the PixelMap class
            Note that it is only defined for squared boxes.
            simply takes as input the number of points which corresponds
            to the number of cells for the PixelMap"""
        assert self.dim[0] == self.dim[1]
        resolution = self.dim[0] / nb_points
        extent = self.extent()
        x_min = extent.x_min + 0.5 * resolution
        x_max = extent.x_max - 0.5 * resolution
        y_min = extent.y_min + 0.5 * resolution
        y_max = extent.y_max - 0.5 * resolution
        x = np.linspace(x_min, x_max, nb_points)
        y = np.linspace(y_min, y_max, nb_points)
        return np.meshgrid(x, y)

    def stacked_meshgrid(self, nb_points=100):
        X, Y = self.meshgrid(nb_points)
        return np.stack([X, Y])

    def sample_uniform(self):
        """ Sample uniformly point in extend"""
        p = np.random.random(self.origin.size)  # p in [0, 1]^n
        return np.multiply(self.dim, p) + self.lower_corner()

    def __str__(self):
        return "origin : {}, dim : {}".format(self.origin, self.dim)


def box_from_limits(x_min, x_max, y_min, y_max):
    assert x_max > x_min
    assert y_max > y_min
    return EnvBox(
        origin=np.array([(x_min + x_max) / 2., (y_min + y_max) / 2.]),
        dim=np.array([x_max - x_min, y_max - y_min]))


def pixelmap_from_box(nb_points, box):
    extent = box.extent()
    assert extent.x() == extent.y()  # Test is square
    resolution = extent.x() / nb_points
    return PixelMap(resolution, extent)


class Workspace:
    """
       Contains obstacles.
    """

    def __init__(self, box=EnvBox()):
        self.box = box
        self.obstacles = []

    def in_collision(self, pt):
        for obst in self.obstacles:
            if obst.dist_from_border(pt) < 0.:
                return True
        return False

    def min_dist(self, pt):
        if pt.shape == (2,):
            d_m = float("inf")
            i_m = -1
        else:
            """ Here we declare special variable for element wise querry """
            shape = (pt.shape[1], pt.shape[2])
            d_m = np.full(shape, np.inf)
            i_m = np.full(shape, -1)

        for i, obst in enumerate(self.obstacles):
            d = obst.dist_from_border(pt)
            closer_to_i = d < d_m
            d_m = np.where(closer_to_i, d, d_m)
            i_m = np.where(closer_to_i, i, i_m)
        return [d_m, i_m]

    def min_dist_gradient(self, pt):
        """ Warning: this gradient is ill defined
            it has a kink when two objects are at the same distance """
        [d_m, i_m] = self.min_dist(pt)
        return self.obstacles[i_m].dist_gradient(pt)

    def add_circle(self, origin=None, radius=None):
        if origin is None and radius is None:
            self.obstacles.append(Circle())
        else:
            self.obstacles.append(Circle(origin, radius))

    def add_segment(self, origin=None, length=None):
        if origin is None and length is None:
            self.obstacles.append(Segment())
        else:
            self.obstacles.append(Segment(origin, length))

    def all_points(self):
        points = []
        for o in self.obstacles:
            points += o.sampled_points()
        return points

    def pixel_map(self, nb_points):
        extent = self.box.extent()
        assert extent.x() == extent.y()
        resolution = extent.x() / nb_points
        return PixelMap(resolution, extent)


def sample_circles(nb_circles):
    """ Samples circles in [0, 1]^2 with radii in [0, 1]"""
    centers = np.random.rand(nb_circles, 2)
    radii = np.random.rand(nb_circles)
    return zip(centers, radii)


def sample_workspace(nb_circles, radius_parameter=.20):
    """ Samples a workspace randomly composed of nb_circles 
        the radius parameter specifies the 
        max fraction of workspace diagonal used for a circle radius. """
    workspace = Workspace()
    extent = workspace.box.extent()
    max_radius = radius_parameter * workspace.box.diag()
    workspace.obstacles = [None] * nb_circles
    for i in range(nb_circles):
        center = workspace.box.sample_uniform()
        radius = max_radius * np.random.rand()
        workspace.obstacles[i] = Circle(center, radius)
    return workspace


def sample_collision_free(workspace):
    """ Samples a collision free point """
    min_dist = workspace.box.diag() / 20.
    while True:
        p = workspace.box.sample_uniform()
        dist = workspace.min_dist(p)[0]
        if dist > min_dist:
            return p
