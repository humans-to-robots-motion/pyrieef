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
#                                         Jim Mainprice on Sunday June 17 2017

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
        """
        x_center = np.zeros(x.shape)
        x_center[0] = x[0] - self.origin[0]
        x_center[1] = x[1] - self.origin[1]
        # Oddly the norm of numpy is slower than the standard library here...
        # d = np.linalg.norm(x_center)
        d = np.sqrt(x_center[0]**2 + x_center[1]**2)
        # print "d1 : {}, d : {}".format(d1, d)
        # print "d : ", d
        return d - self.radius

    def dist_gradient(self, x):
        x_center = x - self.origin
        # d = np.linalg.norm(x_center)
        d = np.sqrt(x_center[0]**2 + x_center[1]**2)
        return x_center / d

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


class Box:
    """
        A box is defined by an origin, which is its center.
        and dimension which are it's size in axis aligned coordinates
    """

    def __init__(self,
                 origin=np.array([0., 0.]),
                 dim=np.array([1., 1.])):
        self.origin = origin
        self.dim = dim

    def box_extends(self):
        return np.array([self.origin[0] - self.dim[0] / 2.,
                         self.origin[0] + self.dim[0] / 2.,
                         self.origin[1] - self.dim[1] / 2.,
                         self.origin[1] + self.dim[1] / 2.,
                         ])

    def extends(self):
        box_extends = self.box_extends()
        extends = Extends()
        extends.x_min = box_extends[0]
        extends.x_max = box_extends[1]
        extends.y_min = box_extends[2]
        extends.y_max = box_extends[3]
        return extends

    def mesh_grid(self, nb_points=100):
        extends = self.extends()
        x = np.linspace(extends.x_min, extends.x_max, nb_points)
        y = np.linspace(extends.y_min, extends.y_max, nb_points)
        return np.meshgrid(x, y)


class SignedDistance2DMap(DifferentiableMap):
    """ 
        This class of wraps the shape class in a 
        differentiable map
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

    def evaluate(self, x):
        """ Warning: this gradient is ill defined
            it has a kink when two objects are at the same distance """
        [mindist, minid] = self._workspace.min_dist(x)
        g_mindist = self._workspace.obstacles[minid].dist_gradient(x)
        J_mindist = np.matrix(g_mindist).reshape((1, 2))
        return [mindist, J_mindist]


class Workspace:
    """
        Contains obstacles.
    """

    def __init__(self, box=Box()):
        self.box = box
        self.obstacles = []

    def in_collision(self, pt):
        for obst in self.obstacles:
            if obst.dist_from_border(pt) < 0.:
                return True
        return False

    def min_dist(self, pt):
        mindist = float("inf")
        minid = -1
        for k, obst in enumerate(self.obstacles):
            dist = obst.dist_from_border(pt)  # Signed distance
            # print "dist.shape : ", dist.shape
            if dist < mindist:
                (mindist, minid) = (dist, k)
        return [mindist, minid]

    def min_dist_gradient(self, pt):
        """ Warning: this gradient is ill defined
            it has a kink when two objects are at the same distance """
        [mindist, minid] = self.min_dist(pt)
        g_mindist = self.obstacles[minid].dist_gradient(pt)
        return g_mindist

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


# run the server
if __name__ == "__main__":
    obstacle = Circle()
    print obstacle.sampled_points()
