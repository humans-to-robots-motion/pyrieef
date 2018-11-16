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
import math
from .pixel_map import *
from abc import abstractmethod
from .differentiable_geometry import *


def vector_norm(x):
    return np.linalg.norm(x, axis=0)
    # return np.sqrt(x[0] ** 2 + x[1] ** 2)


class Shape:
    """
    Shape represents workspace objects in 2D or 3D.

    The contour, distance and gradient and hessian of the distance
    function are represented as analytical or other type of functions.
    The implementations should return a set of points on the
    contour, to allow easy drawing.
    """

    def __init__(self):
        self.nb_points = 50

    @abstractmethod
    def closest_point(self, x):
        """
        Returns the closest point from x on the contour.

        Parameters
        ----------
            x : numpy array
        """
        raise NotImplementedError()

    @abstractmethod
    def dist_from_border(self, x):
        """
        Returns the sign distance at x.

        Parameters
        ----------
            x : numpy array
        """
        raise NotImplementedError()

    def is_inside(self, x):
        """
        Returns true if x is inside the shape
        """
        return False

    def dist_gradient(self, x):
        """
        Returns the gradient of the distance function.

        Note that for the first order, the signed distance function
        of any shape is fully expressed by the closet point to the border.
        The gradient is simply the normalized vecor pointing
        towards or away from that point depending on whether
        we are outside or inside of the shape.
        This does not hold for higher order quatities
        (i.e. the Hessian of the potential field).

        Warning: not parraleized but should work from 3D
        This works for shapes with no volumes such as segments

        Parameters
        ----------
            x : numpy array
        """
        x_center = x - self.closest_point(x)
        sign = -1. if self.is_inside(x) else 1.
        return sign * x_center / vector_norm(x_center)

    def dist_hessian(self, x):
        """
        Returns the hessian of the distance function.

        Parameters
        ----------
            x : numpy array
        """
        raise NotImplementedError()

    @abstractmethod
    def sampled_points(self):
        raise NotImplementedError()


def point_distance_gradient(x, origin):
    """
    Returns the gradient of the distance function to a point

    Parameters
    ----------
        x : numpy array
        origin : numpy array
    """
    x_center = (x.T - origin).T
    return x_center / vector_norm(x_center)


def point_distance_hessian(x, origin):
    """
    Returns the hessian of the distance function to a point

    Note: that it balances two sumands
        1) euclidean metric : identity
        2)  pullback metric : the outer product of gradient

    Parameters
    ----------
        x : numpy array
        origin : numpy array
    """
    x_center = (x.T - origin).T
    d_inv = 1. / vector_norm(x_center)
    return d_inv * np.eye(x.size) - d_inv**3 * np.outer(x_center, x_center)


class Circle(Shape):

    def __init__(self, origin=np.array([0., 0.]), radius=0.2):
        Shape.__init__(self)
        self.origin = origin
        self.radius = radius

    def dist_from_border(self, x):
        """
        Returns the signed distance (SD) to a circle shape;
        negative inside and positive outside.

        This function is paralleized, meanning that if called on
        an array (2 or 3, n, n) it will return a matrix of SDField.

        Parameters
        ----------
            x : numpy array
        """
        return vector_norm((x.T - self.origin).T) - self.radius

    def is_inside(self, x):
        return vector_norm((x.T - self.origin).T) < self.radius

    def closest_point(self, x):
        x_center = (x.T - self.origin).T
        p_in_circle = self.radius * x_center / vector_norm(x_center)
        return p_in_circle + self.origin

    def dist_gradient(self, x):
        """ Warning: not parraleized but should work from 3D """
        return point_distance_gradient(x, self.origin)

    def dist_hessian(self, x):
        """ Warning: not parraleized but should work from 3D """
        return point_distance_hessian(x, self.origin)

    def sampled_points(self):
        """ TODO make this generic (3D) and parallelizable... Tough."""
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
        TODO :
            - define distance
            - test distance"""

    def __init__(self,
                 origin=np.array([0., 0.]),
                 orientation=0.,
                 length=0.8):
        Shape.__init__(self)
        self.origin = origin
        self.orientation = orientation
        self.length = length

    def end_points(self):
        p0 = .5 * self.length * np.array(
            [np.cos(self.orientation), np.sin(self.orientation)])
        p1 = self.origin + p0
        p2 = self.origin + -1. * p0
        return p1, p2

    def sampled_points(self):
        points = []
        p1, p2 = self.end_points()
        for alpha in np.linspace(0., 1., self.nb_points):
            # Linear interpolation
            points.append((1. - alpha) * p1 + alpha * p2)
        return points

    def closest_point(self, x):
        """
        Compute the closest point by projecting to the infite line
        and then checking if the point lines on the segment.
        """
        assert x.shape == self.origin.shape
        p1, p2 = self.end_points()
        u = p2 - p1
        v = x - p1
        d = np.dot(u, v) / np.dot(u, u)
        if d <= 0.:
            p = p1
        elif d >= 1.:
            p = p2
        else:
            p = p1 + d * u
        return p

    def dist_from_border(self, q):
        if q.shape == self.origin.shape:
            p = self.closest_point(q)
            return vector_norm(p - q)
        else:
            p1, p2 = self.end_points()
            u = p2 - p1
            v = (q.T - p1).T
            d = np.tensordot(u, v, axes=1) / np.dot(u, u)
            shape = q.shape
            p = np.full(shape, np.inf)
            is_l_side = d < 0.
            is_r_side = d > 1.
            is_intersection = np.logical_and(d <= 1., d >= 0.)
            for k in range(p.shape[0]):
                p[k] = np.where(is_l_side, p1[k], p[k])
                p[k] = np.where(is_r_side, p2[k], p[k])
                p[k] = np.where(is_intersection, p1[k] + d * u[k], p[k])
            return vector_norm(p - q)

    def dist_hessian(self, x):
        p1, p2 = self.end_points()
        u = p2 - p1
        v = x - p1
        d = np.dot(u, v) / np.dot(u, u)
        if d <= 0.:
            # 1 - closer to p1
            return point_distance_hessian(x, p1)
        elif d >= 1.:
            # 2 - closer to p2
            return point_distance_hessian(x, p2)
        else:
            # 3 - closer to the side
            # Warning: only holds for 2D segments for now
            # in the 3D case we need to implement the Hessian of the distance
            # to a cylinder, which has to do with the projection of the
            # spherical distance to a plane
            assert x.size == 2
            return np.zeros((x.size, x.size))


def segment_from_end_points(p1, p2):
    p12 = p1 - p2
    return Segment(
        origin=(p1 + p2) / 2.,
        orientation=np.arctan2(p12[1], p12[0]),
        length=np.linalg.norm(p12))


class Box(Shape):
    """
        An axis aligned box (hypercube) defined by
            - origin    : its center
            - dim       : its extent

        TODO 1) define distance
             2) class should work for 2D and 3D boxes
             3) test this function
             4) make callable using stack
    """

    def __init__(self,
                 origin=np.array([0., 0.]),
                 dim=np.array([1., 1.])):
        Shape.__init__(self)
        self.origin = origin
        self.dim = dim

    def upper_corner(self):
        return self.origin + .5 * self.dim

    def lower_corner(self):
        return self.origin - .5 * self.dim

    def diag(self):
        return np.linalg.norm(self.dim)

    def is_inside(self, x):
        """
        Returns false if any of the component of the vector
        is smaller or bigger than the lower and top corner
        of the box respectively

        Warning : This is only valid for AxisAligned Box

        Parameters
        ----------
        x : numpy array with
            arbitrary dimensions, 2d and 3d
            or meshgrid data shape = (2 or 3, n, n)
        """
        l_corner = self.lower_corner()
        u_corner = self.upper_corner()
        single = x.shape == (2,) or x.shape == (3,)
        shape = 1 if single else (x.shape[1], x.shape[2])
        inside = np.full(shape, True)
        for k in range(x.shape[0]):
            inside = np.where(np.logical_or(
                x[k] < l_corner[k],
                x[k] > u_corner[k]), False, inside)
        return inside

    def verticies(self):
        """ TODO test """
        v = [None] * 4
        v[0] = self.lower_corner()
        v[1] = np.zeros(2)
        v[1][0] = self.origin[0] + .5 * self.dim[0]
        v[1][1] = self.origin[1] - .5 * self.dim[1]
        v[2] = self.upper_corner()
        v[3] = np.zeros(2)
        v[3][0] = self.origin[0] - .5 * self.dim[0]
        v[3][1] = self.origin[1] + .5 * self.dim[1]
        return v

    def segments(self):
        v = self.verticies()
        s = [None] * 4
        s[0] = segment_from_end_points(v[0], v[1])
        s[1] = segment_from_end_points(v[1], v[2])
        s[2] = segment_from_end_points(v[2], v[3])
        s[3] = segment_from_end_points(v[3], v[0])
        return s

    def closest_segment(self, x):
        min_dist = np.inf
        closest_point = np.zeros(x.shape)
        closest_segment = None
        for segment in self.segments():
            p = segment.closest_point(x)
            d = np.linalg.norm(x - p)
            if min_dist > d:
                min_dist = d
                closest_point = p.copy()
                closest_segment = segment
        return closest_segment, closest_point

    def closest_point(self, x):
        return self.closest_segment(x)[1]

    def dist_hessian(self, x):
        return self.closest_segment(x)[0].dist_hessian(x)

    def dist_from_border(self, x):
        d = [None] * 4
        for i, segment in enumerate(self.segments()):
            d[i] = segment.dist_from_border(x)
        sign = np.where(Box.is_inside(self, x), -1., 1.)
        minimum = np.min(np.array(d), axis=0)
        d = sign * minimum
        return np.asscalar(d) if d.size == 1 else d

    def sample_line(self, p_1, p_2):
        points = []
        for alpha in np.linspace(0., 1., self.nb_points / 4):
            # Linear interpolation
            points.append((1. - alpha) * p_1 + alpha * p_2)
        return points

    def sampled_points(self):
        points = []
        v = self.verticies()
        points.extend(self.sample_line(v[0], v[1]))
        points.extend(self.sample_line(v[1], v[2]))
        points.extend(self.sample_line(v[2], v[3]))
        points.extend(self.sample_line(v[3], v[0]))
        return points


class AxisAlignedBox(Box):
    """
        An axis aligned box (hypercube) defined by
            - origin    : its center
            - dim       : its extent
    """

    def __init__(self,
                 origin=np.array([0., 0.]),
                 dim=np.array([1., 1.])):
        Box.__init__(self, origin, dim)
        """
        In all zones (1) the hessian of the signed distance
        field is zero((n, n)) and gradient (0, 1) or (1, 0)

                   v1 | (1) | v2
                   ___|_____|___
                  (1) | (1) | (1)
                   ___|_____|___
                   v4 | (1) | v3

        ZONES:
                   1  |  5  |  2
                   ___|_____|___
                   8  |  9  |  6
                   ___|_____|___
                   4  |  7  |  3
        """
        self.half_dim = 0.5 * self.dim
        self._v1 = np.array([-self.half_dim[0], self.half_dim[1]])
        self._v2 = np.array([self.half_dim[0], self.half_dim[1]])
        self._v3 = np.array([self.half_dim[0], -self.half_dim[1]])
        self._v4 = np.array([-self.half_dim[0], -self.half_dim[1]])
        self._verticies = [self._v1, self._v2, self._v3, self._v4]

        self._switcher_distance = {
            1: lambda x: vector_norm(x - self._v1),
            2: lambda x: vector_norm(x - self._v2),
            3: lambda x: vector_norm(x - self._v3),
            4: lambda x: vector_norm(x - self._v4),
            5: lambda x: x[1] - self._v1[1],
            6: lambda x: x[0] - self._v3[0],
            7: lambda x: self._v1[1] - x[1],
            8: lambda x: self._v1[0] - x[0],
            9: lambda x: -min(self.half_dim - np.absolute(x))
        }

        self._switcher_gradient = {
            1: lambda x: point_distance_gradient(x, self._v1),
            2: lambda x: point_distance_gradient(x, self._v2),
            3: lambda x: point_distance_gradient(x, self._v3),
            4: lambda x: point_distance_gradient(x, self._v4),
            5: lambda x: np.array([0, 1]),
            6: lambda x: np.array([1, 0]),
            7: lambda x: np.array([0, -1]),
            8: lambda x: np.array([-1, 0]),
            9: lambda x: self._inside_gradient(x)
        }

        self._switcher_hessian = {
            1: lambda x: point_distance_hessian(x, self._v1),
            2: lambda x: point_distance_hessian(x, self._v2),
            3: lambda x: point_distance_hessian(x, self._v3),
            4: lambda x: point_distance_hessian(x, self._v4),
            5: lambda x: np.zeros((self.origin.size, self.origin.size)),
            6: lambda x: np.zeros((self.origin.size, self.origin.size)),
            7: lambda x: np.zeros((self.origin.size, self.origin.size)),
            8: lambda x: np.zeros((self.origin.size, self.origin.size)),
            9: lambda x: np.zeros((self.origin.size, self.origin.size))
        }

    def find_zone(self, x_center):
        """
                   1  |  5  |  2
                   ___|_____|___
                   8  |  9  |  6
                   ___|_____|___
                   4  |  7  |  3
        """
        # ----------------------------------------
        # TOP ------------------------------------
        if x_center[1] > self._v1[1]:

            if x_center[0] < self._v1[0]:
                return 1
            elif x_center[0] > self._v2[0]:
                return 2
            else:
                return 5
        # ----------------------------------------
        # BOTTOM ---------------------------------
        elif x_center[1] < self._v3[1]:

            if x_center[0] < self._v4[0]:
                return 4
            elif x_center[0] > self._v3[0]:
                return 3
            else:
                return 7
        # ----------------------------------------
        # MIDDLE ---------------------------------
        else:
            if x_center[0] < self._v1[0]:
                return 8
            elif x_center[0] > self._v2[0]:
                return 6
            else:
                return 9

    def is_inside(self, x):
        x_center = (x.T - self.origin).T
        return self.find_zone(x_center) == 9

    def _inside_gradient(self, x):
        d1 = np.fabs(self.half_dim[0] - x[0])
        d2 = np.fabs(self.half_dim[1] - x[1])
        d3 = np.fabs(-self.half_dim[0] - x[0])
        d4 = np.fabs(-self.half_dim[1] - x[1])
        switcher = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])}
        idx = np.argmin([d1, d2, d3, d4])
        return switcher[idx]

    def dist_from_border(self, x):
        single = x.shape == (2,) or x.shape == (3,)
        if not single:
            # For now we can not use meshgrid for this class
            return Box.dist_from_border(self, x)
        x_center = (x.T - self.origin).T
        return self._switcher_distance.get(self.find_zone(x_center))(x_center)

    def dist_gradient(self, x):
        x_center = (x.T - self.origin).T
        return self._switcher_gradient.get(self.find_zone(x_center))(x_center)

    def dist_hessian(self, x):
        x_center = (x.T - self.origin).T
        return self._switcher_hessian.get(self.find_zone(x_center))(x_center)


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
    sdf = SignedDistanceWorkspaceMap(workspace)(meshgrid).T
    return (sdf < 0).astype(float)


class EnvBox(Box):
    """
    Specializes a box to defined an environment

    Parameters
    ----------
    origin: is in the center of the box
    dim: size of the box
    """

    def __init__(self,
                 origin=np.array([0., 0.]),
                 dim=np.array([1., 1.])):
        Box.__init__(self, origin, dim)

    def box_extent(self):
        return np.array([self.origin[0] - self.dim[0] / 2.,
                         self.origin[0] + self.dim[0] / 2.,
                         self.origin[1] - self.dim[1] / 2.,
                         self.origin[1] + self.dim[1] / 2.,
                         ])

    def extent(self):
        box_extent = self.box_extent()
        extent = Extent()
        extent.x_min = box_extent[0]
        extent.x_max = box_extent[1]
        extent.y_min = box_extent[2]
        extent.y_max = box_extent[3]
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

    def meshgrid_points(self, nb_points=100):
        """ TODO check what reshape is really doing """
        grid = EnvBox().stacked_meshgrid(nb_points)
        return grid.reshape(nb_points * nb_points, self.origin.size)

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
    return list(zip(centers, radii))


def sample_workspace(nb_circles, radius_parameter=.15):
    """ Samples a workspace randomly composed of nb_circles
        the radius parameter specifies the
        max fraction of workspace diagonal used for a circle radius. """
    workspace = Workspace()
    max_radius = radius_parameter * workspace.box.diag()
    min_radius = .5 * radius_parameter * workspace.box.diag()
    workspace.obstacles = [None] * nb_circles
    for i in range(nb_circles):
        center = workspace.box.sample_uniform()
        radius = (max_radius - min_radius) * np.random.rand() + min_radius
        workspace.obstacles[i] = Circle(center, radius)
    return workspace


def sample_box_workspaces(
        nb_boxes,
        height=.20,
        width=.20):
    """ Samples a workspace randomly composed of nb_boxes
        the height and width parameters specify
        the max fraction of workspace diagonal used for a box. """
    workspace = Workspace()
    diagonal = workspace.box.diag()
    max_h = diagonal * height
    min_h = diagonal * height * .5
    max_w = diagonal * width
    min_w = diagonal * width * .5
    workspace.obstacles = [None] * nb_boxes
    for i in range(nb_boxes):
        origin = workspace.box.sample_uniform()
        h = (max_h - min_h) * np.random.rand() + min_h
        w = (max_w - min_w) * np.random.rand() + min_w
        dimensions = np.array([w, h])
        workspace.obstacles[i] = AxisAlignedBox(origin, dimensions)
    return workspace


def sample_collision_free(workspace, margin=0.):
    """ Samples a collision free point """
    while True:
        p = workspace.box.sample_uniform()
        if margin < workspace.min_dist(p)[0]:
            return p
