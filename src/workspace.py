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
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm 
import sys
import math
from pixel_map import *

# This class of Shape represent two dimensional Shapes that can 
# be represented as analytical or other type of functions. The implementations
# should return a set of points on the perimeter of the Shapes.
class Shape:
    def __init__(self):
        self.nb_points = 50

    def SampledPoints(self):
        return self.nb_points*[np.array(2*[0.])]

class Circle(Shape):
    def __init__(self):
        Shape.__init__(self)
        self.origin = np.array([0., 0.])
        self.radius = 0.2

    def __init__(self, c, r):
        Shape.__init__(self)
        self.origin = c
        self.radius = r

    # Signed distance 
    def DistFromBorder(self, x):
        x_center = x - self.origin
        d = np.linalg.norm(x_center)
        return d - self.radius

    def SampledPoints(self):
        points = []
        for theta in np.linspace(0, 2*math.pi, self.nb_points):
            x = self.origin[0] + self.radius * np.cos(theta)
            y = self.origin[1] + self.radius * np.sin(theta)
            points.append(np.array([x, y]))
        return points

# Define a ellipse shape. This is performed using 
# a and b parameters. (a, b) are the size of the great and small radii.
class Ellipse(Shape):
    def __init__(self):
        Shape.__init__(self)
        self.origin = np.array([0., 0.])
        self.a = 0.2
        self.b = 0.2

    def SampledPoints(self):
        points = []
        for theta in np.linspace(0, 2*math.pi, self.nb_points):
            x = self.origin[0] + self.a * np.cos(theta)
            y = self.origin[1] + self.b * np.sin(theta)
            points.append(np.array([x, y]))
        return points

    # Iterative method described, Signed distance 
    # http://www.am.ub.edu/~robert/Documents/ellipse.pdf
    def DistFromBorder(self, x):
        x_abs = math.fabs(x[0])
        y_abs = math.fabs(x[1])
        a_m_b = self.a**2 - self.b**2
        phi = 0.
        for i in range(100):
            phi = math.atan2( a_m_b * math.sin(phi) + y_abs * self.b,
                                x_abs * self.a )
            # print "phi : ", phi
            if phi > math.pi/2:
                break
        return math.sqrt( (x_abs - self.a * math.cos(phi))**2 + 
                          (y_abs - self.b * math.sin(phi))**2 )

class Segment(Shape):
    def __init__(self):
        Shape.__init__(self)
        self.origin = np.array([0., 0.])
        self.orientation = 0.
        self.length = 0.8

    def SampledPoints(self):
        points = []
        p_0 = 0.5 * self.length * np.array([
            np.cos(self.orientation), np.sin(self.orientation)])
        p_1 = self.origin + p_0
        p_2 = self.origin + -1. * p_0 
        for alpha in np.linspace(0., 1., self.nb_points):
            # Linear interpolation
            points.append((1. - alpha) * p_1 + alpha * p_2 )
        return points

class Box:
    def __init__(self):
        self.origin = np.array([0., 0.])
        self.dim = np.array([1., 1.])

    def Extends(self):
        return np.array([self.origin[0] - self.dim[0]/2.,
                         self.origin[0] + self.dim[0]/2.,
                         self.origin[1] - self.dim[1]/2.,
                         self.origin[1] + self.dim[1]/2.,
                         ])


class Workspace:
    def __init__(self):
        self.box = Box()
        self.obstacles = []

    def InCollision(self, pt):
        for obst in self.obstacles:
            if obst.DistFromBorder(pt) < 0.:
                return True
        return False

    def MinDist(self, pt):
        mindist = float("inf")
        minid = -1
        for k, obst in enumerate(self.obstacles):
            dist = obst.DistFromBorder(pt) # Signed distance 
            if dist < mindist:
              mindist = dist
              minid   = k
        return mindist

    def AddCircle(self, origin=None, radius=None):
        if origin is None and radius is None:
            self.obstacles.append(Circle())
        else:
            circle = Circle()
            circle.origin = origin
            circle.radius = radius
            self.obstacles.append(circle)

    def AddSegment(self, origin=None, length=None):
        if origin is None and length is None:
            self.obstacles.append(Segment())
        else:
            segment = Segment()
            segment.origin = origin
            segment.radius = length
            self.obstacles.append(segment)

    def AllPoints(self):
        points = []
        for o in self.obstacles:
            points += o.SampledPoints()
        return points


# run the server
if __name__ == "__main__":
    obstacle = Circle()
    print obstacle.SampledPoints()
