#!/usr/bin/env python

# Copyright (c) 2020, University of Stuttgart
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
#                                    Jim Mainprice on Friday January 23 2020

from abc import abstractmethod
from .homogeneous_transform import *
from geometry.workspace import *
import json
import os
import math


class Robot:
    """
    Abstract robot class
    """

    def __init__(self):
        self.name = None
        self.shape = None
        self.keypoints = None
        self._maps = []

    @abstractmethod
    def forward_kinematics_map(self):
        raise NotImplementedError()


class Freeflyer(Robot):
    """
    Planar 3DoFs robot
        verticies are stored counter clock-wise
    """

    def __init__(self,
                 name="freeflyer",
                 keypoints={"base": [0., 0.]},
                 shape=[[0, 0], [0, 1], [1, 1], [1, 0]],
                 radii=1.,
                 scale=1.):
        Robot.__init__(self)
        self.name = name
        self.shape = scale * np.array(shape)
        self.radii = radii * np.ones(len(keypoints))
        self.keypoint_names = {}
        self._keypoints = []
        self._kinematics_maps = [None] * len(keypoints)
        for i, name in enumerate(sorted(keypoints.keys())):
            self._create_map(i, name, scale * np.array(keypoints[name]))
            self._keypoints.append(keypoints[name])

    def _create_map(self, i, name, p):
        self._kinematics_maps[i] = HomogeneousTransform2D(p)
        self.keypoint_names[name] = i

    def keypoint_map(self, i):
        return self._kinematics_maps[i]


def assets_data_dir():
    return os.path.abspath(os.path.dirname(__file__)) + os.sep + "../../data"


def create_freeflyer_from_file(
        filename=assets_data_dir() + "/freeflyer.json",
        scale=None):
    print(filename)
    with open(filename, "r") as read_file:
        config = json.loads(read_file.read())
        robot = Freeflyer(
            config["name"],
            config["keypoints"],
            config["contour"],
            config["radii"],
            config["scale"] if scale is None else scale)
    return robot


def create_robot_with_even_keypoints(
        nb_keypoints=20,
        filename=assets_data_dir() + "/freeflyer.json",
        scale=None):
    print(filename)
    with open(filename, "r") as read_file:
        config = json.loads(read_file.read())
        body = Polygon(
            origin=np.array([0, 0]),
            verticies=np.array(config["shape"]))
        s = np.linspace(0., body.perimeter(), nb_keypoints, endpoint=False)
        points = [body.point_along_perimieter(dl) for dl in s]
        keypoints = {}
        for i, p in enumerate(points):
            keypoints["p{}".format(i)] = p.tolist()
            print("add -> ", p)
        robot = Freeflyer(
            config["name"],
            keypoints,
            config["contour"],
            config["radii"],
            config["scale"] if scale is None else scale)
    return robot


def create_keypoints(nb_keypoints, segments):
    """
    Creates keypoints along a line segment 

    Parameters
    ----------
        nb_keypoints: int
            how many keypoints are sampled along the segments
        segments: list of Segment
            list of segemtns
    """
    length = 0.
    for s in(segments):
        length += s.length()
    keypoints = []
    dl = length / nb_keypoints
    for s in segments:
        k = math.floor(s.length() / dl)
        d_alpha = (1. / float(k))
        d = 0.
        for i in range(int(k)):
            keypoints.append(d * s.p1() + (1 - d) * s.p2())
            d += d_alpha
    return keypoints


def create_freeflyer_from_segments(
        nb_keypoints=20,
        filename=assets_data_dir() + "/freeflyer.json",
        scale=None):
    """
    Loads a freeflyer description from a json file and creates
    a freeflyer with segments equally spaced along the line segments

    Parameters
    ----------
        nb_keypoints: int
            how many keypoints are sampled along the segments
        filename: string
            json file that should be loaded
        scale: float
            how the freeflyer should be scaled
    """
    print(filename)
    with open(filename, "r") as read_file:
        config = json.loads(read_file.read())
        points = config["segments"]
        n = 2 if config["planar"] else 3  # dimensionality of the FF
        segments = [Segment(
            p1=np.array(p[0:n]),
            p2=np.array(p[n:2 * n])) for k, p in sorted(points.items())]
        keypoints = create_keypoints(nb_keypoints, segments)
        keypoints_dic = {}
        for i, point in enumerate(keypoints):
            keypoints_dic["s" + str(i)] = point
        robot = Freeflyer(
            config["name"],
            keypoints_dic,
            config["contour"],
            config["radii"],
            config["scale"] if scale is None else scale)
    return robot
