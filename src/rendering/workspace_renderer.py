#!/usr/bin/env python

# Copyright (c) 2018 University of Stuttgart
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

from common_imports import *
from opengl import *
from learning import random_environment
from geometry.workspace import *
import time
import random
from skimage import img_as_ubyte
from skimage.color import rgba2rgb
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
cmap = plt.get_cmap('inferno')


# Red, Green, Blue
COLORS = [(139, 0, 0),  (0, 100, 0), (0, 0, 139)]


def to_rgb3(im):
    # we can use dstack and an array copy
    # this has to be slow, we create an array with
    # 3x the data we need and truncate afterwards
    return np.asarray(np.dstack((im, im, im)), dtype=np.uint8)


class WorkspaceRender(Viewer):

    def __init__(self, workspace, display=None):
        self._workspace = workspace
        self._extends = workspace.box.extends()
        self._scale = 700.
        self.width = self._scale * (self._extends.x_max - self._extends.x_min)
        self.height = self._scale * (self._extends.y_max - self._extends.y_min)
        Viewer.__init__(self, self.width, self.height, display)

        # Get SDF as image
        # signed_distance_field = SignedDistanceWorkspaceMap(self._workspace)
        # self.draw_ws_background(signed_distance_field)

        # Draw WS obstacles
        # self.draw_ws_obstacles()

    def draw_ws_circle(self, radius, origin):
        t = Transform(translation=self._scale * (
            origin - np.array([self._extends.x_min, self._extends.y_min])))
        circ = make_circle(self._scale * radius, 30)
        circ.add_attr(t)
        circ.set_color(0, 1, 0)
        self.add_onetime(circ)

    def draw_ws_line(self, p1, p2):
        corner = np.array([self._extends.x_min, self._extends.y_min])
        p1_ws = self._scale * (p1 - corner)
        p2_ws = self._scale * (p2 - corner)
        self.draw_line(p1_ws, p2_ws, linewidth=7, color=(1, 0, 0))

    def draw_ws_background(self, function):
        X, Y = self._workspace.box.meshgrid()
        Z = function(np.stack([X, Y]))
        Z = (Z - np.ones(Z.shape) * Z.min()) / Z.max()
        Z = rgba2rgb(cmap(Z))
        Z = resize(Z, (self.width, self.height))  # Normalize to [0, 1]
        Z = np.flip(Z, 0)
        image = Image(width=self.width, height=self.height,
                      arr=img_as_ubyte(Z))
        image.add_attr(Transform())
        self.add_geom(image)

    def draw_ws_obstacles(self):
        for i, o in enumerate(self._workspace.obstacles):
            if isinstance(o, Circle):
                circ = make_circle(self._scale * o.radius, 30)
                origin = o.origin - np.array(
                    [self._extends.x_min, self._extends.y_min])
                t = Transform(translation=self._scale * origin)
                print "o.origin {}, o.radius {}".format(o.origin, o.radius)
                circ.add_attr(t)
                circ.set_color(*COLORS[i])
                self.add_geom(circ)


if __name__ == '__main__':
    # random.seed(0)
    box = Box()
    workspace = random_environment.sample_circle_workspace(box)
    viewer = WorkspaceRender(workspace)
    while True:
        viewer.render()
        time.sleep(0.01)
