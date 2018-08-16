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

# Red, Green, Blue
COLORS = [(139, 0, 0),  (0, 100, 0), (0, 0, 139)]


class WorkspaceRender(Viewer):

    def __init__(self, workspace, display=None):
        self._workspace = workspace
        extends = workspace.box.extends()
        width = extends.x_max - extends.x_min
        height = extends.y_max - extends.y_min
        scale = 400
        Viewer.__init__(self, scale * width, scale * height, display)
        for i, o in enumerate(self._workspace.obstacles):
            if isinstance(o, Circle):
                circ = make_circle(scale * o.radius, 30)
                origin = o.origin - np.array([extends.x_min, extends.y_min])
                t = Transform(translation=scale * origin)
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
