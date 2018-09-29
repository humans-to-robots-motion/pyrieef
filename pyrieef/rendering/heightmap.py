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

import os

import pyglet
from pyglet import gl
from pyglet.gl import glu

from itertools import izip
import matplotlib.cm as cm
import matplotlib.pyplot as plt
cmap = plt.get_cmap('viridis')

from skimage import data
from skimage.util import img_as_float
from skimage.transform import rescale, resize
from skimage.color import rgb2gray

# colors
BLACK = (0, 0, 0, 1)
WHITE = (1, 1, 1, 1)
GRAY = (0.5, 0.5, 0.5)


class Heightmap:

    def __init__(self):
        self.vertices = []

        # heightmap dimensions
        self.x_length = 0
        self.y_length = 0
        self.z_length = 0

        # image dimensions
        self.image_width = 0
        self.image_height = 0

        # translation and rotation values
        self.x = self.y = self.z = 0  # heightmap translation
        self.rx = self.ry = self.rz = 0  # heightmap rotation
        self.z = -40

    # def get_verticies(self, Z):

    def load(self, image, dx, dy, dz):
        """ loads the vertices positions from an image """

        # image dimensions
        self.image_width = width = image.shape[0]
        self.image_height = height = image.shape[0]

        # heightmap dimensions
        self.x_length = (self.image_width - 1) * dx
        self.y_length = (self.image_height - 1) * dy

        # used for centering the heightmap
        half_x_length = self.x_length / 2.
        half_y_length = self.y_length / 2.

        max_z = 0

        # loads the vertices
        for y in xrange(height - 1):
            # a row of triangles
            row = []
            for x in xrange(width):
                # gets the red component of the pixel
                # in a grayscale image; the red, green and blue components have
                # the same value
                r = image[x, y]
                # centers the heightmap and inverts the y axis
                row.extend((
                    x * dx - half_x_length,
                    half_y_length - y * dy,
                    r * dz))
                # gets the maximum component value
                max_z = max(max_z, r)

                # gets the red component of the pixel
                # in a grayscale image; the red, green and blue components have
                # the same value
                r = image[x, y + 1]
                # centers the heightmap and inverts the y axis
                row.extend((
                    x * dx - half_x_length,
                    half_y_length - (y + 1) * dy,
                    r * dz))
                # gets the maximum component value
                max_z = max(max_z, r)

            self.vertices.append(row)

        self.colors = [None] * len(self.vertices)
        for i, row in enumerate(self.vertices):
            self.colors[i] = []
            for v_x, v_y, v_z in izip(*[iter(row)] * 3):
                color = cmap(v_z)
                color = (
                    int(255 * color[0]),
                    int(255 * color[1]),
                    int(255 * color[2]))
                self.colors[i].extend(color)
        print "Done."

        self.z_length = max_z * dz

    def draw(self, black=False):
        gl.glLoadIdentity()
        # position (move away 3 times the
        # z_length of the heightmap in the z
        # axis)
        gl.glTranslatef(self.x, self.y, self.z - self.z_length * 3)
        # rotation
        gl.glRotatef(self.rx - 40, 1, 0, 0)
        gl.glRotatef(self.ry, 0, 1, 0)
        gl.glRotatef(self.rz - 40, 0, 0, 1)
        # color
        # gl.glColor3f(*BLACK)

        # draws the primitives (GL_TRIANGLE_STRIP)
        for i, row in enumerate(self.vertices):
            normals = [0, 0, 1] * (len(row) / 3)
            colors = (0, 0, 0) * (len(row) / 3) if black else self.colors[i]
            # print color
            assert len(row) == len(normals), "{} {}".format(
                len(row), len(normals))
            assert len(colors) == len(normals), "{} {}".format(
                len(colors), len(normals))

            vlist = pyglet.graphics.vertex_list(
                self.image_width * 2,
                ('v3f', row),
                ('t3f', normals),
                ('c3B', colors))
            vlist.draw(gl.GL_TRIANGLE_STRIP)

window = pyglet.window.Window(
    width=1000, height=800, caption='Heightmap', resizable=True)

# background color
gl.glClearColor(*WHITE)

# image
image = img_as_float(resize(rgb2gray(data.astronaut()), (50, 50)))
print image.shape

# heightmap
height_map = Heightmap()
height_map.load(image, 1, 1, 3.)


@window.event
def on_resize(width, height):
    # sets the viewport
    gl.glViewport(0, 0, width, height)

    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    # sets the projection
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(60.0, width / float(height), 0.1, 1000.0)

    # sets the model view
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

    return pyglet.event.EVENT_HANDLED


@window.event
def on_draw():
    # clears the background with the background color
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)  # fill mode
    height_map.draw()

    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)  # wire-frame mode
    height_map.draw(black=True)


@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    # scroll the MOUSE WHEEL to zoom
    height_map.z -= scroll_y / 1.0


@window.event
def on_mouse_drag(x, y, dx, dy, button, modifiers):
    # press the LEFT MOUSE BUTTON to rotate
    if button == pyglet.window.mouse.LEFT:
        height_map.ry += dx / 5.0
        height_map.rx -= dy / 5.0
    # press the LEFT and RIGHT MOUSE BUTTON simultaneously to pan
    if button == pyglet.window.mouse.MIDDLE:
        height_map.x += dx / 10.0
        height_map.y += dy / 10.0


pyglet.app.run()
