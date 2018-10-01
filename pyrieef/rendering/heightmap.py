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

from itertools import izip

import pyglet
from pyglet import *
from pyglet.gl import *

import matplotlib.cm as cm
import matplotlib.pyplot as plt
cmap = plt.get_cmap('viridis')
# cmap = plt.get_cmap('inferno')


# colors
BLACK = (0, 0, 0, 1)
WHITE = (1, 1, 1, 1)
GRAY = (0.5, 0.5, 0.5)


def potential_surface(nb_points):
    """ generete an abstract mathematical surface """
    import numpy as np

    def flux_qubit_potential(phi_m, phi_p):
        alpha = 0.7
        phi_ext = 2 * np.pi * 0.5
        return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(
            phi_ext - 2 * phi_p)
    phi_m = np.linspace(0, 2 * np.pi, nb_points)
    phi_p = np.linspace(0, 2 * np.pi, nb_points)
    X, Y = np.meshgrid(phi_p, phi_m)
    Z = flux_qubit_potential(X, Y).T
    return Z


def image_surface(nb_points):
    """ generate surface from data """
    from skimage import data
    from skimage.util import img_as_float
    from skimage.transform import rescale, resize
    from skimage.color import rgb2gray
    shape = (nb_points, nb_points)
    image = img_as_float(resize(rgb2gray(data.astronaut()), shape))
    print image.shape
    print type(image)
    return image


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
        self.z = -50

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
                r = image[x, y]
                # centers the heightmap and inverts the y axis
                row.extend((
                    x * dx - half_x_length,
                    half_y_length - y * dy,
                    r * dz))
                # gets the maximum component value
                max_z = max(max_z, r)

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
                color = cmap(v_z / dz)
                color = (
                    int(255 * color[0]),
                    int(255 * color[1]),
                    int(255 * color[2]))
                self.colors[i].extend(color)
        print "Done."

        self.z_length = max_z * dz

    def draw(self, black=False):
        glLoadIdentity()
        glTranslatef(self.x, self.y, self.z - self.z_length * 3)
        glRotatef(self.rx - 40, 1, 0, 0)
        glRotatef(self.ry, 0, 1, 0)
        glRotatef(self.rz - 40, 0, 0, 1)
        # color

        # draws the primitives (GL_TRIANGLE_STRIP)
        for i, row in enumerate(self.vertices):
            normals = [0, 0, 1] * (len(row) / 3)  # Wrong
            colors = (0, 0, 0) * (len(row) / 3) if black else self.colors[i]
            vlist = pyglet.graphics.vertex_list(
                self.image_width * 2,
                ('v3f/static', row),
                ('t3f/static', normals),
                ('c3B/static', colors))
            vlist.draw(GL_TRIANGLE_STRIP)


def _gl_setup():

    def _gl_vector(*args):
        """ Define a simple function to create ctypes arrays of floats """
        return (GLfloat * len(args))(*args)

    # clears the background with the background color
    # glClear(GL_COLOR_BUFFER_BIT)

    light0pos = [20.0,   20.0, 20.0, 1.0]  # positional light !
    light1pos = [-20.0, -20.0, 20.0, 0.0]  # infinitely away light !

    glClearColor(1, 1, 1, 1)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)

    glLightfv(GL_LIGHT0, GL_POSITION, _gl_vector(*light0pos))
    glLightfv(GL_LIGHT0, GL_SPECULAR, _gl_vector(.5, .5, 1, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, _gl_vector(1, 1, 1, 1))
    glLightfv(GL_LIGHT1, GL_POSITION, _gl_vector(*light1pos))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, _gl_vector(.5, .5, .5, 1))
    glLightfv(GL_LIGHT1, GL_SPECULAR, _gl_vector(1, 1, 1, 1))

    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def resize_gl(width, height):
    # sets the viewport
    glViewport(0, 0, 2 * width, 2 * height)

    # background color
    glClearColor(*WHITE)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # sets the projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, width / float(height), 0.1, 1000.0)

    # sets the model view
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    _gl_setup()


def draw_gl():
     # background color
    glClearColor(*WHITE)

    # clears the background with the background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    _gl_setup()
    glPolygonMode(GL_FRONT, GL_FILL)  # fill mode
