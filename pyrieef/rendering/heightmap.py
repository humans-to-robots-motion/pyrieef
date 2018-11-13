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

import pyglet
from pyglet import *
from pyglet.gl import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
cmap = plt.get_cmap('viridis')
# cmap = plt.get_cmap('inferno')
import numpy as np


# colors
BLACK = (0, 0, 0, 1)
WHITE = (1, 1, 1, 1)
RED = (1, 0, 0, 1)
GRAY = (0.5, 0.5, 0.5)


def flux_qubit_potential(phi_m, phi_p):
    alpha = 0.7
    phi_ext = 2 * np.pi * 0.5
    return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(
        phi_ext - 2 * phi_p)


def potential_surface(nb_points):
    """ generete an abstract mathematical surface """
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
    print((image.shape))
    print((type(image)))
    return image


def sphere_coordinates(lat, lon):
    x = -np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = np.sin(np.radians(lat))
    z = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    s = (lon + 180) / 360.0
    t = (lat + 90) / 180.0
    return (np.array([x, y, z]), [s, t])


def sphere_verticies(origin, radius=1, step=10):
    """ For the some reason the GLU makes colors as shit 
        TODO It would be easier to implement with:
            sphere = gluNewQuadric()
            gluSphere(sphere, o.radius, 50, 50)
        This way of proceeding is so stupid...
        """
    vlists = []
    for lat in range(-90, 90, step):
        verts = []
        texc = []
        for lon in range(-180, 181, step):
            vertex, coord = sphere_coordinates(lat, lon)
            verts += (radius * vertex + origin).tolist()
            texc += coord

            vertex, coord = sphere_coordinates(lat + step, lon)
            verts += (radius * vertex + origin).tolist()
            texc += coord

        vlist = pyglet.graphics.vertex_list(
            int(len(verts) / 3), ('v3f', verts), ('t2f', texc))
        vlists.append(vlist)
    return vlists


def circle_verticies(origin, radius=1, nb_verticies=20, z=0.):
    """ This way of proceeding is so stupid... """
    verts = []
    texc = []
    for theta in np.linspace(0, 2 * np.pi, nb_verticies):
        vertex = np.array([np.cos(theta), np.sin(theta), z])
        verts += (radius * vertex + origin).tolist()
        texc += [(theta + np.pi) / 2 * np.pi, 0]
    return pyglet.graphics.vertex_list(
        int(len(verts) / 3), ('v3f', verts), ('t2f', texc))


class Primitive:

    def __init__(self, object_type, origin, radius, color, alpha):
        self.object_type = object_type
        self.origin = np.array(origin).flatten()
        self.radius = radius
        self.color = color
        self.alpha = alpha
        self.vertices = None
        if object_type == "sphere":
            self.vertices = sphere_verticies(self.origin, self.radius)
        if object_type == "circle":
            self.vertices = circle_verticies(self.origin, self.radius)


class Heightmap:

    def __init__(self):

        self.vertices = []
        self.objects = []

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

        self._dx = self._dy = self._dz = 1.

        # cash sphere object
        self._sphere_primitive = Primitive(
            "sphere", np.zeros(3), 1., (1, 0, 0), 1.)

        self._circle_primitive = Primitive(
            "circle", np.zeros(3), 1., (0, 0, 0), 1.)

    def transform(self, x_min, x_max, y_min, y_max):
        """ From certain box coordinates to heightmap coordinates """
        T = np.eye(3)
        T[0, 0] = -self.x_length / (x_max - x_min)
        T[1, 1] = self.y_length / (y_max - y_min)
        T[0, 2] = self.half_x_length
        T[1, 2] = -self.half_y_length
        return np.matrix(T)

    def get_x(self, i, dx):
        return i * dx - self.half_x_length

    def get_y(self, j, dy):
        return self.half_y_length - j * dy

    def load(self, image, dx, dy, dz):
        """ loads the vertices positions from an image """
        self.vertices = []
        self.objects = []

        self._dx = dx
        self._dy = dy
        self._dz = dz

        # image dimensions
        self.image_width = width = image.shape[0]
        self.image_height = height = image.shape[1]

        # heightmap dimensions
        self.x_length = (self.image_width - 1) * dx
        self.y_length = (self.image_height - 1) * dy

        # used for centering the heightmap
        self.half_x_length = self.x_length / 2.
        self.half_y_length = self.y_length / 2.

        # loads the vertices
        for j in range(height - 1):
            # a row of triangles
            row = []
            for i in range(width):

                # centers the heightmap and inverts the y axis
                r = image[i, j] * dz
                row.extend((self.get_x(i, dx), self.get_y(j, dy), r))

                r = image[i, j + 1] * dz
                row.extend((self.get_x(i, dx), self.get_y(j + 1, dy), r))

            self.vertices.append(row)

        max_z = 0
        self.colors = [None] * len(self.vertices)
        for k, row in enumerate(self.vertices):
            self.colors[k] = []
            for x, y, z in zip(*[iter(row)] * 3):
                max_z = max(max_z, z)
                color = (255 * np.array(cmap(z / dz)[:3])).astype(int)
                self.colors[k].extend(color)
        self.z_length = max_z

    def add_sphere_to_draw(self, origin, radius=1., color=(1, 0, 0), alpha=1.):
        origin_normalzied = origin.copy()
        origin_normalzied[2] *= self._dz
        self.objects.append(("sphere", origin_normalzied))

    def add_circle_to_draw(self, origin, radius=1., color=(1, 0, 0), alpha=1.):
        origin_normalzied = origin.copy()
        origin_normalzied[2] *= self._dz
        verticies = circle_verticies(np.zeros(3), radius)
        self.objects.append(("circle", origin_normalzied, verticies))

    def draw_objects(self):
        _gl_setup()
        for o in self.objects:
            if o[0] == "circle":
                glPushMatrix()
                glLineWidth(3)
                glColor4f(*BLACK)
                glTranslatef(o[1][0], o[1][1], o[1][2])
                o[2].draw(GL_LINE_LOOP)
                glPopMatrix()
            elif o[0] == "sphere":
                glPushMatrix()
                glColor4f(*RED)
                glTranslatef(o[1][0], o[1][1], o[1][2])
                for vlist in self._sphere_primitive.vertices:
                    vlist.draw(GL_TRIANGLE_STRIP)
                glPopMatrix()

    def draw(self, black=False):
        glLoadIdentity()
        glTranslatef(self.x, self.y, self.z - self.z_length * 2)
        glRotatef(self.rx - 40, 1, 0, 0)
        glRotatef(self.ry, 0, 1, 0)
        glRotatef(self.rz - 40, 0, 0, 1)

        # draws the primitives (GL_TRIANGLE_STRIP)
        for i, row in enumerate(self.vertices):
            normals = [0, 0, 1] * int(len(row) / 3)  # Wrong
            colors = (0, 0, 0) * int(len(row) / 3) if black else self.colors[i]
            vlist = pyglet.graphics.vertex_list(
                self.image_width * 2,
                ('v3f/static', row),
                ('t3f/static', normals),
                ('c3B/static', colors))
            vlist.draw(GL_TRIANGLE_STRIP)

        # Draw spheres and other objects
        self.draw_objects()


def _gl_setup():

    def _gl_vector(*args):
        """ Define a simple function to create ctypes arrays of floats """
        return (GLfloat * len(args))(*args)

    # clears the background with the background color
    # glClear(GL_COLOR_BUFFER_BIT)

    light0pos = [20.0,   20.0, 60.0, 1.0]  # positional light !
    light1pos = [-20.0, -20.0, 60.0, 0.0]  # infinitely away light !

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
