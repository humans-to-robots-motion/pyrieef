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

from common_imports import *
from plannar_gl import *
from learning import random_environment
from geometry.workspace import *
from utils import timer
import random
import time
from skimage import img_as_ubyte
from skimage.color import rgba2rgb
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap = plt.get_cmap('inferno')
from abc import abstractmethod
import heightmap as hm

# Red, Green, Blue
COLORS = [(139, 0, 0),  (0, 100, 0), (0, 0, 139)]


def to_rgb3(im):
    """
     we can use dstack and an array copy
     this has to be slow, we create an array with
     3x the data we need and truncate afterwards
    """
    return np.asarray(np.dstack((im, im, im)), dtype=np.uint8)


class WorkspaceRender:

    """ Abstract class to draw a 2D workspace """

    def __init__(self, workspace):
        self.set_workspace(workspace)

    def set_workspace(self, workspace):
        self._workspace = workspace
        self._extends = workspace.box.extent()

    @abstractmethod
    def draw_ws_circle(self, radius, origin, color=(0, 1, 0)):
        raise NotImplementedError()

    @abstractmethod
    def draw_ws_line(self, p1, p2, color=(1, 0, 0)):
        raise NotImplementedError()

    @abstractmethod
    def draw_ws_background(self, function):
        raise NotImplementedError()

    @abstractmethod
    def draw_ws_obstacles(self):
        raise NotImplementedError()


class WorkspaceDrawer(WorkspaceRender):

    """ Workspace display based on matplotlib backend """

    def __init__(self, workspace, wait_for_keyboard=False,
                 rows=1, cols=1, scale=1.):
        WorkspaceRender.__init__(self, workspace)
        plt.rcParams.update({'font.size': int(scale * 5)})
        self._plot3d = False
        self._wait_for_keyboard = wait_for_keyboard
        self.size = scale * np.array([7, 6.5])
        self.init(rows, cols)

    def init(self, rows, cols):
        assert rows > 0 and cols > 0
        self._fig, self._axes = plt.subplots(rows, cols, figsize=self.size)
        if rows > 1 or cols > 1:
            for ax in self._axes.flatten():
                ax.axis('equal')
                ax.axis(self._workspace.box.box_extent())
            self.set_drawing_axis(0)
        else:
            self._ax = self._axes
            self._ax.axis('equal')
            self._ax.axis(self._workspace.box.box_extent())
            self._axes = None

    def set_drawing_axis(self, i):
        assert i >= 0
        if self._axes is not None:
            self._ax = self._axes.flatten()[i]
            self._ax.axis('equal')
            self._ax.axis(self._workspace.box.box_extent())

    def draw_ws_obstacles(self):
        colorst = [cm.gist_ncar(i) for i in np.linspace(
            0, 0.9, len(self._workspace.obstacles))]
        for i, o in enumerate(self._workspace.obstacles):
            self._ax.plot(o.origin[0], o.origin[1], 'kx')
            points = o.sampled_points()
            X = np.array(points)[:, 0]
            Y = np.array(points)[:, 1]
            self._ax.plot(X, Y, color=colorst[i], linewidth=2.0)

    def draw_ws_background(self, phi, nb_points=100):
        X, Y = self._workspace.box.meshgrid(nb_points)
        Z = phi(np.stack([X, Y])).transpose()
        self.draw_ws_img(Z)

    def draw_ws_img(self, Z):
        """ 
        Examples of coloring are : [viridis, hot, bone, magma]
            see page : 
            https://matplotlib.org/examples/color/colormaps_reference.html
        """
        color_style = plt.cm.viridis
        im = self._ax.imshow(
            Z.transpose(),
            extent=self._workspace.box.box_extent(),
            origin='lower',
            interpolation='nearest',
            # vmin=0,
            # vmax=100,
            cmap=color_style)
        if self._axes is None:
            self._fig.colorbar(im, fraction=0.05, pad=0.02)
        # cs = plt.contour(X, Y, Z, 16, cmap=color_style)
        # if self._plot3d:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.plot_surface(X, Y, Z, cmap=color_style,
        #                     linewidth=0, antialiased=False)

    def draw_ws_line(self, line, color='r'):
        [self._ax.plot(point[0], point[1], color + 'o') for point in line]

    def draw_ws_line_fill(self, line, color='r'):
        line_x = [point[0] for point in line]
        line_y = [point[1] for point in line]
        self._ax.plot(line_x, line_y, color)

    def draw_ws_point(self, point, color='b', shape='x'):
        self._ax.plot(point[0], point[1], color + shape)

    def show(self):
        plt.show()

    def show_once(self):
        plt.show(block=False)
        plt.draw()
        plt.pause(0.0001)
        if self._wait_for_keyboard:
            raw_input("Press Enter to continue...")
        plt.close(self._fig)


class WorkspaceOpenGl(WorkspaceRender):

    """ Workspace display based on pyglet backend """

    def __init__(self, workspace, display=None):
        WorkspaceRender.__init__(self, workspace)
        print self._workspace.box
        self._scale = 700.
        self.width = self._scale * (self._extends.x_max - self._extends.x_min)
        self.height = self._scale * (self._extends.y_max - self._extends.y_min)
        self.gl = Viewer(self.width, self.height, display)

        # Get SDF as image
        # signed_distance_field = SignedDistanceWorkspaceMap(self._workspace)
        # self.draw_ws_background(signed_distance_field)

        # Draw WS obstacles
        # self.draw_ws_obstacles()

    def draw_ws_circle(self, radius, origin, color=(0, 1, 0)):
        t = Transform(translation=self._scale * (
            origin - np.array([self._extends.x_min, self._extends.y_min])))
        circ = make_circle(self._scale * radius, 30)
        circ.add_attr(t)
        circ.set_color(*color)
        self.gl.add_onetime(circ)

    def draw_ws_line(self, p1, p2, color=(1, 0, 0)):
        corner = np.array([self._extends.x_min, self._extends.y_min])
        p1_ws = self._scale * (p1 - corner)
        p2_ws = self._scale * (p2 - corner)
        self.gl.draw_line(p1_ws, p2_ws, linewidth=7, color=(1, 0, 0))

    def draw_ws_background(self, function):
        Z = function(self._workspace.box.stacked_meshgrid())
        Z = (Z - np.ones(Z.shape) * Z.min()) / Z.max()
        Z = rgba2rgb(cmap(Z))
        Z = resize(Z, (self.width, self.height))  # Normalize to [0, 1]
        Z = np.flip(Z, 0)
        image = Image(width=self.width, height=self.height,
                      arr=img_as_ubyte(Z))
        image.add_attr(Transform())
        self.gl.add_geom(image)

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
                self.gl.add_geom(circ)

    def show(self):
        self.gl.render()


class WorkspaceHeightmap(WorkspaceRender):

    """ Workspace display based on pyglet heighmap """

    def __init__(self, workspace):
        WorkspaceRender.__init__(self, workspace)
        print self._workspace.box
        self._scale = 1.
        self.width = 50
        self.height = 50
        self.load_background = True
        self._window = window = pyglet.window.Window(
            width=int(self._scale * 400),
            height=int(self._scale * 400),
            caption='Heightmap', resizable=True)
        self._height_map = hm.Heightmap()
        pyglet.clock.schedule(self.update)
        self._window.push_handlers(self)

    def update(self, dt):
        self._height_map.rz -= 10. * dt

    def draw_ws_circle(self, radius, origin, color=(0, 1, 0)):
        # t = Transform(translation=self._scale * (
        #     origin - np.array([self._extends.x_min, self._extends.y_min])))
        # circ = make_circle(self._scale * radius, 30)
        # circ.add_attr(t)
        # circ.set_color(*color)
        # self.gl.add_onetime(circ)
        return None

    def draw_ws_line(self, p1, p2, color=(1, 0, 0)):
        # corner = np.array([self._extends.x_min, self._extends.y_min])
        # p1_ws = self._scale * (p1 - corner)
        # p2_ws = self._scale * (p2 - corner)
        # self.gl.draw_line(p1_ws, p2_ws, linewidth=7, color=(1, 0, 0))
        return None

    def draw_ws_background(self, function):
        Z = function(self._workspace.box.stacked_meshgrid(self.width))
        print Z.shape
        Z = (Z - np.ones(Z.shape) * Z.min()) / Z.max()
        # Z = np.flip(Z, 1)
        self._height_map.load(Z, 2, 2, 50.)

    def draw_ws_obstacles(self):
        # for i, o in enumerate(self._workspace.obstacles):
        #     if isinstance(o, Circle):
        #         circ = make_circle(self._scale * o.radius, 30)
        #         origin = o.origin - np.array(
        #             [self._extends.x_min, self._extends.y_min])
        #         t = Transform(translation=self._scale * origin)
        #         print "o.origin {}, o.radius {}".format(o.origin, o.radius)
        #         circ.add_attr(t)
        #         circ.set_color(*COLORS[i])
        #         self.gl.add_geom(circ)
        return None

    def on_resize(self, width, height):
        hm.resize_gl(width, height)
        self._height_map.draw()
        return pyglet.event.EVENT_HANDLED

    def on_draw(self):
        hm.draw_gl()
        self._height_map.draw()

        # glPolygonMode(GL_FRONT, GL_LINE)  # wire-frame mode
        # height_map.draw(black=True)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        # scroll the MOUSE WHEEL to zoom
        self._height_map.z -= scroll_y / 1.0

    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        # press the LEFT MOUSE BUTTON to rotate
        if button == pyglet.window.mouse.LEFT:
            self._height_map.ry += dx / 5.0
            self._height_map.rx -= dy / 5.0
        # press the LEFT and RIGHT MOUSE BUTTON simultaneously to pan
        if button == pyglet.window.mouse.MIDDLE:
            self._height_map.x += dx / 10.0
            self._height_map.y += dy / 10.0

    def show(self):
        pyglet.app.run()
