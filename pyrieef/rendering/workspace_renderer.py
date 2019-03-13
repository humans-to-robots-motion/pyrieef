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

from .common_imports import *
from .plannar_gl import *
from learning import random_environment
from geometry.workspace import *
from utils import timer
from utils.misc import *
import random
import time
from skimage import img_as_ubyte
from skimage.color import rgba2rgb
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap = plt.get_cmap('inferno')
from abc import abstractmethod
try:
    from . import heightmap as hm
except ImportError as e:
    print(e)
from itertools import product

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
        self._wait_for_keyboard = False
        self.set_workspace(workspace)
        self.workspace_id = 0

    def set_workspace(self, workspace):
        assert workspace is not None
        self._workspace = workspace
        self._extent = self._workspace.box.extent()

    def reset_objects(self):
        return True

    @abstractmethod
    def draw_ws_circle(self, radius, origin, color=(0, 1, 0)):
        raise NotImplementedError()

    @abstractmethod
    def draw_ws_line(self, line, color=(1, 0, 0)):
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
        self._continuously_drawing = True
        if self._continuously_drawing:
            plt.ion()   # continuously plot
        self._plot3d = False
        self._wait_for_keyboard = wait_for_keyboard
        self.size = scale * np.array([7, 6.5])
        self._fig = None
        self._ax = None
        self._colorbar = None
        self.init(rows, cols)

    def init(self, rows, cols):
        assert rows > 0 and cols > 0
        if self._fig is None:
            self._fig = plt.figure(figsize=self.size)
        if self._ax is not None:
            self._ax.clear()
        self._axes = self._fig.add_subplot(rows, cols, 1)
        if self._colorbar is not None:
            self._colorbar.remove()
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

    def draw_ws_circle(self, radius, origin, color=(0, 1, 0)):
        o = Circle(origin=origin, radius=radius)
        self._ax.plot(o.origin[0], o.origin[1], 'kx')
        points = o.sampled_points()
        X = np.array(points)[:, 0]
        Y = np.array(points)[:, 1]
        self._ax.plot(X, Y, color=color, linewidth=2.0)

    def draw_ws_background(self, phi, nb_points=100):
        X, Y = self._workspace.box.stacked_meshgrid(nb_points)
        Z = phi(np.stack([X, Y])).T
        # Z = two_dimension_function_evaluation(X, Y, phi).T
        self.draw_ws_img(Z)

    def draw_ws_img(self, Z):
        """
        Examples of coloring are : [viridis, hot, bone, magma]
            see page :
            https://matplotlib.org/examples/color/colormaps_reference.html
        """
        color_style = plt.cm.viridis
        im = self._ax.imshow(
            Z.T,
            extent=self._workspace.box.box_extent(),
            origin='lower',
            interpolation='nearest',
            # vmin=0,
            # vmax=100,
            cmap=color_style)
        if self._axes is None:
            self._colorbar = self._fig.colorbar(im, fraction=0.05, pad=0.02)
        # cs = plt.contour(X, Y, Z, 16, cmap=color_style)
        # if self._plot3d:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.plot_surface(X, Y, Z, cmap=color_style,
        #                     linewidth=0, antialiased=False)

    def draw_ws_line(self, line, color='r', color_id=None):
        if color_id is not None:
            color = cm.rainbow(float(color_id % 100) / 20.)
        [self._ax.plot(point[0], point[1], c=color) for point in line]

    def draw_ws_line_fill(self, line, color='r', color_id=None, linewidth=2.0):
        if color_id is not None:
            color = cm.rainbow(float(color_id % 100) / 20.)
        line_x = [point[0] for point in line]
        line_y = [point[1] for point in line]
        self._ax.plot(line_x, line_y, linewidth=linewidth,
                      marker='o', linestyle="-", c=color)

    def draw_ws_point(self, point, color='b', shape='x'):
        self._ax.plot(point[0], point[1], color + shape)

    def show(self):
        if self._continuously_drawing:
            plt.draw()
            plt.pause(0.01)
            self._fig.canvas.flush_events()
        else:
            plt.show()

    def show_once(self, t_sleep=0.0001):
        plt.show(block=False)
        plt.draw()
        plt.pause(t_sleep)
        if self._wait_for_keyboard:
            raw_input("Press Enter to continue...")
        plt.close(self._fig)


class WorkspaceOpenGl(WorkspaceRender):

    """ Workspace display based on pyglet backend """

    def __init__(self, workspace, display=None):
        WorkspaceRender.__init__(self, workspace)
        print((self._workspace.box))
        self._scale = 700.
        self.width = self._scale * (self._extent.x_max - self._extent.x_min)
        self.height = self._scale * (self._extent.y_max - self._extent.y_min)
        self.gl = Viewer(self.width, self.height, display)

        # Get SDF as image
        # signed_distance_field = SignedDistanceWorkspaceMap(self._workspace)
        # self.draw_ws_background(signed_distance_field)

        # Draw WS obstacles
        # self.draw_ws_obstacles()

    def draw_ws_circle(self, radius, origin, color=(0, 1, 0)):
        t = Transform(translation=self._scale * (
            origin - np.array([self._extent.x_min, self._extent.y_min])))
        circ = make_circle(self._scale * radius, 30)
        circ.add_attr(t)
        circ.set_color(*color)
        self.gl.add_onetime(circ)

    def draw_ws_line(self, line, color=(1, 0, 0)):
        p1 = line[0]
        p2 = line[1]
        corner = np.array([self._extent.x_min, self._extent.y_min])
        p1_ws = self._scale * (p1 - corner)
        p2_ws = self._scale * (p2 - corner)
        self.gl.draw_line(p1_ws, p2_ws, linewidth=7, color=(1, 0, 0))

    def draw_ws_background(self, function):
        Z = function(self._workspace.box.stacked_meshgrid())
        # Z = Z.clip(max=1)
        self._max_z = Z.max()
        self._min_z = Z.min()
        Z = (Z - self._min_z * np.ones(Z.shape)) / (self._max_z - self._min_z)
        Z = rgba2rgb(cmap(Z))
        Z = resize(Z, (self.width, self.height))  # Normalize to [0, 1]
        Z = np.flip(Z, 0)
        image = Image(width=self.width, height=self.height,
                      arr=img_as_ubyte(Z))
        image.add_attr(Transform())
        self.gl.add_geom(image)

    def draw_ws_obstacles(self):
        ws_o = np.array([self._extent.x_min, self._extent.y_min])
        for i, o in enumerate(self._workspace.obstacles):
            if isinstance(o, Circle):
                circ = make_circle(self._scale * o.radius, 30, False)
                center = self._scale * (o.origin - ws_o)
                circ.add_attr(Transform(translation=center))
                circ.set_color(*COLORS[i % 3])
                self.gl.add_geom(circ)
            if isinstance(o, Box):
                vertices = [self._scale * (v - ws_o) for v in o.verticies()]
                box = PolyLine(vertices, True)
                box.set_color(*COLORS[i % 3])
                self.gl.add_geom(box)

    def show(self):
        time.sleep(0.05)
        self.gl.render()


class WorkspaceHeightmap(WorkspaceRender):

    """ Workspace display based on pyglet heighmap """

    def __init__(self, workspace):
        WorkspaceRender.__init__(self, workspace)
        self._scale = 1.
        self.width = 30
        self.height = self.width
        self.load_background = True
        self._window = pyglet.window.Window(
            width=int(self._scale * 800),
            height=int(self._scale * 600),
            caption='Heightmap', resizable=True)
        self._height_map = hm.Heightmap()
        self._height_function = None
        self._max_z = None
        self._min_z = None
        self.isopen = True
        self._window.push_handlers(self)
        self._window.on_close = self.window_closed_by_user
        # pyglet.clock.schedule(self.update)
        self._t_render_latest = time.time()
        self.save_images = False
        self.workspace_id = 0
        self.image_id = 0

    def update(self, dt):
        self._height_map.rz -= 2. * dt

    def normalize_height(self, c):
        return (c - self._min_z) / (self._max_z - self._min_z)

    def heightmap_coordinates(self, p, height):
        T = self._height_map.transform(*self._workspace.box.box_extent())
        corner = np.array([self._extent.x_min, self._extent.y_min])
        p_ws = p - corner
        p = T * np.matrix([p_ws[0], p_ws[1], 1]).T
        p_swap = p.copy()
        p_swap[0] = p[1]
        p_swap[1] = p[0]
        p_swap[2] = height
        return p_swap

    def draw_ws_point(self, point, color='b', shape='x'):
        self.draw_ws_sphere(point)

    def draw_ws_line(self, line, color=(1, 0, 0)):
        for p in line:
            z = self.normalize_height(self._height_function(p))
            p_ws = self.heightmap_coordinates(p, z)
            self._height_map.add_sphere_to_draw(p_ws)

    def draw_ws_circle(self, radius, origin, color=(0, 1, 0), height=20.):
        alpha = self._height_map.x_length / self._workspace.box.dim[0]
        p_h = self.heightmap_coordinates(origin, height)
        self._height_map.add_circle_to_draw(p_h, alpha * radius)

    def draw_ws_sphere(self, p, height=20., color=(1, 0, 0)):
        p_h = self.heightmap_coordinates(p, height)
        self._height_map.add_sphere_to_draw(p_h, radius=0.5)

    def draw_ws_background(self, function):
        self._height_function = function
        Z = function(self._workspace.box.stacked_meshgrid(self.width))
        self._max_z = Z.max()
        self._min_z = Z.min()
        Z = (Z - self._min_z * np.ones(Z.shape)) / (self._max_z - self._min_z)
        self._height_map.load(Z, 2, 2, 20.)

    def draw_ws_obstacles(self):
        for o in self._workspace.obstacles:
            if isinstance(o, Circle):
                p = o.origin + o.radius * np.array([0, 1])
                z = self.normalize_height(self._height_function(p) + 20)
                self.draw_ws_circle(o.radius, o.origin, height=z)

    def reset_objects(self):
        self._height_map.objects = []

    def reset_spheres(self):
        objects = self._height_map.objects
        self._height_map.objects = []
        for o in objects:
            if o[0] != "sphere":
                self._height_map.objects.append(o)

    def on_resize(self, width, height):
        hm.resize_gl(width, height)
        self._height_map.draw()
        return pyglet.event.EVENT_HANDLED

    def on_draw(self):
        hm.draw_gl()
        self._height_map.draw()

        # glPolygonMode(GL_FRONT, GL_LINE)  # wire-frame mode
        # height_map.draw(black=True)

    def close(self):
        self._window.close()

    def window_closed_by_user(self):
        self.isopen = False

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

    def render(self, return_rgb_array=False):
        t_render = time.time()
        glClearColor(1, 1, 1, 1)
        self._window.clear()
        self._window.switch_to()
        self._window.dispatch_events()
        self._height_map.draw()
        self.update(t_render - self._t_render_latest)
        arr = None
        if return_rgb_array:
            buffer_p = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer_p.get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(buffer_p.height, buffer_p.width, 4)
            arr = arr[::-1, :, 0:3]
        if self.save_images:
            driectory = "videos/workspace_{0:03d}/".format(self.workspace_id)
            image = 'screenshot_{0:03d}.png'.format(self.image_id)
            make_directory(driectory)
            pyglet.image.get_buffer_manager().get_color_buffer().save(
                driectory + image)
        self._window.flip()
        self._t_render_latest = t_render
        self.image_id += 1
        return arr if return_rgb_array else self.isopen

    def show_once(self, t_sleep=0.2):
        self.render()
        time.sleep(t_sleep)
        self.close()

    def show(self):
        # pyglet.app.run()
        self.render()
