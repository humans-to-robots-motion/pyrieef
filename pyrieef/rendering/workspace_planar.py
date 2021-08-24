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
from .gl_planar import *
from learning import random_environment
from geometry.workspace import *
from geometry.utils import *
from utils import timer
from utils.misc import *
import random
import time
from textwrap import wrap
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
COLORS = [(139, 0, 0), (0, 100, 0), (0, 0, 139)]


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
        self.background_matrix_eval = True

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
    def draw_ws_polygon(self, polygon, color=(1, 0, 0)):
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
                 rows=1, cols=1, scale=1., dynamic=False):
        WorkspaceRender.__init__(self, workspace)
        plt.rcParams.update({'font.size': int(scale * 5)})
        self._continuously_drawing = dynamic
        if self._continuously_drawing:
            plt.ion()  # continuously plot
        self._plot3d = False
        self._wait_for_keyboard = wait_for_keyboard
        self.size = scale * np.array([cols * 7, rows * 6.5])
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
        self._axes = self._fig.subplots(nrows=rows, ncols=cols)
        if rows > 1 or cols > 1:
            for ax in self._axes.flat:
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

    def draw_ws_polygon(self, vertices, origin, rotation, color=(1, 0, 0)):
        self._ax.plot(origin[0], origin[1], 'kx')
        R = rotation_matrix_2d_radian(rotation)
        vertices = np.vstack([vertices, vertices[0]])
        for k, v in enumerate(vertices):
            vertices[k] = np.dot(R, v) + origin
        X = np.array(vertices)[:, 0]
        Y = np.array(vertices)[:, 1]
        self._ax.plot(X, Y, color=color, linewidth=2.0)

    def draw_ws_background(self, phi, nb_points=100,
                           color_style=plt.cm.magma, interpolate="bilinear"):
        X, Y = self._workspace.box.stacked_meshgrid(nb_points)
        if self.background_matrix_eval:
            Z = phi(np.stack([X, Y])).T
        else:
            Z = two_dimension_function_evaluation(X, Y, phi).T
        self.draw_ws_img(Z, interpolate, color_style)

    def draw_ws_img(self, Z, interpolate="nearest", color_style=plt.cm.magma):
        """
        Draws an image in the background

        Parameters:
            Z : image numpy array
            interpolate : ["nearest", "none", "bicubic", "bilinear"]
            color_style : [viridis, hot, bone, magma]

        Examples of coloring are : [viridis, hot, bone, magma]
            see page :
            https://matplotlib.org/examples/color/colormaps_reference.html
        """
        im = self._ax.imshow(
            Z.T,
            extent=self._workspace.box.box_extent(),
            origin='lower',
            interpolation=interpolate,
            cmap=color_style)
        if self._axes is None:
            if self._colorbar is not None:
                self._colorbar.remove()
            self._colorbar = self._fig.colorbar(im, fraction=0.05, pad=0.02)

    def draw_ws_line(self, line, color='r', color_id=None):
        if color_id is not None:
            color = cm.rainbow(float(color_id % 100) / 20.)
        [self._ax.plot(point[0], point[1], 'o', c=color) for point in line]

    def draw_ws_line_fill(self, line, color='r', color_id=None, linewidth=2.0):
        """ draws a line where points are given as a list """
        if color_id is not None:
            color = cm.rainbow(float(color_id % 100) / 20.)
        line_x = [point[0] for point in line]
        line_y = [point[1] for point in line]
        self._ax.plot(line_x, line_y, linewidth=linewidth,
                      marker='o', linestyle="-", c=color)

    def draw_ws_point(self, point, color='b', shape='x'):
        self._ax.plot(point[0], point[1], shape, c=color)

    def show(self):
        if self._continuously_drawing:
            plt.draw()
            plt.pause(0.001)
            self._fig.canvas.flush_events()
        else:
            plt.show()

    def show_once(self, t_sleep=0.0001, close_window=True):
        """
        Notes
            Use close_window=False with viewer._ax.clear()

            Example:

                viewer = render.WorkspaceDrawer(
                        rows=1, cols=1, workspace=workspace, 
                        wait_for_keyboard=True)
                viewer.set_drawing_axis(0)

                for i in range(10):
                    viewer._ax.clear()
                    viewer.draw_ws_img(A, interpolate="bilinear")
                    viewer.draw_ws_obstacles()
                    viewer.draw_ws_point(p, "r")
                    viewer.show_once(close_window=False)
        """
        plt.show(block=False)
        plt.draw()
        plt.pause(t_sleep)
        if self._wait_for_keyboard:
            input("Press Enter to continue...")
        if close_window:
            plt.close(self._fig)

    def set_title(self, title, fontsize=15):
        plt.title(
            '\n'.join(
                wrap(title, int(self.size[0]) * 7)), fontsize=fontsize)

    def remove_axis(self):
        if self._axes is not None:
            self._ax.axis('off')
        else:
            plt.axis('off')

    def save_figure(self, path):
        plt.savefig(path)


class WorkspaceOpenGl(WorkspaceRender):
    """ Workspace display based on pyglet backend """

    def __init__(self, workspace,  wait_for_keyboard=False,
                 display=None, scale=700.):
        WorkspaceRender.__init__(self, workspace)
        self._scale = scale
        self.width = self._scale * (self._extent.x_max - self._extent.x_min)
        self.height = self._scale * (self._extent.y_max - self._extent.y_min)
        self.gl = Viewer(self.width, self.height, display)
        self._wait_for_keyboard = wait_for_keyboard
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

    def draw_ws_point(self, point, color='b', shape='x'):
        self.draw_ws_circle(.01, point)

    def draw_ws_line(self, line, color=(1, 0, 0)):
        p1 = line[0]
        p2 = line[1]
        corner = np.array([self._extent.x_min, self._extent.y_min])
        p1_ws = self._scale * (p1 - corner)
        p2_ws = self._scale * (p2 - corner)
        self.gl.draw_line(p1_ws, p2_ws, linewidth=7, color=(1, 0, 0))

    def draw_ws_polygon(self, vertices, origin, rotation, color=(1, 0, 0)):
        t = Transform(
            translation=self._scale * (
                origin - np.array([self._extent.x_min, self._extent.y_min])),
            rotation=rotation)
        polygon = make_polygon(self._scale * vertices, filled=False)
        polygon.add_attr(t)
        polygon.set_color(*color)
        self.gl.add_onetime(polygon)

    def draw_ws_background(self, phi,
                           nb_points=100,
                           color_style=plt.cm.magma,
                           interpolate="bilinear"):
        X, Y = self._workspace.box.stacked_meshgrid(nb_points)
        if self.background_matrix_eval:
            Z = phi(np.stack([X, Y]))
        else:
            Z = two_dimension_function_evaluation(X, Y, phi)
        self.draw_ws_img(Z)

    def draw_ws_img(self, Z):
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
            if hasattr(o, '_is_circle'):
                circ = make_circle(self._scale * o.radius, 30, False)
                center = self._scale * (o.origin - ws_o)
                circ.add_attr(Transform(translation=center))
                circ.set_color(*COLORS[i % 3])
                self.gl.add_geom(circ)
            if hasattr(o, '_is_box'):
                vertices = [self._scale * (v - ws_o) for v in o.verticies()]
                box = PolyLine(vertices, True)
                box.set_color(*COLORS[i % 3])
                self.gl.add_geom(box)
            if hasattr(o, '_is_oriented_box'):
                vertices = [self._scale * (v - o.origin) for v in o.verticies()]
                box = PolyLine(vertices, True)
                center = self._scale * (o.origin - ws_o)
                box.add_attr(Transform(translation=center, rotation=o.theta()))
                box.set_color(*COLORS[i % 3])
                self.gl.add_geom(box)

    def show_once(self):
        self.show()
        if self._wait_for_keyboard:
            input("Press Enter to continue...")

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
