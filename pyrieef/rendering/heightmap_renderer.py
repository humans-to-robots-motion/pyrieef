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
from . import heightmap as hm

window = pyglet.window.Window(
    width=400, height=400, caption='Heightmap', resizable=True)

image = hm.potential_surface(50)
image = hm.image_surface(50)
image -= image.min()
image /= image.max()

# heightmap
height_map = hm.Heightmap()
height_map.load(image, 1, 1, 10.)


def update(dt):
    height_map.rz -= 10. * dt
pyglet.clock.schedule(update)


@window.event
def on_resize(width, height):
    hm.resize_gl(width, height)
    height_map.draw()
    return pyglet.event.EVENT_HANDLED


@window.event
def on_draw():
    hm.draw_gl()
    height_map.draw()

    # glPolygonMode(GL_FRONT, GL_LINE)  # wire-frame mode
    # height_map.draw(black=True)


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
