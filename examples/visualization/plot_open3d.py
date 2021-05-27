#!/usr/bin/env python

# Copyright (c) 2021, University of Stuttgart
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
#                                        Jim Mainprice on Wed April 21 2021

import common_imports
from utils.timer import Rate

import open3d as o3d
import numpy as np
import time

# Download bunny at http://graphics.stanford.edu/data/3Dscanrep/"
DATA_DIRECTORY = "/Users/jmainpri/Dropbox/Work/workspace/pyrieef/data/"
# BUNNY = DATA_DIRECTORY + "bunny/data/bun045.ply"
# BUNNY = DATA_DIRECTORY + "bunny/reconstruction/bun_zipper_res2.ply"
BUNNY = DATA_DIRECTORY + "bunny/reconstruction/bun_zipper.ply"
BUNNY = DATA_DIRECTORY + "scan.stl"

ROBOTS = "/Users/jmainpri/Dropbox/Work/workspace/pybullet_robots/"
BUNNY = ROBOTS + "data/baxter_common/baxter_description/meshes/lower_elbow/E1.STL"

# print("Let's define some primitives")

mesh_box = o3d.geometry.TriangleMesh.create_box(
    width=1.0, height=1.0, depth=1.0)
mesh_box.compute_vertex_normals()
mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
    radius=1.0)

mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
    radius=0.3, height=4.0)

mesh_cylinder.compute_vertex_normals()
mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[-2, -2, -2])

mesh_bunny = o3d.io.read_triangle_mesh(BUNNY)
mesh_bunny.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_bunny])

# print("We draw a few primitives using collection.")
# o3d.visualization.draw_geometries(
#     [mesh_box, mesh_sphere, mesh_cylinder, mesh_frame])

# # print("We draw a few primitives using + operator of mesh.")
# # o3d.visualization.draw_geometries(
# #     [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])

# width = 800
# height = 600

# rate = Rate(20) # Hz
# viz = o3d.visualization.Visualizer()
# viz.create_window(width=width, height=height, visible=True)
# viz.add_geometry(mesh_box)
# for i in range(100000):
#     # do ICP single iteration
#     # transform geometry using ICP
#     viz.update_geometry(mesh_bunny)
#     viz.poll_events()
#     viz.update_renderer()
#     # rate.sleep() # TODO make that work...
