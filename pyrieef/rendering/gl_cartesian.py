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

# Copy of 
# https://github.com/rock-learning/pytransform3d/blob/master/\
# pytransform3d/visualizer.py

"""Optional 3D renderer based on Open3D's visualizer."""
import warnings
try:
    import open3d as o3d
    import numpy as np
    from . import rotations as pr
    from . import transformations as pt
    from . import trajectories as ptr
    from . import urdf
    from itertools import chain

    def figure(window_name="Open3D", width=1920, height=1080,
               with_key_callbacks=False):
        """Create a new figure.

        Parameters
        ----------
        window_name : str, optional (default: Open3D)
            Window title name.

        width : int, optional (default: 1920)
            Width of the window.

        height : int, optional (default: 1080)
            Height of the window.

        with_key_callbacks : bool, optional (default: False)
            Creates a visualizer that allows to register callbacks
            for keys.

        Returns
        -------
        figure : Figure
            New figure.
        """
        return Figure(window_name, width, height, with_key_callbacks)

    class Figure:
        """The top level container for all the plot elements.

        You can close the visualizer with the keys `escape` or `q`.

        Parameters
        ----------
        window_name : str, optional (default: Open3D)
            Window title name.

        width : int, optional (default: 1920)
            Width of the window.

        height : int, optional (default: 1080)
            Height of the window.

        with_key_callbacks : bool, optional (default: False)
            Creates a visualizer that allows to register callbacks
            for keys.
        """

        def __init__(self, window_name="Open3D", width=1920, height=1080,
                     with_key_callbacks=False):
            if with_key_callbacks:
                self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
            else:
                self.visualizer = o3d.visualization.Visualizer()
            self.visualizer.create_window(
                window_name=window_name, width=width, height=height)

        def add_geometry(self, geometry):
            """Add geometry to visualizer.

            Parameters
            ----------
            geometry : Geometry
                Open3D geometry.
            """
            self.visualizer.add_geometry(geometry)

        def update_geometry(self, geometry):
            """Indicate that geometry has been updated.

            Parameters
            ----------
            geometry : Geometry
                Open3D geometry.
            """
            self.visualizer.update_geometry(geometry)

        def set_line_width(self, line_width):
            """Set render option line width.

            Note: this feature does not work in Open3D's visualizer at the
            moment.

            Parameters
            ----------
            line_width : float
                Line width.
            """
            self.visualizer.get_render_option().line_width = line_width
            self.visualizer.update_renderer()

        def set_zoom(self, zoom):
            """Set zoom.

            Parameters
            ----------
            zoom : float
                Zoom of the visualizer.
            """
            self.visualizer.get_view_control().set_zoom(zoom)

        def animate(self, callback, n_frames, loop=False, fargs=()):
            """Make animation with callback.

            Parameters
            ----------
            callback : callable
                Callback that will be called in a loop to update geometries.
                The first input of the function will be the current frame
                index from [0, `n_frames`). Further arguments can be given as
                `fargs`. The function should return one artist object or a
                list of artists that have been updated.

            n_frames : int
                Total number of frames.

            loop : bool, optional (default: False)
                Run callback in an infinite loop.

            fargs : list, optional (default: [])
                Arguments that will be passed to the callback.
            """
            initialized = False
            window_open = True
            while window_open and (loop or not initialized):
                for i in range(n_frames):
                    drawn_artists = callback(i, *fargs)

                    if drawn_artists is None:
                        raise RuntimeError(
                            "The animation function must return a "
                            "sequence of Artist objects.")
                    try:
                        drawn_artists = [a for a in drawn_artists]
                    except TypeError:
                        drawn_artists = [drawn_artists]

                    for a in drawn_artists:
                        for geometry in a.geometries:
                            self.update_geometry(geometry)

                    window_open = self.visualizer.poll_events()
                    if not window_open:
                        break
                    self.visualizer.update_renderer()
                initialized = True

        def view_init(self, azim=-60, elev=30):
            """Set the elevation and azimuth of the axes.

            Parameters
            ----------
            azim : float, optional (default: -60)
                Azimuth angle in the x,y plane in degrees.

            elev : float, optional (default: 30)
                Elevation angle in the z plane.
            """
            vc = self.visualizer.get_view_control()
            pcp = vc.convert_to_pinhole_camera_parameters()
            distance = np.linalg.norm(pcp.extrinsic[:3, 3])
            R_azim_elev_0_world2camera = np.array([
                [0, 1, 0],
                [0, 0, -1],
                [-1, 0, 0]])
            R_azim_elev_0_camera2world = R_azim_elev_0_world2camera.T
            # azimuth and elevation are defined in world frame
            R_azim = pr.active_matrix_from_angle(2, np.deg2rad(azim))
            R_elev = pr.active_matrix_from_angle(1, np.deg2rad(-elev))
            R_elev_azim_camera2world = R_azim.dot(R_elev).dot(
                R_azim_elev_0_camera2world)
            pcp.extrinsic = pt.transform_from(  # world2camera
                R=R_elev_azim_camera2world.T,
                p=[0, 0, distance])
            vc.convert_from_pinhole_camera_parameters(pcp)

        def plot(self, P, c=(0, 0, 0)):
            """Plot line.

            Parameters
            ----------
            P : array-like, shape (n_points, 3)
                Points of which the line consists.

            c : array-like, shape (n_points - 1, 3) or (3,), optional (default: black)
                Color can be given as individual colors per line segment or
                as one color for each segment. A color is represented by 3
                values between 0 and 1 indicate representing red, green, and
                blue respectively.

            Returns
            -------
            Line3D : line
                New line.
            """
            line3d = Line3D(P, c)
            line3d.add_artist(self)
            return line3d

        def plot_basis(self, R=None, p=np.zeros(3), s=1.0, strict_check=True):
            """Plot basis.

            Parameters
            ----------
            R : array-like, shape (3, 3), optional (default: I)
                Rotation matrix, each column contains a basis vector

            p : array-like, shape (3,), optional (default: [0, 0, 0])
                Offset from the origin

            s : float, optional (default: 1)
                Scaling of the frame that will be drawn

            strict_check : bool, optional (default: True)
                Raise a ValueError if the rotation matrix is not numerically
                close enough to a real rotation matrix. Otherwise we print a
                warning.

            Returns
            -------
            Frame : frame
                New frame.
            """
            if R is None:
                R = np.eye(3)
            R = pr.check_matrix(R, strict_check=strict_check)

            frame = Frame(pt.transform_from(R=R, p=p), s=s)
            frame.add_artist(self)

            return frame

        def plot_transform(self, A2B=None, s=1.0, name=None, strict_check=True):
            """Plot coordinate frame.

            Parameters
            ----------
            A2B : array-like, shape (4, 4)
                Transform from frame A to frame B

            s : float, optional (default: 1)
                Length of basis vectors

            label : str, optional (default: None)
                Name of the frame

            strict_check : bool, optional (default: True)
                Raise a ValueError if the transformation matrix is not
                numerically close enough to a real transformation matrix.
                Otherwise we print a warning.

            Returns
            -------
            Frame : frame
                New frame.
            """
            if A2B is None:
                A2B = np.eye(4)
            A2B = pt.check_transform(A2B, strict_check=strict_check)

            frame = Frame(A2B, name, s)
            frame.add_artist(self)

            return frame

        def plot_trajectory(self, P, n_frames=10, s=1.0, c=(0, 0, 0)):
            """Trajectory of poses.

            Parameters
            ----------
            P : array-like, shape (n_steps, 7), optional (default: None)
                Sequence of poses represented by positions and quaternions in
                the order (x, y, z, w, vx, vy, vz) for each step

            n_frames : int, optional (default: 10)
                Number of frames that should be plotted to indicate the
                rotation

            s : float, optional (default: 1)
                Scaling of the frames that will be drawn

            c : array-like, shape (3,), optional (default: black)
                A color is represented by 3 values between 0 and 1 indicate
                representing red, green, and blue respectively.

            Returns
            -------
            trajectory : Trajectory
                New trajectory.
            """
            H = ptr.matrices_from_pos_quat(P)
            trajectory = Trajectory(H, n_frames, s, c)
            trajectory.add_artist(self)
            return trajectory

        def plot_sphere(self, radius=1.0, A2B=np.eye(4), resolution=20, c=None):
            """Plot sphere.

            Parameters
            ----------
            radius : float, optional (default: 1)
                Radius of the sphere

            A2B : array-like, shape (4, 4)
                Transform from frame A to frame B

            resolution : int, optianal (default: 20)
                The resolution of the sphere. The longitues will be split into
                resolution segments (i.e. there are resolution + 1 latitude
                lines including the north and south pole). The latitudes will
                be split into 2 * resolution segments (i.e. there are
                2 * resolution longitude lines.)

            c : array-like, shape (3,), optional (default: None)
                Color

            Returns
            -------
            sphere : Sphere
                New sphere.
            """
            sphere = Sphere(radius, A2B, resolution, c)
            sphere.add_artist(self)
            return sphere

        def plot_box(self, size=np.ones(3), A2B=np.eye(4), c=None):
            """Plot box.

            Parameters
            ----------
            size : array-like, shape (3,), optional (default: [1, 1, 1])
                Size of the box per dimension

            A2B : array-like, shape (4, 4), optional (default: I)
                Center of the box

            c : array-like, shape (3,), optional (default: None)
                Color

            Returns
            -------
            box : Box
                New box.
            """
            box = Box(size, A2B, c)
            box.add_artist(self)
            return box

        def plot_cylinder(self, length=2.0, radius=1.0, A2B=np.eye(4), resolution=20, split=4, c=None):
            """Plot cylinder.

            Parameters
            ----------
            length : float, optional (default: 1)
                Length of the cylinder

            radius : float, optional (default: 1)
                Radius of the cylinder

            A2B : array-like, shape (4, 4)
                Center of the cylinder

            resolution : int, optional (default: 20)
                The circle will be split into resolution segments

            split : int, optional (default: 4)
                The height will be split into split segments

            c : array-like, shape (3,), optional (default: None)
                Color

            Returns
            -------
            cylinder : Cylinder
                New cylinder.
            """
            cylinder = Cylinder(length, radius, A2B, resolution, split, c)
            cylinder.add_artist(self)
            return cylinder

        def plot_mesh(self, filename, A2B=np.eye(4), s=np.ones(3), c=None):
            """Plot mesh.

            Parameters
            ----------
            filename : str
                Path to mesh file

            A2B : array-like, shape (4, 4)
                Center of the mesh

            s : array-like, shape (3,), optional (default: [1, 1, 1])
                Scaling of the mesh that will be drawn

            c : array-like, shape (n_vertices, 3) or (3,), optional (default: None)
                Color(s)

            Returns
            -------
            mesh : Mesh
                New mesh.
            """
            mesh = Mesh(filename, A2B, s, c)
            mesh.add_artist(self)
            return mesh

        def plot_graph(
                self, tm, frame, show_frames=False, show_connections=False,
                show_visuals=False, show_collision_objects=False,
                show_name=False, whitelist=None, s=1.0):
            """Plot graph of connected frames.

            Parameters
            ----------
            tm : TransformManager
                Representation of the graph

            frame : str
                Name of the base frame in which the graph will be displayed

            show_frames : bool, optional (default: False)
                Show coordinate frames

            show_connections : bool, optional (default: False)
                Draw lines between frames of the graph

            show_visuals : bool, optional (default: False)
                Show visuals that are stored in the graph

            show_collision_objects : bool, optional (default: False)
                Show collision objects that are stored in the graph

            show_name : bool, optional (default: False)
                Show names of frames

            whitelist : list, optional (default: all)
                List of frames that should be displayed

            Returns
            -------
            graph : Graph
                New graph.
            """
            graph = Graph(tm, frame, show_frames, show_connections,
                          show_visuals, show_collision_objects, show_name,
                          whitelist, s)
            graph.add_artist(self)
            return graph

        def plot_camera(self, M, cam2world=None, virtual_image_distance=1,
                        sensor_size=(1920, 1080), strict_check=True):
            """Plot camera in world coordinates.

            This function is inspired by Blender's camera visualization. It will
            show the camera center, a virtual image plane, and the top of the virtual
            image plane.

            Parameters
            ----------
            M : array-like, shape (3, 3)
                Intrinsic camera matrix that contains the focal lengths on the diagonal
                and the center of the the image in the last column. It does not matter
                whether values are given in meters or pixels as long as the unit is the
                same as for the sensor size.

            cam2world : array-like, shape (4, 4), optional (default: I)
                Transformation matrix of camera in world frame. We assume that the
                position is given in meters.

            virtual_image_distance : float, optional (default: 1)
                Distance from pinhole to virtual image plane that will be displayed.
                We assume that this distance is given in meters. The unit has to be
                consistent with the unit of the position in cam2world.

            sensor_size : array-like, shape (2,), optional (default: [1920, 1080])
                Size of the image sensor: (width, height). It does not matter whether
                values are given in meters or pixels as long as the unit is the same as
                for the sensor size.

            strict_check : bool, optional (default: True)
                Raise a ValueError if the transformation matrix is not numerically
                close enough to a real transformation matrix. Otherwise we print a
                warning.
            """
            camera = Camera(M, cam2world, virtual_image_distance, sensor_size,
                            strict_check)
            camera.add_artist(self)
            return camera

        def save_image(self, filename):
            """Save rendered image to file.

            Parameters
            ----------
            filename : str
                Path to file in which the rendered image should be stored
            """
            self.visualizer.capture_screen_image(filename, True)

        def show(self):
            """Display the figure window."""
            self.visualizer.run()
            self.visualizer.destroy_window()

    class Artist:
        """Abstract base class for objects that can be rendered."""

        def add_artist(self, figure):
            """Add artist to figure.

            Parameters
            ----------
            figure : Figure
                Figure to which the artist will be added.
            """
            for g in self.geometries:
                figure.add_geometry(g)

        @property
        def geometries(self):
            """Expose geometries.

            Returns
            -------
            geometries : list
                List of geometries that can be added to the visualizer.
            """
            return []

    class Line3D(Artist):
        """A line.

        Parameters
        ----------
        P : array-like, shape (n_points, 3)
            Points of which the line consists.

        c : array-like, shape (n_points - 1, 3) or (3,), optional (default: black)
            Color can be given as individual colors per line segment or as one
            color for each segment. A color is represented by 3 values between
            0 and 1 indicate representing red, green, and blue respectively.
        """

        def __init__(self, P, c=(0, 0, 0)):
            self.line_set = o3d.geometry.LineSet()
            self.set_data(P, c)

        def set_data(self, P, c=None):
            """Update data.

            Parameters
            ----------
            P : array-like, shape (n_points, 3)
                Points of which the line consists.

            c : array-like, shape (n_points - 1, 3) or (3,), optional (default: black)
                Color can be given as individual colors per line segment or
                as one color for each segment. A color is represented by 3
                values between 0 and 1 indicate representing red, green, and
                blue respectively.
            """
            self.line_set.points = o3d.utility.Vector3dVector(P)
            self.line_set.lines = o3d.utility.Vector2iVector(np.hstack((
                np.arange(len(P) - 1)[:, np.newaxis],
                np.arange(1, len(P))[:, np.newaxis])))

            if c is not None:
                try:
                    if len(c[0]) == 3:
                        self.line_set.colors = o3d.utility.Vector3dVector(c)
                except TypeError:  # one color for all segments
                    self.line_set.colors = o3d.utility.Vector3dVector(
                        [c for _ in range(len(P) - 1)])

        @property
        def geometries(self):
            """Expose geometries.

            Returns
            -------
            geometries : list
                List of geometries that can be added to the visualizer.
            """
            return [self.line_set]

    class Frame(Artist):
        """Coordinate frame.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        label : str, optional (default: None)
            Name of the frame

        s : float, optional (default: 1)
            Length of basis vectors
        """

        def __init__(self, A2B, label=None, s=1.0):
            self.A2B = None
            self.label = None
            self.s = s

            self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.s)

            self.set_data(A2B, label)

        def set_data(self, A2B, label=None):
            """Update data.

            Parameters
            ----------
            A2B : array-like, shape (4, 4)
                Transform from frame A to frame B

            label : str, optional (default: None)
                Name of the frame
            """
            previous_A2B = self.A2B
            if previous_A2B is None:
                previous_A2B = np.eye(4)
            self.A2B = A2B
            self.label = label
            if label is not None:
                warnings.warn(
                    "This viewer does not support text. Frame label "
                    "will be ignored.")

            self.frame.transform(
                pt.invert_transform(previous_A2B, check=False))
            self.frame.transform(self.A2B)

        @property
        def geometries(self):
            """Expose geometries.

            Returns
            -------
            geometries : list
                List of geometries that can be added to the visualizer.
            """
            return [self.frame]

    class Trajectory(Artist):
        """Trajectory of poses.

        Parameters
        ----------
        H : array-like, shape (n_steps, 4, 4)
            Sequence of poses represented by homogeneous matrices

        n_frames : int, optional (default: 10)
            Number of frames that should be plotted to indicate the rotation

        s : float, optional (default: 1)
            Scaling of the frames that will be drawn

        c : array-like, shape (3,), optional (default: black)
            A color is represented by 3 values between 0 and 1 indicate
            representing red, green, and blue respectively.
        """

        def __init__(self, H, n_frames=10, s=1.0, c=(0, 0, 0)):
            self.H = H
            self.n_frames = n_frames
            self.s = s
            self.c = c

            self.key_frames = []
            self.line = Line3D(H[:, :3, 3], c)

            self.key_frames_indices = np.linspace(
                0, len(self.H) - 1, self.n_frames, dtype=np.int)
            for i, key_frame_idx in enumerate(self.key_frames_indices):
                self.key_frames.append(Frame(self.H[key_frame_idx], s=self.s))

            self.set_data(H)

        def set_data(self, H):
            """Update data.

            Parameters
            ----------
            H : array-like, shape (n_steps, 4, 4)
                Sequence of poses represented by homogeneous matrices
            """
            self.line.set_data(H[:, :3, 3])
            for i, key_frame_idx in enumerate(self.key_frames_indices):
                self.key_frames[i].set_data(H[key_frame_idx])

        @property
        def geometries(self):
            """Expose geometries.

            Returns
            -------
            geometries : list
                List of geometries that can be added to the visualizer.
            """
            return self.line.geometries + list(
                chain(*[kf.geometries for kf in self.key_frames]))

    class Sphere(Artist):
        """Sphere.

        Parameters
        ----------
        radius : float, optional (default: 1)
            Radius of the sphere

        A2B : array-like, shape (4, 4)
            Center of the sphere

        resolution : int, optianal (default: 20)
            The resolution of the sphere. The longitues will be split into
            resolution segments (i.e. there are resolution + 1 latitude lines
            including the north and south pole). The latitudes will be split
            into 2 * resolution segments (i.e. there are 2 * resolution
            longitude lines.)

        c : array-like, shape (3,), optional (default: None)
            Color
        """

        def __init__(self, radius=1.0, A2B=np.eye(4), resolution=20, c=None):
            self.sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius, resolution)
            if c is not None:
                n_vertices = len(self.sphere.vertices)
                colors = np.zeros((n_vertices, 3))
                colors[:] = c
                self.sphere.vertex_colors = o3d.utility.Vector3dVector(colors)
            self.sphere.compute_vertex_normals()
            self.A2B = None
            self.set_data(A2B)

        def set_data(self, A2B):
            """Update data.

            Parameters
            ----------
            A2B : array-like, shape (4, 4)
                Center of the sphere
            """
            previous_A2B = self.A2B
            if previous_A2B is None:
                previous_A2B = np.eye(4)
            self.A2B = A2B

            self.sphere.transform(
                pt.invert_transform(previous_A2B, check=False))
            self.sphere.transform(self.A2B)

        @property
        def geometries(self):
            """Expose geometries.

            Returns
            -------
            geometries : list
                List of geometries that can be added to the visualizer.
            """
            return [self.sphere]

    class Box(Artist):
        """Box.

        Parameters
        ----------
        size : array-like, shape (3,), optional (default: [1, 1, 1])
            Size of the box per dimension

        A2B : array-like, shape (4, 4), optional (default: I)
            Center of the box

        c : array-like, shape (3,), optional (default: None)
            Color
        """

        def __init__(self, size=np.ones(3), A2B=np.eye(4), c=None):
            self.half_size = np.asarray(size) / 2.0
            width, height, depth = size
            self.box = o3d.geometry.TriangleMesh.create_box(
                width, height, depth)
            if c is not None:
                n_vertices = len(self.box.vertices)
                colors = np.zeros((n_vertices, 3))
                colors[:] = c
                self.box.vertex_colors = o3d.utility.Vector3dVector(colors)
            self.box.compute_vertex_normals()
            self.A2B = None
            self.set_data(A2B)

        def set_data(self, A2B):
            """Update data.

            Parameters
            ----------
            A2B : array-like, shape (4, 4)
                Center of the box
            """
            previous_A2B = self.A2B
            if previous_A2B is None:
                self.box.transform(
                    pt.transform_from(R=np.eye(3), p=-self.half_size))
                previous_A2B = np.eye(4)
            self.A2B = A2B

            self.box.transform(pt.invert_transform(previous_A2B, check=False))
            self.box.transform(self.A2B)

        @property
        def geometries(self):
            """Expose geometries.

            Returns
            -------
            geometries : list
                List of geometries that can be added to the visualizer.
            """
            return [self.box]

    class Cylinder(Artist):
        """Cylinder.

        Parameters
        ----------
        length : float, optional (default: 1)
            Length of the cylinder

        radius : float, optional (default: 1)
            Radius of the cylinder

        A2B : array-like, shape (4, 4)
            Center of the cylinder

        resolution : int, optional (default: 20)
            The circle will be split into resolution segments

        split : int, optional (default: 4)
            The height will be split into split segments

        c : array-like, shape (3,), optional (default: None)
            Color
        """

        def __init__(self, length=2.0, radius=1.0, A2B=np.eye(4), resolution=20, split=4, c=None):
            self.cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius, height=length, resolution=resolution,
                split=split)
            if c is not None:
                n_vertices = len(self.cylinder.vertices)
                colors = np.zeros((n_vertices, 3))
                colors[:] = c
                self.cylinder.vertex_colors = \
                    o3d.utility.Vector3dVector(colors)
            self.cylinder.compute_vertex_normals()
            self.A2B = None
            self.set_data(A2B)

        def set_data(self, A2B):
            """Update data.

            Parameters
            ----------
            A2B : array-like, shape (4, 4)
                Center of the cylinder
            """
            previous_A2B = self.A2B
            if previous_A2B is None:
                previous_A2B = np.eye(4)
            self.A2B = A2B

            self.cylinder.transform(
                pt.invert_transform(previous_A2B, check=False))
            self.cylinder.transform(self.A2B)

        @property
        def geometries(self):
            """Expose geometries.

            Returns
            -------
            geometries : list
                List of geometries that can be added to the visualizer.
            """
            return [self.cylinder]

    class Mesh(Artist):
        """Mesh.

        Parameters
        ----------
        filename : str
            Path to mesh file

        A2B : array-like, shape (4, 4)
            Center of the mesh

        s : array-like, shape (3,), optional (default: [1, 1, 1])
            Scaling of the mesh that will be drawn

        c : array-like, shape (n_vertices, 3) or (3,), optional (default: None)
            Color(s)
        """

        def __init__(self, filename, A2B=np.eye(4), s=np.ones(3), c=None):
            self.mesh = o3d.io.read_triangle_mesh(filename)
            self.mesh.vertices = o3d.utility.Vector3dVector(
                np.asarray(self.mesh.vertices) * s)
            self.mesh.compute_vertex_normals()
            if c is not None:
                n_vertices = len(self.mesh.vertices)
                colors = np.zeros((n_vertices, 3))
                colors[:] = c
                self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            self.A2B = None
            self.set_data(A2B)

        def set_data(self, A2B):
            """Update data.

            Parameters
            ----------
            A2B : array-like, shape (4, 4)
                Center of the mesh
            """
            previous_A2B = self.A2B
            if previous_A2B is None:
                previous_A2B = np.eye(4)
            self.A2B = A2B

            self.mesh.transform(pt.invert_transform(previous_A2B, check=False))
            self.mesh.transform(self.A2B)

        @property
        def geometries(self):
            """Expose geometries.

            Returns
            -------
            geometries : list
                List of geometries that can be added to the visualizer.
            """
            return [self.mesh]

    class Camera(Artist):
        """Camera.

        Parameters
        ----------
        M : array-like, shape (3, 3)
            Intrinsic camera matrix that contains the focal lengths on the diagonal
            and the center of the the image in the last column. It does not matter
            whether values are given in meters or pixels as long as the unit is the
            same as for the sensor size.

        cam2world : array-like, shape (4, 4), optional (default: I)
            Transformation matrix of camera in world frame. We assume that the
            position is given in meters.

        virtual_image_distance : float, optional (default: 1)
            Distance from pinhole to virtual image plane that will be displayed.
            We assume that this distance is given in meters. The unit has to be
            consistent with the unit of the position in cam2world.

        sensor_size : array-like, shape (2,), optional (default: [1920, 1080])
            Size of the image sensor: (width, height). It does not matter whether
            values are given in meters or pixels as long as the unit is the same as
            for the sensor size.

        strict_check : bool, optional (default: True)
            Raise a ValueError if the transformation matrix is not numerically
            close enough to a real transformation matrix. Otherwise we print a
            warning.
        """

        def __init__(self, M, cam2world=None, virtual_image_distance=1,
                     sensor_size=(1920, 1080), strict_check=True):
            self.M = None
            self.cam2world = None
            self.virtual_image_distance = None
            self.sensor_size = None
            self.strict_check = strict_check

            self.line_set = o3d.geometry.LineSet()

            if cam2world is None:
                cam2world = np.eye(4)

            self.set_data(M, cam2world, virtual_image_distance, sensor_size)

        def set_data(self, M=None, cam2world=None, virtual_image_distance=None,
                     sensor_size=None):
            """Update camera parameters.

            Parameters
            ----------
            M : array-like, shape (3, 3), optional (default: old value)
                Intrinsic camera matrix that contains the focal lengths on the
                diagonal and the center of the the image in the last column. It
                does not matter whether values are given in meters or pixels as
                long as the unit is the same as for the sensor size.

            cam2world : array-like, shape (4, 4), optional (default: old value)
                Transformation matrix of camera in world frame. We assume that
                the position is given in meters.

            virtual_image_distance : float, optional (default: old value)
                Distance from pinhole to virtual image plane that will be
                displayed. We assume that this distance is given in meters.
                The unit has to be consistent with the unit of the position
                in cam2world.

            sensor_size : array-like, shape (2,), optional (default: old value)
                Size of the image sensor: (width, height). It does not matter
                whether values are given in meters or pixels as long as the
                unit is the same as for the sensor size.
            """
            if M is not None:
                self.M = M
            if cam2world is not None:
                self.cam2world = pt.check_transform(
                    cam2world, strict_check=self.strict_check)
            if virtual_image_distance is not None:
                self.virtual_image_distance = virtual_image_distance
            if sensor_size is not None:
                self.sensor_size = sensor_size

            camera_center_in_cam = np.zeros(3)
            camera_center_in_world = pt.transform(
                cam2world, pt.vector_to_point(camera_center_in_cam))
            focal_length = np.mean(np.diag(M[:2, :2]))
            sensor_corners_in_cam = np.array([
                [-M[0, 2], -M[1, 2], focal_length],
                [-M[0, 2], sensor_size[1] - M[1, 2], focal_length],
                [sensor_size[0] - M[0, 2], sensor_size[1] - M[1, 2],
                 focal_length],
                [sensor_size[0] - M[0, 2], -M[1, 2], focal_length],
            ])
            sensor_corners_in_world = pt.transform(
                cam2world, pt.vectors_to_points(sensor_corners_in_cam))[:, :3]
            virtual_image_corners = (
                sensor_corners_in_world -
                camera_center_in_world[np.newaxis, :3])
            virtual_image_corners = (
                virtual_image_distance / focal_length *
                virtual_image_corners +
                camera_center_in_world[np.newaxis, :3])

            up = virtual_image_corners[0] - virtual_image_corners[1]
            camera_line_points = np.vstack((
                camera_center_in_world[:3],
                virtual_image_corners[0],
                virtual_image_corners[1],
                virtual_image_corners[2],
                virtual_image_corners[3],
                virtual_image_corners[0] + 0.1 * up,
                0.5 * (virtual_image_corners[0] +
                       virtual_image_corners[3]) + 0.5 * up,
                virtual_image_corners[3] + 0.1 * up
            ))

            self.line_set.points = o3d.utility.Vector3dVector(
                camera_line_points)
            self.line_set.lines = o3d.utility.Vector2iVector(
                np.array([[0, 1], [0, 2], [0, 3], [0, 4],
                          [1, 2], [2, 3], [3, 4], [4, 1],
                          [5, 6], [6, 7], [7, 5]]))

        @property
        def geometries(self):
            """Expose geometries.

            Returns
            -------
            geometries : list
                List of geometries that can be added to the visualizer.
            """
            return [self.line_set]

except ImportError:
    warnings.warn("3D visualizer is not available. Install open3d.")
