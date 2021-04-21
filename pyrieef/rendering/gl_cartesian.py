import open3d as o3d

print("Let's define some primitives")
mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,
                                                height=1.0,
                                                depth=1.0)
mesh_box.compute_vertex_normals()
mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3,
                                                          height=4.0)
mesh_cylinder.compute_vertex_normals()
mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[-2, -2, -2])

print("We draw a few primitives using collection.")
o3d.visualization.draw_geometries(
    [mesh_box, mesh_sphere, mesh_cylinder, mesh_frame])

print("We draw a few primitives using + operator of mesh.")
o3d.visualization.draw_geometries(
    [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])