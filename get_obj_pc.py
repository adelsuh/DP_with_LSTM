import open3d as o3d
import numpy as np

with_obj = o3d.io.read_point_cloud("./with_obj.ply")
without_obj = o3d.io.read_point_cloud("./without_obj.ply")
dists = with_obj.compute_point_cloud_distance(without_obj)
dists = np.asarray(dists)
ind = np.where(dists > 7)[0]
pcd_obj = with_obj.select_by_index(ind)
o3d.visualization.draw_geometries([pcd_obj])
o3d.io.write_point_cloud("pcd_obj_with_outliers.ply", pcd_obj)

cl, ind = pcd_obj.remove_radius_outlier(nb_points=4, radius=10)
pcd_obj = pcd_obj.select_by_index(ind)
o3d.visualization.draw_geometries([pcd_obj])
o3d.io.write_point_cloud("pcd_obj.ply", pcd_obj)

