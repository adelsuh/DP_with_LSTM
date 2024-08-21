import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("./data.ply")
plane_model, inliers = pcd.segment_plane(distance_threshold=7, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model
print(a, b, c, d)
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color((0, 1.0, 0))
outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color((0.0, 0, 1.0))
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])