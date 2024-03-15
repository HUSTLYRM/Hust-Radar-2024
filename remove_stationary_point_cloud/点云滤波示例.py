import open3d as o3d

# 读取点云
pcd = o3d.io.read_point_cloud("path_to_your_point_cloud.pcd")

# 体素网格滤波
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)

# 统计离群值去除
cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
inlier_cloud = voxel_down_pcd.select_by_index(ind)

# 半径离群值去除
cl, ind = inlier_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
final_cloud = inlier_cloud.select_by_index(ind)

# 保存滤波后的点云
o3d.io.write_point_cloud("filtered_point_cloud.pcd", final_cloud)