import open3d as o3d
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
pcd = o3d.io.read_point_cloud('../pcd_data/rabbit.pcd')




# 点云移动
pc = np.asarray(pcd.points)
pc1 = pc.copy()

# 创建坐标轴
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

# 添加随机噪点点云,x,y,z均在0-10之间
noise = o3d.geometry.PointCloud()
noise.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3) * 10)



# 不是获取引用，而是获取副本
x = pc[:, 0]
y = pc[:, 1]
z = pc[:, 2]
# 将所有的点云沿x轴移动1个单位
x += 10
y += 1
z += 1
pc1[:, 0] = x
pc1[:, 1] = y
pc1[:, 2] = z
# 还原一个移动后的点云
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(pc1)

vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(pcd1) # 移动过的点云
vis.add_geometry(coordinate_frame) # 添加坐标轴
vis.add_geometry(noise) # 添加噪点
# vis.add_geometry(inlier_cloud) # 添加平面点云
vis.run()
vis.update_renderer()
vis.destroy_window()
