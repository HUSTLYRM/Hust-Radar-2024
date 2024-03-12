import cv2
import PIL
import os
import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np


# set param
fx = 1269.8676

fy = 1276.0659

cx = 646.6841

cy = 248.7859

# 相机内参矩阵


# pycharm如果不设置工作目录会变为项目根目录，需要设置，或者使用绝对路径

# 创建一个只包含原点的点云

origin = o3d.geometry.PointCloud()
origin.points = o3d.utility.Vector3dVector([[0, 0, 0]])  # 坐标原点

# 设置原点的颜色和大小
origin.paint_uniform_color([1, 0, 0])  # 红色
origin.points = o3d.utility.Vector3dVector(np.array(origin.points)*4)  # 大小为2

# 添加坐标轴
# 创建坐标轴
axis_length = 5  # 坐标轴长度
axis = o3d.geometry.LineSet()
axis.points = o3d.utility.Vector3dVector([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
axis.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
axis.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # X轴红色，Y轴绿色，Z轴蓝色

# 坐标轴 红色为x，绿色为y，蓝色为z
#激光雷达看的正方向，向前为x，向左为y，向上为z
# 图片读取
# img = cv2.imread('/home/nvidia/RadarWorkspace/code/ros_receive_test/pcd_data/0.png')
# img1 = cv2.imread('../../pcd_data/0.png')
# /home/nvidia/RadarWorkspace/code/ros_receive_test/pcd_data
# cv2.imshow('img', img)
# cv2.waitKey(0)

# pcd文件读取
pcd = o3d.io.read_point_cloud('/home/nvidia/RadarWorkspace/pcd/calib2.pcd')
vis= o3d.visualization.Visualizer()
vis.create_window('pcl',1920, 1080, 50, 50, True)
vis.add_geometry(pcd) # 添加点云
vis.add_geometry(origin) # 添加坐标原点
vis.add_geometry(axis) # 添加坐标轴
vis.run()
vis.update_renderer()

# vis = o3d.visualization.Visualizer()
# vis.create_window(visible = True)
# vis.add_geometry(pcd)

# depth = vis.capture_depth_float_buffer(False)

# image = vis.capture_screen_float_buffer(False)
# plt.imshow(np.asarray(depth))
# o3d.io.write_image("./test_depth.png", depth)


