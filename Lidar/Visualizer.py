# 定义一个visualizer，用于可视化点云数据，深度图等

import numpy as np
import open3d as o3d
import cv2
import yaml

class Visualizer:
    def __init__(self, data_loader_path = 'parameters.yaml' , update = True):
        # 传入data_loader路径,用data_loader初始化类
        with open(data_loader_path, 'r') as file:
            data_loader = yaml.safe_load(file)
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window()
        self.pcd = o3d.geometry.PointCloud()
        self.update = update # 使用的是update_geometry还是add_geometry
        self.first = True # 是否是第一次添加几何体

    def set_visualizer_pcd(self, geometry):
        self.visualizer.add_geometry(geometry)

    # 更新几何体
    def update_visualizer(self):
        # 自己调用update_geometry
        self.visualizer.update_geometry(self.pcd)

