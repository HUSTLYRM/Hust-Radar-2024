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

    # 首次添加几何体
    def set_visualizer_pcd(self, pcd):
        # 如果geometry非空，则添加几何体
        if self.is_pcd_empty(pcd):
            self.visualizer.add_geometry(pcd)
            self.first = False # 如果添加了pcd，则已经有pcd了不为空
        else:
            print("pcd is empty")

    # 更新几何体
    def update_geometry(self):
        # 自己调用update_geometry
        if self.first:
            print("no pcd")
            return
        self.visualizer.update_geometry(self.pcd)

    def update_pcd_points(self, pc): # 把可视化对象的pcd的points属性修改为pc
        self.pcd.points = o3d.utility.Vector3dVector(pc)

    # 每一次可视化的流程，修改vis里的geometry的points属性，然后update_geometry，然后vis.poll_events()，然后vis.update_renderer()

    # 更新visualizer
    def update_visualizer(self):
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

    # 整体update，直接调用这个
    def update(self, pc): # 把新的点云数据传入，更新可视化
        self.update_pcd_points(pc)
        self.update_geometry()
        self.update_visualizer()

    # 判断geometry是否为空
    def is_pcd_empty(self):
        if len(self.pcd.points) == 0:
            return True
        else:
            return False


