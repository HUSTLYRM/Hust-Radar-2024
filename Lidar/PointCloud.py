import cv2
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from collections import deque

# 定义自己的点云处理类
class Pcd:
    def __init__(self , pcd_name = ""):
        self.pcd = o3d.geometry.PointCloud()
        self.pcd_name = pcd_name

    # 将本Pcd的pcd属性设置为传入的pcd，修改引用
    def set_pcd(self, pcd):
        self.pcd = pcd


    # 更新pcd属性的points
    def update_pcd_points(self, pc):
        self.pcd.points = o3d.utility.Vector3dVector(pc)

    #
class PcdQueue(object):
    def __init__(self, max_size,voxel_size = 0.05):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)
        # 创建一个空的voxel
        self.voxel = o3d.geometry.VoxelGrid()
        self.voxel_size = voxel_size
        self.pcd_all = o3d.geometry.PointCloud()

    # 目前是每次添加都会把所有点云转为voxel，然后更新voxel
    def add(self, pcd):
        self.queue.append(pcd)
        self.pcd_all = self.get_all_pcd()
        self.update_voxel(self.voxel_size)

    def get_all(self):
        return list(self.queue)

    # 获得队列中点的数量，而非队列的大小
    def point_num(self):
        num = 0
        for pcd in self.queue:
            num += len(pcd.points)
        return num

    # 获得队列中的所有点云，以o3d的geometry的PointCloud的形式
    def get_all_pcd(self):
        pcd_all = o3d.geometry.PointCloud()
        for pcd in self.queue:
            pcd_all += pcd
        return pcd_all

    # 将队列中的点云转为voxel格式
    def update_voxel(self,voxel_size = 0.05):
        self.voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd_all, voxel_size=voxel_size)


    # 获取一份新的可处理的voxel
    def get_voxel_copy(self):
        voxel = o3d.geometry.VoxelGrid(self.voxel)
        return voxel

    # 获取只读voxel
    def get_voxel(self):
        return self.voxel

    # 聚类并返回最大簇的点云和中心点坐标
    def cluster(self,pcd):
        # 使用DBSCAN进行聚类
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=0.40, min_points=10, print_progress=True))

        # 如果没有找到任何簇，返回一个空的点云和中心点
        if labels.max() == -1:
            return np.array([]), np.array([0, 0, 0])


        # 计算每个簇的大小
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        cluster_sizes = [len(np.where(labels == i)[0]) for i in range(max_label + 1)]

        # 找到最大簇的索引
        max_cluster_idx = np.argmax(cluster_sizes)

        # 找到最大簇的所有点
        max_cluster_points = np.asarray(pcd.points)[np.where(labels == max_cluster_idx)]

        # 计算最大簇的中心点
        centroid = max_cluster_points.mean(axis=0)

        return max_cluster_points, centroid

    # 获取点云的距离中值的点,返回的是一个numpy数组
    def get_median_point(self,pcd):
        # 计算每个点到原点的距离
        distance = np.sqrt(np.sum(np.asarray(pcd.points) ** 2, axis=1))

        # 取中值点
        median_distance = np.median(distance)
        # print("测据:", median_distance)
        # 取中值点的索引
        median_index = np.where(distance == median_distance)
        # 取中值点
        median_point = np.asarray(pcd.points)[median_index]

        return median_point

    # 传入一个点（numpy数组），和一个点云，返回一个高亮了这个点的点云
    def highlight_point(self,point,pcd):
        # 将点转换为numpy数组
        points = np.asarray(pcd.points)

        # 添加新的点
        points = np.vstack([points, point])

        # 转换回Vector3dVector并赋值给show_pcd
        pcd.points = o3d.utility.Vector3dVector(points)
        # 创建一个颜色数组，对应于show_pcd中的每个点，将所有的点设置为绿色
        colors = np.ones((len(pcd.points), 3)) * [0, 1, 0]  # 所有点默认为绿色

        # 将最后一个颜色设置为红色
        colors[-1] = [1, 0, 0]  # 最后一个点为红色

        # 设置点云的颜色
        pcd.colors = o3d.utility.Vector3dVector(colors)

