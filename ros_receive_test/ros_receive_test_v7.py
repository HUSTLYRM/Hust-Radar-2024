import cv2
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from collections import deque


class depth:
    def __init__(self, fx, fy, cx, cy, EPS, MAX_DEPTH, CAM_WID, CAM_HGT):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.EPS = EPS
        self.MAX_DEPTH = MAX_DEPTH
        self.CAM_WID = CAM_WID
        self.CAM_HGT = CAM_HGT

    def pcd_to_depth(self, pcd):
        pc = np.asarray(pcd.points)
        z = np.copy(pc[:, 0])
        y = -np.copy(pc[:, 2])
        x = -np.copy(pc[:, 1])
        pc[:, 0] = x
        pc[:, 1] = y
        pc[:, 2] = z
        valid = pc[:, 2] > self.EPS
        z = pc[valid, 2]
        u = np.round(pc[valid, 0] * self.fx / z + self.cx).astype(int)
        v = np.round(pc[valid, 1] * self.fy / z + self.cy).astype(int)
        valid = np.bitwise_and(np.bitwise_and((u >= 0), (u < self.CAM_WID)),
                               np.bitwise_and((v >= 0), (v < self.CAM_HGT)))
        u, v, z = u[valid], v[valid], z[valid]
        img_z = np.full((self.CAM_HGT, self.CAM_WID), np.inf)
        for ui, vi, zi in zip(u, v, z):
            img_z[vi, ui] = min(img_z[vi, ui], zi)
        img_z_shift = np.array([img_z, \
                                np.roll(img_z, 1, axis=0), \
                                np.roll(img_z, -1, axis=0), \
                                np.roll(img_z, 1, axis=1), \
                                np.roll(img_z, -1, axis=1)])
        img_z = np.min(img_z_shift, axis=0)
        img_z = np.where(img_z > self.MAX_DEPTH, self.MAX_DEPTH, img_z)
        img_z = cv2.normalize(img_z, None, 0, 2550, cv2.NORM_MINMAX, cv2.CV_8U)
        return img_z

# 创建点云队列
class PcdQueue(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)

    def add(self, pcd):
        self.queue.append(pcd)

    def get_all(self):
        return list(self.queue)

    # 获得队列中点的数量，而非队列的大小
    def point_num(self):
        num = 0
        for pcd in self.queue:
            num += len(pcd.points)
        return num




def main():
    # set param
    fx = 1269.8676
    fy = 1276.0659
    cx = 646.6841
    cy = 248.7859

    CAM_WID, CAM_HGT = 1280, 640  # 重投影到的深度图尺寸
    CAM_FX, CAM_FY = fx, fy  # fx/fy
    CAM_CX, CAM_CY = cx, cy  # cx/cy

    EPS = 1.0e-16
    MAX_DEPTH = 50.0  # 最大深度值
    # 创建深度图对象
    d = depth(fx, fy, cx, cy, EPS, MAX_DEPTH, CAM_WID, CAM_HGT)
    # 创建可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    view_control = vis.get_view_control()
    # view_control.set_front([0, 0, -1])


    # 创建点云队列
    pcd_queue = PcdQueue(max_size=50)

    def callback(msg):
        # 将 ROS PointCloud2 消息转换为表示每个点的生成器对象
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        # 总点云
        pcd_merge = o3d.geometry.PointCloud()

        # 将点生成器转换为 numpy 数组，并将其转换为 Open3D 点云
        point_cloud = np.array(list(gen))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pc = np.asarray(pcd.points)
        # print(pc)

        # 添加点云到队列
        pcd_queue.add(pcd)

        # 清除所有的几何体
        vis.clear_geometries()

        # 把所有点云放入一个merge中


        # 添加所有的点云到可视化窗口
        for pcd in pcd_queue.get_all():
            vis.add_geometry(pcd)
            # 把所有点云放入一个merge中
            pcd_merge += pcd

        img_z = d.pcd_to_depth(pcd_merge)
        # print('1')
        cv2.imshow('img_z', img_z)
        cv2.waitKey(1)
        # print('2')


        # print(pcd_queue.point_num())

        vis.poll_events()
        vis.update_renderer() # 重新渲染

    rospy.init_node('open3d_visualize_node', anonymous=True)
    rospy.Subscriber('livox/lidar', PointCloud2, callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    vis.destroy_window()

if __name__ == '__main__':
    main()