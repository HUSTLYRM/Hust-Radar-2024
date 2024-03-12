import cv2
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from collections import deque

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
    # 创建可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])

    # 创建点云队列
    pcd_queue = PcdQueue(max_size=5)

    def callback(msg):
        # 将 ROS PointCloud2 消息转换为表示每个点的生成器对象
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

        # 将点生成器转换为 numpy 数组，并将其转换为 Open3D 点云
        point_cloud = np.array(list(gen))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pc = np.asarray(pcd.points)
        print(pc.shape)

        # 添加点云到队列
        pcd_queue.add(pcd)

        # 清除所有的几何体
        vis.clear_geometries()

        # 添加所有的点云到可视化窗口
        for pcd in pcd_queue.get_all():
            vis.add_geometry(pcd)


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