import cv2
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from collections import deque

# 定义全局变量
show_pcd = True
show_depth = False
show_all_in_one = False
if show_all_in_one:
    show_pcd = False
    show_depth = False
global first
first = True

# 创建深度图建造器
class DepthMapGenerator:
    def __init__(self, width=1280, height=640):
        self.width = width
        self.height = height

    def generate_depth_map(self, pcd):
        # 将点云转换为 numpy 数组
        point_cloud = np.asarray(pcd.points)

        # 计算每个点到相机原点的距离
        depth = np.sqrt(np.sum(point_cloud**2, axis=1))

        # 将深度值转换为一个 1280x640 的图像
        depth_img = cv2.resize(depth, (self.width, self.height))

        depth_img = depth_img.astype(np.float32)

        # 将深度图归一化到0-1
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())

        return depth_img

# 创建点云队列
class PcdQueue(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)

    def add(self, pcd):
        self.queue.append(pcd)

    def get_all(self):
        return list(self.queue)

    def get_depth_img(self):
        # 获取深度图
        depth_img = np.zeros((self.max_size, 1024))
        for i, pcd in enumerate(self.queue):
            # 将点云转换为深度图
            depth = np.array(pcd.points)[:, 2]
            depth_img[i, :] = depth
        return depth_img

def main():
    # 创建可视化对象
    if show_pcd:
        pcd_vis = o3d.visualization.Visualizer()
        pcd_vis.create_window()
        view_control = pcd_vis.get_view_control()
    if show_depth:
        dep_vis = o3d.visualization.Visualizer()
        dep_vis.create_window()
        view_control = dep_vis.get_view_control()
    if show_all_in_one:
        dep_vis = o3d.visualization.Visualizer()
        dep_vis.create_window()
        view_control = dep_vis.get_view_control()

    view_control.set_front([1, 1, 1])


    # 创建点云队列
    pcd_queue = PcdQueue(max_size=10)

    def callback(msg):
        global first
        # 将 ROS PointCloud2 消息转换为表示每个点的生成器对象
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

        # 将点生成器转换为 numpy 数组，并将其转换为 Open3D 点云
        point_cloud = np.array(list(gen))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        # 设置视图控制看向点云的中心
        pcd_center = pcd.get_center()
        print(pcd_center)
        view_control.set_lookat(pcd_center)

        # 检查新接收到的点云是否已经存在于累积的点云中
        if not np.any([np.array_equal(pcd.points, old_pcd.points) for old_pcd in pcd_queue.get_all()]):
            # 如果不存在，那么就将其添加到累积的点云中
            pcd_queue.add(pcd)

        # 从队列中获取所有的点云
        all_pcds = pcd_queue.get_all()


        # 创建一个新的点云，然后将所有的点云添加到这个新的点云中
        merged_pcd = o3d.geometry.PointCloud()
        for pcd in all_pcds:
            merged_pcd += pcd

        # print(len(merged_pcd))
        # 查看点的数量
        # print(len(merged_pcd.points))
        # 打印点云查看格式
        print(merged_pcd.points)

        # 使用 DepthMapGenerator 生成深度图
        depth_map_generator = DepthMapGenerator()
        depth_img = depth_map_generator.generate_depth_map(merged_pcd)

        # 将深度图转换为 Open3D 图像，并添加到可视化窗口中
        if show_pcd:
            if first:
                pcd_vis.add_geometry(merged_pcd)
                first = False
                pcd_vis.poll_events()
                pcd_vis.update_renderer()
            else:
                pcd_vis.update_geometry(merged_pcd)
                pcd_vis.poll_events()
                pcd_vis.update_renderer()
            print(first)

        if show_depth:
            depth_img_o3d = o3d.geometry.Image(depth_img)
            dep_vis.add_geometry(depth_img_o3d)
            dep_vis.poll_events()
            dep_vis.update_renderer()

        if show_all_in_one:
            depth_img_o3d = o3d.geometry.Image(depth_img)
            dep_vis.add_geometry(depth_img_o3d)
            dep_vis.add_geometry(merged_pcd)
            dep_vis.poll_events()
            dep_vis.update_renderer()

    rospy.init_node('open3d_visualize_node', anonymous=True)
    rospy.Subscriber('livox/lidar', PointCloud2, callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    dep_vis.destroy_window()

if __name__ == '__main__':
    main()