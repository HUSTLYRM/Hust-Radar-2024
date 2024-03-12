import cv2
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np

def main():
    # 创建可视化对象
    vis = o3d.visualization.Visualizer()
    #vis.create_window()
    pcd = o3d.geometry.PointCloud()
    # vis.add_geometry(pcd)
    vis.create_window()
    # vis.add_geometry(pcd)
    global first
    first = True

    def callback(msg):
        global first
        # 将 ROS PointCloud2 消息转换为表示每个点的生成器对象
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

        # 将点生成器转换为 numpy 数组，并将其转换为 Open3D 点云
        point_cloud = np.array(list(gen))
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # 更新点云
        # vis.create_window()
        if first:
            vis.add_geometry(pcd)
            first = False
        else:
            vis.update_geometry(pcd)
        vis.poll_events()
        # print("vis")
        vis.update_renderer() # 重新渲染
        # 等待一段时间，然后关闭窗口
        # vis.destroy_window()

    rospy.init_node('open3d_visualize_node', anonymous=True)
    rospy.Subscriber('livox/lidar', PointCloud2, callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    # 从可视化器中移除云，并关闭可视化器的窗口
    vis.remove_geometry(pcd)
    vis.destroy_window()

if __name__ == '__main__':
    main()