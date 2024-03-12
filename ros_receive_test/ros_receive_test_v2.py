import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import threading

# 创建可视化对象
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()

# 创建一个锁对象和一个全局变量来保存点云数据
lock = threading.Lock()
global_point_cloud = None

def callback(msg):
    # Convert ROS PointCloud2 message to a generator object representing each point
    gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

    # Convert point generator to numpy array
    point_cloud = np.array(list(gen))

    # Update the global point cloud data
    with lock:
        global global_point_cloud
        global_point_cloud = point_cloud

rospy.init_node('open3d_visualize_node', anonymous=True)
rospy.Subscriber('livox/lidar', PointCloud2, callback)

try:
    while not rospy.is_shutdown():
        # Get the global point cloud data
        with lock:
            point_cloud = global_point_cloud

        # If there is new point cloud data, update the point cloud object and the visualizer
        if point_cloud is not None:
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

except KeyboardInterrupt:
    print("Shutting down")

# Remove the cloud from the visualizer and close the visualizer's window
vis.remove_geometry(pcd)
vis.destroy_window()