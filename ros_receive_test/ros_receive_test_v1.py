import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np

# 创建可视化对象
vis = o3d.visualization.Visualizer()
print("1")
vis.create_window()
print("2")
pcd = o3d.geometry.PointCloud()
print("3")


def callback(msg):
    # Convert ROS PointCloud2 message to a generator object representing each point
    gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    print("7")

    # Convert point generator to numpy array and transform it into an Open3D point cloud
    point_cloud = np.array(list(gen))
    print("8")
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    print("9")

    # Update Point Cloud
    vis.update_geometry(pcd)
    print("11")
    vis.poll_events()
    print("12")
    vis.update_renderer()
    print("13")


rospy.init_node('open3d_visualize_node', anonymous=True)
print("4")
rospy.Subscriber('livox/lidar', PointCloud2, callback)
print("5")

try:
    rospy.spin()
    print("6")
except KeyboardInterrupt:
    print("Shutting down")

# Remove the cloud from the visualizer and close the visualizer's window
vis.remove_geometry(pcd)
vis.destroy_window()
