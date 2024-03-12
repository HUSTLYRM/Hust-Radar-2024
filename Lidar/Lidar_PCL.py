import threading
import pcl
from queue import Queue
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np

class PointCloudVisualizer:
    def __init__(self):
        rospy.init_node('open3d_visualize_node', anonymous=True)
        self.queue = Queue()
        self.lock = threading.Lock()
        self.pcl_visualizer = pcl.pcl_visualization.CloudViewing()

    def callback(self, msg):
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        point_cloud = np.array(list(gen))
        with self.lock:
            self.queue.put(point_cloud)

    def receive_and_process_data(self):
        rospy.Subscriber('livox/lidar', PointCloud2, self.callback)
        rospy.spin()

    def display_data(self):
        while True:
            if not self.queue.empty():
                with self.lock:
                    point_cloud = self.queue.get()
                pcl_cloud = pcl.PointCloud()
                pcl_cloud.from_array(point_cloud.astype(np.float32))
                self.pcl_visualizer.ShowMonochromeCloud(pcl_cloud)

    def start(self):
        threading.Thread(target=self.receive_and_process_data).start()
        threading.Thread(target=self.display_data).start()


if __name__ == '__main__':
    visualizer = PointCloudVisualizer()
    visualizer.start()