# 建立ros节点，订阅/hikrobot_camera/rgb，获取rgb图像，存到图像队列中
# 话题的内容是sensor_msgs/Image

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge , CvBridgeError
import cv2
import time
import threading
import numpy as np
from collections import deque
from Tools.Tools import Tools

class CameraNode:
    def __init__(self):
        # 标志
        self.init_flag = False # 节点初始化
        self.working_flag = False # 开始接收数据

        # 线程
        self.threading = None

        # 图像队列
        self.image_queue = deque(maxlen=10)

        # 话题名称
        self.camera_topic_name = "/image" # hik_camera/image # /hikrobot_camera/rgb

        # bridge
        self.bridge = CvBridge()

        # 录制
        self.out = cv2.VideoWriter(f'/home/nvidia/RadarWorkspace/data/{Tools.get_time_stamp()}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (4024, 3036))

        if not self.init_flag:
            # 当还未有一个节点时，初始化一个节点
            self.listener_begin(self.camera_topic_name)
            # print("listener_begin")
            self.init_flag = True
            self.threading = threading.Thread(target=self.main_loop, daemon=True)

    # 节点启动
    def listener_begin(self, camera_topic_name):
        rospy.init_node('camera_listener', anonymous=True)
        # print("init ")
        rospy.Subscriber(camera_topic_name, Image, self.callback)
        # print("sub")

    def callback(self, msg):
        print("callback")


        # print(dir(msg))
        # print(msg.data)
        if not self.working_flag:
            print("not working")
            return
        try:
            # 使用CvBridge进行图像转换
            print("try")
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print(type(cv_image))
            print("after convert")
            # cv_image = cv2.resize(cv_image, (1920, 1080))
            self.out.write(cv_image)
            cv2.imshow("camera", cv_image)
            cv2.waitKey(1)

            # 如果需要将其转为BGR用于OpenCV处理
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            print("Converted to BGR")

        except CvBridgeError as e:
            print("error")
            print("CvBridgeError: ", e)
            return
        self.image_queue.put(cv_image)
        print(self.image_queue.qsize())


    def main_loop(self):
        print("spin")
        rospy.spin()

    # 线程启动
    def start(self):
        if not self.init_flag:
            print("init failed")
            return

        if not self.working_flag and not self.threading.is_alive():
            self.working_flag = True
            self.threading.start()

    # 线程停止
    def stop(self):
        if self.working_flag and self.threading is not None:
            self.working_flag = False
            rospy.signal_shutdown("stop camera node")
            # self.threading.join()

    # del
    def __del__(self):
        self.stop()


# 测试
if __name__ == '__main__':
    camera_node = CameraNode()
    camera_node.start()
    time.sleep(60)
    camera_node.stop()
