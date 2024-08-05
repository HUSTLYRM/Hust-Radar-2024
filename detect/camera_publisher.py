import time

import rospy
from sensor_msgs.msg import Image
from Capture import Capture
from cv_bridge import CvBridge
import multiprocessing
import cv2




def main_loop():
    # 初始化ROS节点
    rospy.init_node('image_publisher')

    # 初始化capture
    capture = Capture()

    first_img = capture.get_frame()
    # cv2.imshow("camera", first_img)

    # 创建一个发布器
    pub = rospy.Publisher('/image', Image, queue_size=10)

    # 创建一个CvBridge对象
    bridge = CvBridge()

    # 无限循环中捕获图像
    while not rospy.is_shutdown():
        # 获取图像
        img = capture.get_frame()
        # print(type(img))
        # img = cv2.resize(img , (1920,1080))
        # cv2.imshow("camera", img)

        # 将img从numpy.ndarray转换为OpenCV图像
        # img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # 将OpenCV图像转换为ROS图像消息
        img_msg = bridge.cv2_to_imgmsg(img, "bgr8")
        pub.publish(img_msg)


# 创建一个新的进程来运行capture方法
p = multiprocessing.Process(target=main_loop)
p.start()
# capture = Capture()
# while True:
#     first_img = capture.get_frame()
#     cv2.imshow("camera", first_img)
#     cv2.waitKey(1)
# time.sleep(10)


# 等待capture方法的进程结束
p.join()