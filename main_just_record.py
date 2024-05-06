import rosbag
import rospy
import cv2
from detect.Capture import Capture

round = 1
bag = rosbag.Bag( f'output{round}.bag', 'w')
capture = Capture("configs/bin_cam_config.yaml", "new_cam")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码器
out = cv2.VideoWriter(f'output{round}.mp4', fourcc, 24, (capture.width, capture.height))  # 文件名，编码器，帧率，帧大小

while True:
    # 获取ROS消息和帧
    msg = ...
    frame = capture.get_frame()

    # 写入rosbag
    bag.write('topic', msg, rospy.Time.now())

    # 写入视频
    out.write(frame)

    # 其他代码...