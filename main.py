
from detect.Detector import Detector
from detect.Video import Video
from detect.Capture import Capture
from Lidar.Lidar import Lidar
from Lidar.Converter import Converter
from Lidar.PointCloud import *
from Car import *
import cv2
import time
from collections import deque
from ruamel.yaml import YAML

# 创建一个长度为N的队列

mode = "video" # "video" or "camera"
round = 2 #训练赛第几轮

if __name__ == '__main__':
    video_path = "/home/nvidia/RadarWorkspace/code/Radar_Develop/data/tran_record/0504/video/round2.mp4"
    detector_config_path = "configs/detector_config.yaml"
    binocular_camera_cfg_path = "configs/bin_cam_config.yaml"
    main_config_path = "configs/main_config.yaml"
    main_cfg = YAML().load(open(main_config_path, encoding='Utf-8', mode='r'))

    # 类初始化
    detector = Detector(detector_config_path)
    lidar = Lidar(main_cfg)

    if mode == "video":
        capture = Video(video_path)
    elif mode == "camera":
        capture = Capture(binocular_camera_cfg_path,"new_cam")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码器
    # out = cv2.VideoWriter(f'output{round}.mp4', fourcc, 24, (1920, 1280))  # 文件名，编码器，帧率，帧大小


    # fps计算
    N = 10
    fps_queue = deque(maxlen=N)
    start_time = time.time()

    # 开启激光雷达线程
    lidar.start()

    # 主循环
    while True:
        # print("main loop")
        # 读取frame
        frame = capture.get_frame()

        # 计算fps
        now = time.time()
        fps = 1 / (now - start_time)
        start_time = now
        # 将FPS值添加到队列中
        fps_queue.append(fps)
        # 计算平均FPS
        avg_fps = sum(fps_queue) / len(fps_queue)

        # 读图失败，推出
        if frame is None:
            print("no frame")
            break

        # 目标检测部分
        ori_frame = frame.copy()
        # 获得推理结果=
        infer_result = detector.infer(frame)
        image = ori_frame
        if infer_result is not None:
            result_img ,results = infer_result
            if results is not None:
                pc_all = lidar.get_all_pc()
                # 对每个结果进行分析 , 进行目标定位
                for result in results:
                    # 对每个检测框进行处理，获取对应点云
                    xywh_box , track_id , label = result





            if result_img is not None:
                # 用新图替代
                image = result_img

        cv2.putText(image, "fps: {:.2f}".format(avg_fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122),
                    2)

        image = cv2.resize(image, (1920, 1280))
        # out.write(image)
        cv2.imshow("frame", image)
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    # out.release()
    lidar.stop()

    cv2.destroyAllWindows()