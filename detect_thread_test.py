from detect.Detector import Detector
from detect.Video import Video
from detect.Capture import Capture

import cv2
import time
from collections import deque
from ruamel.yaml import YAML
import threading


mode = "video"  # "video" or "camera"
round = 1  # 训练赛第几轮

if __name__ == '__main__':
    video_path = "/home/nvidia/RadarWorkspace/code/Radar_Develop/data/tran_record/0505/ori_data/video10.mp4"
    detector_config_path = "configs/detector_config.yaml"
    binocular_camera_cfg_path = "configs/bin_cam_config.yaml"
    main_config_path = "configs/main_config.yaml"
    main_cfg = YAML().load(open(main_config_path, encoding='Utf-8', mode='r'))

    # 类初始化
    detector = Detector(detector_config_path)

    if mode == "video":
        capture = Video(video_path)
    elif mode == "camera":
        capture = Capture(binocular_camera_cfg_path, "new_cam")


    # fps计算
    N = 10
    fps_queue = deque(maxlen=N)
    start_time = time.time()

    # 启动图像处理子线程
    threading.Thread(target=detector.detect_thread, args=(capture,), daemon=True).start()


    # 主循环
    while True:


        # 计算fps
        now = time.time()
        fps = 1 / (now - start_time)
        start_time = now
        # 将FPS值添加到队列中
        fps_queue.append(fps)
        # 计算平均FPS
        avg_fps = sum(fps_queue) / len(fps_queue)

        infer_result = detector.get_results()
        if infer_result is not None and len(infer_result) == 2:
            result_img, zip_results = infer_result

            cv2.imshow("result", result_img)


        print("out:",avg_fps)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()

    cv2.destroyAllWindows()