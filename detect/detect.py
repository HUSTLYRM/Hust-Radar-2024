from Capture import Capture
from Detector import Detector
from Video import Video
import cv2
import time
from collections import deque

# 创建一个长度为N的队列


def main():
    video_path = "../data/right.mp4"
    detector_config_path = "../configs/detector_config.yaml"
    video = Video(video_path)
    detector = Detector(detector_config_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码器
    out = cv2.VideoWriter('output.mp4', fourcc, 24, (video.width, video.height))  # 文件名，编码器，帧率，帧大小

    N = 10
    fps_queue = deque(maxlen=N)

    start_time = time.time()
    while True:
        # detector.id_candidate = [0] * 1000 # 每一轮都要刷新？
        # 计算fps,fps使用均值滤波


        frame = video.get_frame()

        now = time.time()
        fps = 1 / (now - start_time)
        start_time = now

        # 将FPS值添加到队列中
        fps_queue.append(fps)

        # 计算平均FPS
        avg_fps = sum(fps_queue) / len(fps_queue)

        if frame is None:
            print("no frame")
            break
        ori_frame = frame.copy()
        result_img = detector.infer(frame)
        if result_img is None:
            # 画上fps
            cv2.putText(ori_frame, "fps: {:.2f}".format(avg_fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122), 2)
            out.write(ori_frame)
            cv2.imshow("frame", ori_frame)
            cv2.waitKey(1)
            continue

        cv2.putText(result_img, "fps: {:.2f}".format(avg_fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122), 2)
        out.write(result_img)
        cv2.imshow("frame", result_img)
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()

main()