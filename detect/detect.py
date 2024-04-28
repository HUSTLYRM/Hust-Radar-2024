from Capture import Capture
from Detector import Detector
from Video import Video
import cv2

def main():
    video_path = "../data/right.mp4"
    detector_config_path = "../configs/detector_config.yaml"
    video = Video(video_path)
    detector = Detector(detector_config_path)

    while True:
        # detector.id_candidate = [0] * 1000 # 每一轮都要刷新？
        frame = video.get_frame()
        if frame is None:
            print("no frame")
            break
        ori_frame = frame.copy()
        result_img = detector.infer(frame)
        if result_img is None:
            cv2.imshow("frame", ori_frame)
            cv2.waitKey(1)
            continue
        cv2.imshow("frame", result_img)
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

main()