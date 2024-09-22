# 仿照capture，创建一个Video类，用于处理视频文件的读取

import cv2




class Video:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            return frame
        else:
            return None

    def show_ori_video(self):
        while True:
            frame = self.read()
            if frame is None:
                break
            cv2.imshow('video', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

    def save(self, frame , output_path):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        out.write(frame)






    def __del__(self):
        if self.cap is not None:
            self.release()
        # cv2.destroyAllWindows()

    # release the video capture
    def release(self):
        self.cap.release()
        # cv2.destroyAllWindows()
