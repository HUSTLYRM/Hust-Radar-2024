# 仿照capture，创建一个Video类，用于处理视频文件的读取

import cv2
from ruamel.yaml import YAML

# # 加载配置文件
# main_cfg_path = "../configs/main_config.yaml"
# binocular_camera_cfg_path = "../configs/bin_cam_config.yaml"
# main_cfg = YAML().load(open(main_cfg_path, encoding='Utf-8', mode='r'))
# bin_cam_cfg = YAML().load(open(binocular_camera_cfg_path, encoding='Utf-8', mode='r'))

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

# def main():
#     image_path = "../data/002808.jpg"
#     video_path = "../data/right.mp4"
#     detector_config_path = "../configs/detector_config.yaml"
#     image = cv2.imread(image_path)
#
#     video = Video(video_path)
#     detector = Detector(detector_config_path)
#
#     # 复制一份图像用于一阶段检测
#     stage_one = image.copy()
#     roi_results = detector.track_infer(stage_one)
#     # roi_results有成员confidences，boxes，track_ids,用parse_results解析
#     confidences, boxes, track_ids = detector.parse_results(roi_results)
#     # 将一阶段检测结果提取并绘制到图像上，并imwrite保存
#     for box, track_id, conf in zip(boxes, track_ids, confidences):
#         # x,y,w,h = box
#         # roi = stage_one[y:y+h, x:x+w]
#         stage_one = detector.draw_result(stage_one, box, track_id, conf)
#     cv2.imwrite('stage_one.jpg', stage_one)
#
#     # 把一阶段的所有检测框截取roi，进行二阶段检测后，将二阶段的检测结果绘制到roi小图上，并保存
#     for box, track_id, conf in zip(boxes, track_ids, confidences):
#         x, y, w, h = box
#         roi = image[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
#         cv2.imwrite(f'roi_ori{track_id}.jpg', roi)
#         results = detector.model_car2.predict(roi, conf=0.5, iou=0.7)
#         if results is not None:
#             for result in results:
#                 data = result.boxes.data
#                 for i in range(len(data)):
#                     # data[i][4]是confidence，data[i][5]是label
#                     box = data[i][:4]
#                     conf = data[i][4].item()
#                     # print(conf)
#                     label = data[i][5].item()
#                     # print(label)
#                     label_name = detector.labels[int(label)]
#                     xs, ys, ws, hs = box
#                     # print(box)
#
#                     xs = xs.item()
#                     ys = ys.item()
#                     ws = ws.item()
#                     hs = hs.item()
#                     print(xs, ys, ws, hs)
#
#                     cv2.putText(roi, str(label_name), (int(box[0] - 5), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                 (0, 255, 122), 2)
#                     # 画上置信度 debug
#                     cv2.putText(roi, str(round(conf, 2)), (int(box[0] - 5), int(box[1] - 25)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 122), 2)
#                     # cv2.rectangle(roi, (int(xs - ws / 2), int(ys - hs / 2)), (int(xs + ws / 2), int(ys + hs / 2)),(0, 255, 122), 2)
#                     cv2.imwrite(f'roi{track_id}.jpg', roi)
#     return
#
#
#
#     results = detector.infer(image)
#     if results is not None:
#         for box, result, conf in results:
#             # 根据box裁剪出目标区域，并保存
#             x,y,w,h = box
#             roi = image[y:y+h, x:x+w]
#             print(result)
#             image = detector.draw_result(image, box, result, conf)
#     cv2.imshow('image', image)
#     cv2.imwrite('result.jpg', image)
#     cv2.waitKey(0)
#     return
#     while True:
#         frame = video.get_frame()
#         if frame is None:
#             break
#         results = detector.infer(frame)
#         if results is not None:
#             for box, result, conf in results:
#                 frame = detector.draw_result(frame, box, result, conf)
#         cv2.imshow('video', frame)
#         key = cv2.waitKey(1)
#         if key == 27:
#             break
# main()