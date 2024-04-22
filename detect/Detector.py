import cv2
import torch
import datetime
from ruamel.yaml import YAML
from ultralytics import YOLO
from Capture import Capture
import math
import time

# 封装Detector类
class Detector:
    def __init__(self, detector_config_path):
        # 加载配置文件
        self.cfg = YAML().load(open(detector_config_path, encoding='Utf-8', mode='r'))

        # 检测模型
        print('Loading Car Model')
        self.model_car = YOLO(self.cfg['path']['stage_one_path'])
        self.model_car2 = YOLO(self.cfg['path']['stage_two_path'])
        print('Done\n')
        # 设置参数
        self.tracker_path = self.cfg['params']['tracker_path']
        self.stage_one_conf = self.cfg['params']['stage_one_conf']
        self.stage_two_conf = self.cfg['params']['stage_two_conf']
        self.life_time = self.cfg['params']['life_time'] # 生命周期

        self.labels = self.cfg['params']['labels'] # 类别标签列表
        # 由labels长度初始化class_num
        self.class_num = len(self.labels) # 类别数量
        self.id_label = {} # 这是tracker的track_id和类别的字典，主键是追踪器编号，值是一个列表，记录每个类别的识别次数。每个追踪器有所有类别的识别次数
        for i in range(1000):
            self.id_label[i] = [0] * self.class_num
        # 设置计数器
        self.loop_times = 0



    # 二阶段分类推理Classify
    def classify_infer(self, frame, box): # 输入原图和box, 返回分类结果

        x, y, w, h = box

        x_left = x - w / 2
        y_left = y - h / 2

        roi = frame[int(y_left): int(y_left + h), int(x_left): int(x_left + w)]

        results = model_car2.predict(roi, conf=0.5, iou=0.7)
        maxConf = -1
        label = -1
        if len(results) == 0:  # no detect
            return -1
        for result in results:
            data = result.boxes.data
            for i in range(len(data)):
                if data[i][4] > maxConf:
                    maxConf = data[i][4]
                    label = data[i][5]

        return int(label)

    # 一阶段追踪推理
    def track_infer(self, frame):
        results = self.model_car.track(frame, persist=True,tracker=self.tracker_path)
        return results

    # 对results的结果进行判空
    def is_results_empty(self, results):
        if results is None:
            print("No results!")
            return True
        if results[0].boxes.id is None:
            print("No detect!")
            return True
        return False

    # 解析results
    def parse_results(self, results):
        confidences = results[0].boxes.conf.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        return confidences, boxes, track_ids

    # 总的推理
    def infer(self, frame): # 输入原图，返回推理结果
        if frame is None:
            print("No frame!")
            return None

        # 获取推理结果
        results = self.track_infer(frame)

        # 判空
        if self.is_results_empty(results):
            self.show_img(frame)
            return None

        confidences, boxes, track_ids = self.parse_results(results) # 可以不要

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            if conf < self.stage_one_conf:
                continue
            new_id = self.classify_infer(frame, box)

            if new_id != -1:
                id_label[int(track_id)][int(float(new_id))] += 1 # 对追踪器的类别计数器进行更新
                if self.loop_times % self.life_time == 0: # 当进行了30次处理（30帧每秒的话，则是1秒）后，对计数器进行更新，每个类别的计数器除以10
                    for i in range(self.class_num): # 对所有类别进行衰减,防止过大无法修正 TODO:检查如果某一帧没有track_id号追踪器，后面是否不会有这一号，如果有，会不会出现刚好这一帧没有导致的不衰减问题
                        id_label[int(track_id)][i] = math.floor(id_label[int(track_id)][i] / 10)

            result = id_label[int(track_id)].index(max(id_label[int(track_id)])) # 找到计数器最大的类别,即追踪器的分类结果
            pd = id_label[int(track_id)][0] # 获取计数器的第一个值
            # 判断是否所有类别的计数器都一样，如果一样，说明没有识别出来，返回NULL
            same = True
            for i in range(self.class_num-1):
                if pd != id_label[int(track_id)][i + 1]:
                    same = False
                    break
            if same == True:
                result = "NULL"
            else:
                result = str(labels[result])

            # result是预测的类别的字符形式，如果不是NULL, 画上分类结果
            if result != "NULL":
                # 画上分类结果
                x, y, w, h = box
                cv2.putText(frame, result, (int(box[0] - 5), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 122), 2)
                # 画上置信度 debug
                cv2.putText(frame, str(round(conf, 2)), (int(box[0] - 5), int(box[1] - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122), 2)
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                              (0, 255, 122), 2)
            # 返回结果,现在是画上了结果的frame
            # TODO:把识别的box，conf和最终的类别组合后返回
            return frame







# 检测模型
print('Loading Car Model')
model_car = YOLO("../weights/train/stage_one/weights/best.pt")
model_car2 = YOLO("../weights/train/stage_two/weights/best.pt")
print('Done\n')

id_label = {}
for i in range(1000):
    id_label[i] = [0] * 12

labels = ["B1", "B2", "B3", "B4", "B5", "B7", "R1", "R2", "R3", "R4", "R5", "R7"]


# Classify function
def classify(frame, box):
    x, y, w, h = box
    x_left = x - w / 2
    y_left = y - h / 2

    roi = frame[int(y_left): int(y_left + h), int(x_left): int(x_left + w)]

    results = model_car2.predict(roi, conf=0.5, iou=0.7)
    maxConf = -1
    label = -1
    if len(results) == 0:  # no detect
        return -1
    for result in results:
        data = result.boxes.data
        for i in range(len(data)):
            if data[i][4] > maxConf:
                maxConf = data[i][4]
                label = data[i][5]

    return int(label)

# 加载配置文件
main_cfg_path = "../configs/main_config.yaml"
binocular_camera_cfg_path = "../configs/bin_cam_config.yaml"
main_cfg = YAML().load(open(main_cfg_path, encoding='Utf-8', mode='r'))
bin_cam_cfg = YAML().load(open(binocular_camera_cfg_path, encoding='Utf-8', mode='r'))
detector_config_path = "../configs/detector_config.yaml"
def main():
    print("\nLoading right camera")
    # capture初始化
    capture = Capture(binocular_camera_cfg_path, 'new_cam')
    detector = Detector(detector_config_path)
    print("Done")


    loop_times = 0

    while True:
        image_right = capture.get_frame()

        # 如果按下q，那么停止循环
        if cv2.waitKey(1) == ord('q'):
            break

        if image_right is not None:

            # 获取推理结果
            results = detector.track_infer(image_right)
            # results = model_car.track(image_right, persist=True,tracker='../configs/bytetrack.yaml')

            # 判空
            if detector.is_results_empty(results):
                capture.show_img(image_right)
                continue
            # if results is None:
            #     # 就算没有检测到，也要显示右摄像头的画面
            #     print("No results!")
            #     capture.show_img(image_right)
            #     #cv2.waitKey(1)
            #     continue
            #
            # if results[0].boxes.id is None:
            #     # 就算没有检测到，也要显示右摄像头的画面
            #     print("No detect!")
            #     capture.show_img(image_right)
            #     #cv2.waitKey(1)
            #     continue

            # confidences = results[0].boxes.conf.cpu().numpy()
            # boxes = results[0].boxes.xywh.cpu().numpy()
            # track_ids = results[0].boxes.id.int().cpu().tolist()
            # 解析results
            confidences, boxes, track_ids = detector.parse_results(results)

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                if conf < detector.stage_one_conf:
                    capture.show_img(image_right)
                    #cv2.waitKey(1)
                    continue
                new_id = classify(image_right, box)

                if new_id != -1:
                    id_label[int(track_id)][int(float(new_id))] += 1
                    if loop_times % 29 == 0:
                        for i in range(12):
                            id_label[int(track_id)][i] = math.floor(id_label[int(track_id)][i] / 10)

                result = id_label[int(track_id)].index(max(id_label[int(track_id)]))
                pd = id_label[int(track_id)][0]
                same = True
                for i in range(11):
                    if pd != id_label[int(track_id)][i + 1]:
                        same = False
                        break
                if same == True:
                    result = "NULL"
                else:
                    result = str(labels[result])



                # 如果result不是null, 画上分类结果
                if result != "NULL":
                    # 画上分类结果
                    x, y, w, h = box
                    cv2.putText(image_right, result, (int(box[0] - 5), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 255, 122), 2)
                    # 画上置信度 debug
                    cv2.putText(image_right, str(round(conf, 2)), (int(box[0] - 5), int(box[1] - 25)),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 122), 2)
                    cv2.rectangle(image_right, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                  (0, 255, 122), 2)

            print("show!")
            capture.show_img(image_right)
            #cv2.waitKey(1)


        # 确定我们有一幅生效的图片
        loop_times += 1

    # 关闭摄像头
    capture.camera_close()

    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

