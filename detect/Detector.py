import cv2
import torch
import datetime
from ruamel.yaml import YAML
from ultralytics import YOLO
# from Capture import Capture
import math
import time
import threading
import numpy as np
# import cupy as cp
import queue
import os
# 封装Tracker类，

# 封装Detector类
class Detector:
    def __init__(self, detector_config_path):
        # 加载配置文件
        self.cfg = YAML().load(open(detector_config_path, encoding='Utf-8', mode='r'))

        # flag
        self.is_record = self.cfg['is_record']
        self.record_fps = self.cfg['record_fps']
        if self.is_record:
            save_video_folder_path = "data/train_record/"  # 保存视频的文件夹
            today = time.strftime("%Y%m%d", time.localtime())  # 今日日期，例如2024年5月6日则为20240506
            today_video_folder_path = save_video_folder_path + today + "/"  # 今日的视频文件夹
            if not os.path.exists(today_video_folder_path):  # 当天的视频文件夹不存在则创建
                os.makedirs(today_video_folder_path)
            video_name = time.strftime("%H%M%S", time.localtime())  # 视频名称，以时分秒命名，19：29：30则为192930
            video_save_path = today_video_folder_path + video_name + ".avi"  # 视频保存路径
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.out = cv2.VideoWriter(video_save_path, fourcc, self.record_fps, (4024 , 3036))
            self.frame_queue = queue.Queue(maxsize=self.record_fps * 2)
            self.save_thread = threading.Thread(target=self.save_video, args=(self.out,self.frame_queue,), daemon=True)

        # 检测模型
        print('Loading Car Model')
        # 打印绝对路径
        # print(self.cfg['path']['stage_one_path'])
        self.model_car = YOLO(self.cfg['path']['stage_one_path'] , task = "detect")

        self.model_car2 = YOLO(self.cfg['path']['stage_two_path'])
        print('Done\n')
        # 设置参数
        self.tracker_path = self.cfg['path']['tracker_path']
        # print(self.tracker_path)
        self.stage_one_conf = self.cfg['params']['stage_one_conf']
        self.stage_two_conf = self.cfg['params']['stage_two_conf']
        self.life_time = self.cfg['params']['life_time'] # 生命周期
        self.id_candidate = [0] * 1000

        self.labels = self.cfg['params']['labels'] # 类别标签列表
        # 由labels长度初始化class_num
        self.class_num = len(self.labels) # 类别数量
        self.Track_value = {} # 这是tracker的track_id和类别的字典，主键是追踪器编号，值是一个列表，记录每个类别的识别次数。每个追踪器有所有类别的识别次数
        for i in range(1000):
            self.Track_value[i] = [0] * self.class_num
        self.id_candidate = [0] * 10000
        
        self.Gray2Blue = {12:5,13:1,14:0,15:3,16:2,17:4}
        self.Gray2Red = {12:11,13:7,14:6,15:9,16:8,17:10}
        self.Blue2Gray ={v: k for k, v in self.Gray2Blue.items()}
        self.Red2Gray ={v: k for k, v in self.Gray2Red.items()}
        self.gray_thresh = 15
        # 设置计数器
        self.loop_times = 0

        # 保存视频多线程操作
        self.save_video_working_flag = False
        self.get_first_frame_flag = False
        # 多线程操作
        self.threading = None
        self.working_flag = False
        self.init_flag = False
        # 初始化锁对象和结果列表
        self._result_lock = threading.Lock()
        self._results = [None , None]


    # 保存视频线程开始工作
    def start_save_video(self):
        if self.is_record:
            self.save_video_working_flag = True
            self.save_thread.start()
    # 保存视频线程停止工作
    def stop_save_video(self):
        if self.is_record:

            self.save_video_working_flag = False
            print("set stop done")
            if self.out is not None:
                self.out.release()
                print("release done")
            # self.save_thread.join()

    # 创建线程
    def create(self, capture):
        if not self.init_flag:
            self.threading = threading.Thread(target=self.detect_thread, args=(capture,), daemon=True)
            self.init_flag = True

    # 启动线程
    def start(self):
        if self.init_flag:
            print("detect start function")
            self.working_flag = True  # 先设True再开始，否则开始太快检测为False直接结束了
            self.threading.start()

    # 关闭线程
    def stop(self):
        if self.init_flag:
            print("detect stop")


            self.working_flag = False


            # self.threading.join()

    # 清除
    def release(self):
        self.model_car = None
        self.model_car2 = None
        self.tracker_path = None
        self.stage_one_conf = None
        self.stage_two_conf = None
        self.life_time = None
        self.labels = None
        self.class_num = None
        self.Track_value = None
        self.id_candidate = None
        self.loop_times = None
        self.threading = None
        self.working_flag = None
        self.init_flag = None
        self._result_lock = None
        self._results = None
        if self.is_record:
            if self.out is not None:
                self.out.release()

    # 二阶段分类推理Classify
    def classify_infer(self, roi_list): # 输入原图和box, 返回分类结果


        
        results = self.model_car2.predict(roi_list, conf=0.5, iou=0.7 , device = 0 ,verbose = False  )
        if len(results) == 0:  # no detect
            return -1

        label_list = []
        conf_list = []
        for result,roi in zip(results,roi_list):
            data = result.boxes.data
            maxConf = -1
            label = -1
            index = -1
            tmp = []
            for i in range(len(data)):
                if data[i][4] > maxConf:
                    maxConf = data[i][4]
                    label = data[i][5]
                    index = i
                    tmp=data[i]
            if len(tmp) != 0:
                x1,y1,x2,y2 = tmp[:4]
                gray = roi[int(y1):int(y2),int(x1):int(x2)]
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                if cv2.mean(gray)[0] < self.gray_thresh :
                    if label < 6 : # Blue
                        label = self.Blue2Gray[int(label)]
                    else:
                        label = self.Red2Gray[int(label)]
                
            label_list.append(int(label))
            conf_list.append(float(maxConf))
            


        return label_list,conf_list


    # 一阶段追踪推理
    def track_infer(self, frame):
        results = self.model_car.track(frame, persist=True,tracker=self.tracker_path,verbose = False)
        return results

    # 对results的结果进行判空
    def is_results_empty(self, results):
        if results is None:
            # print("No results!")
            return True
        if results[0].boxes.id is None:
            # print("No detect!")
            return True
        return False

    # 解析results
    def parse_results(self, results):
        confidences = results[0].boxes.conf.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        return confidences, boxes, track_ids

    # 画出result
    def draw_result(self, frame, box, result, conf):
        if result != "NULL":
            # 画上分类结果
            x, y, w, h = box
            cv2.putText(frame, str(result), (int(box[0] - 5), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 122), 2)
            # 画上置信度 debug
            cv2.putText(frame, str(round(conf, 2)), (int(box[0] - 5), int(box[1] - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122), 2)
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                          (0, 255, 122), 5)
        return frame
    

    # 总的推理
    def infer(self, frame): # 输入原图，返回推理结果
        # print("infer")
        # frame判空
        if frame is None:
            # print("No frame!")
            return None , None

        # # 展示frame的尺寸
        # print(frame.shape)


        # 获取推理结果
        # print("track_infer")
        results = self.track_infer(frame)
        # print("after track")



        # 一阶段results判空
        if self.is_results_empty(results):
            # print("No results!")
            return frame , None


        exist_armor = [-1] * 12 # armor number映射Track_id
        draw_candidate = [] # 待draw的bbox(track_id,x_left,y_left,x_right,y_right,label)

        confidences, boxes, track_ids = self.parse_results(results) # 可以不要
        zip_results = [] # 最终返回存储追踪器的结果，有box, 分类id, conf


        roi_list = []
        id_list = []
        box_list = []


        for box, track_id, conf in zip(boxes, track_ids, confidences):


            if self.loop_times % self.life_time == 1: # 0的话无法单张预测
                for i in range(12):
                    self.Track_value[int(track_id)][i] = math.floor(self.Track_value[int(track_id)][i] / 10)

            # 获取二阶段roi_batch
            x,y,w,h = box
            x_left = x - w / 2
            y_left = y - h / 2
            roi = frame[int(y_left): int(y_left + h), int(x_left): int(x_left + w)]
            roi_list.append(roi)

            # 获取track_id的列表
            id_list.append(track_id)
            # 获取boxes列表
            box_list.append(box)
        # print("to_infer")
        label_list,conf_list = self.classify_infer(roi_list)


        index = 0
        for i in range(len(roi_list)):

            classify_label = label_list[i]
            conf = conf_list[i]
            track_id = id_list[i]
            box = box_list[i]
            x,y,h,w = box

            status = 0 # normal

            # 二阶段识别次数和置信度的加权
            if classify_label != -1:
                label = self.Track_value[int(track_id)].index(max(self.Track_value[int(track_id)]))
                if classify_label > 11: # is Gray
                    status = 1
                    if label < 6 : # is Blue
                        self.Track_value[int(track_id)][int(float(self.Gray2Blue[classify_label]))] += 0.5 + conf * 0.5
                    else : # is Red
                        self.Track_value[int(track_id)][int(float(self.Gray2Red[classify_label]))] += 0.5 + conf * 0.5
                else :
                    self.Track_value[int(track_id)][int(float(classify_label))] += 0.5 + conf * 0.5

            label = self.Track_value[int(track_id)].index(max(self.Track_value[int(track_id)])) # 找到计数器最大的类别,即追踪器的分类结果

            # 判重
            '''
                判重:
                    这部分直接举例 ：假设id_1匹配上了B3，并且此时id_3的匹配结果也是B3，那么对其维护的B3装甲板的加权和进行比较，
                    若当前id的加权和较小，则当前维护的加权和清零
                    若当前id的加权和较大，则将old_id维护的加权和清零，并且将draw_candidate[old_id]的label置为NULL
            '''
            if exist_armor[label] != -1:
                old_id = exist_armor[label]
                if self.Track_value[int(track_id)][label] < self.Track_value[int(old_id)][label]:
                    self.Track_value[(int(track_id))][label] = 0
                    label = "NULL"
                else:
                    self.Track_value[(int(old_id))][label] = 0
                    old_id_index = self.id_candidate[old_id]
                    draw_candidate[old_id_index][5] = "NULL"
                    exist_armor[label] = track_id
            else:
                exist_armor[label] = track_id

            pd = self.Track_value[int(track_id)][0] # 获取计数器的第一个值

            # 判断是否所有类别的计数器都一样，如果一样，说明没有识别出来，返回NULL
            same = True
            for j in range(self.class_num - 1):
                if pd != self.Track_value[int(track_id)][j + 1]:
                    same = False
                    break

            if same == False and label != "NULL": # label是分类信息
                label = str(self.labels[label])
            else:
                label = "NULL" 


            # result是预测的类别的字符形式，如果不是NULL, 画上分类结果
            # self.draw_result(frame, box, result, conf)
            # 返回结果,现在是画上了结果的frame
            # 把识别的box，conf和最终的类别组合后返回
            # tracker_results.append((box, label, conf)) # 返回的是一个列表，每个元素是一个元组，包含了box, 分类结果和置信度

            x_left = int(x - w / 2)
            y_left = int(y - h / 2)
            x_right = int(x + w / 2)
            y_right = int(y + h / 2)

            xywh_box = [x,y,w,h]
            xyxy_box = [x_left,y_left,x_right,y_right]
            # score = self.Track_value[int(track_id)][int(float(classify_label))]
            draw_candidate.append([track_id, x_left, y_left, x_right, y_right, label])
            # print("in",xyxy_box,xywh_box)
            zip_results.append([xyxy_box, xywh_box , track_id , label ])

            self.id_candidate[track_id] = index

            index = index + 1

        for box in draw_candidate:
            track_id, x_left, y_left, x_right, y_right, label = box
            cv2.rectangle(frame, (x_left, y_left), (x_right, y_right), (255, 128, 0), 3, 8)
            cv2.putText(frame, label, (int(x_left - 10), int(y_right + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 122), 2)
            cv2.putText(frame, str(track_id), (int(x_right + 5), int(y_right + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 122), 2)

        self.loop_times = self.loop_times + 1
        return frame , zip_results

    # 保存视频
    def save_video(self, out, frame_queue):
        last_time_save_video = time.time()
        while True:
            if not self.save_video_working_flag:
                print("save video stop")
                return
            if not self.get_first_frame_flag:
                print("waiting for first frame")
                time.sleep(1)
                continue
            try:
                frame = frame_queue.get(False) # 从队列中取出帧,如果队列为空，抛出异常
            except queue.Empty:
                print("queue empty")
                time.sleep((1/self.record_fps))
                continue
            if frame is not None:
                if time.time() - last_time_save_video < (1/self.record_fps):
                    time.sleep((1/self.record_fps) - (time.time() - last_time_save_video))
                print("write frame")
                out.write(frame)
                last_time_save_video = time.time()
            else:
                time.sleep((1/self.record_fps))
                continue

    # 目标检测子线程
    def detect_thread(self, capture):
        start_time = time.time()
        if self.is_record:
            self.start_save_video()
        while True:  # 无限循环以持续处理图像
            if not self.working_flag:
                break
            now = time.time()
            fps = 1 / (now - start_time)
            start_time = now
            print("in fps",fps)
            frame = capture.get_frame()
            if frame is not None:
                if self.is_record:
                    self.frame_queue.put(np.copy(frame))
                    self.get_first_frame_flag = True
                # 执行目标检测
                infer_result = self.infer(frame)

                if infer_result is not None:
                    with self._result_lock:
                        # print("update detect result")
                        self._results = infer_result  # 更新结果列表
        if self.is_record:
            self.stop_save_video()





        # 方法，来源于主线程调取最新的结果
    def get_results(self):
        # with self._result_lock: # TODO:把锁去掉了,否则更新结果那里一直在等待锁,想个办法解决
        return self._results  # 返回最新的结果列表






# 检测模型
# print('Loading Car Model')
# model_car = YOLO("../weights/train/stage_one/weights/best.pt")
# model_car2 = YOLO("../weights/train/stage_two/weights/best.pt")
# print('Done\n')
#
# id_label = {}
# for i in range(1000):
#     id_label[i] = [0] * 12
#
# labels = ["B1", "B2", "B3", "B4", "B5", "B7", "R1", "R2", "R3", "R4", "R5", "R7"]
#

# Classify function
# def classify(frame, box):
#     x, y, w, h = box
#     x_left = x - w / 2
#     y_left = y - h / 2
#
#     roi = frame[int(y_left): int(y_left + h), int(x_left): int(x_left + w)]
#
#     results = model_car2.predict(roi, conf=0.5, iou=0.7)
#     maxConf = -1
#     label = -1
#     if len(results) == 0:  # no detect
#         return -1
#     for result in results:
#         data = result.boxes.data
#         for i in range(len(data)):
#             if data[i][4] > maxConf:
#                 maxConf = data[i][4]
#                 label = data[i][5]
#
#     return int(label)
#
# # 加载配置文件
# main_cfg_path = "../configs/main_config.yaml"
# binocular_camera_cfg_path = "../configs/bin_cam_config.yaml"
# main_cfg = YAML().load(open(main_cfg_path, encoding='Utf-8', mode='r'))
# bin_cam_cfg = YAML().load(open(binocular_camera_cfg_path, encoding='Utf-8', mode='r'))
# detector_config_path = "../configs/detector_config.yaml"
# def main():
#     print("Loading right camera")
#     # capture初始化
#     capture = Capture(binocular_camera_cfg_path, 'new_cam')
#     detector = Detector(detector_config_path)
#     print("Done")
#
#     # 计算fps
#     last_time = time.time()-1
#
#     while True:
#
#
#         # 获取图像`
#         frame = capture.get_frame()
#
#         # 计算fps
#         now = time.time()
#         fps = 1 / (now - last_time)
#         # print("fps: ", fps)
#         last_time = now
#
#         # 如果按下q，那么停止循环
#         if cv2.waitKey(1) == ord('q'):
#             break
#
#         result_img = detector.infer(frame)
#         # if results is not None:
#         #     # 提取results并画上结果
#         #     for box, result, conf in results:
#         #        frame = detector.draw_result(frame, box, result, conf)
#
#         # 绘制fps
#         cv2.putText(frame, "fps: " + str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122), 2)
#         capture.show_img(result_img)
#
#
#
#         # 确定我们有一幅生效的图片
#         detector.loop_times += 1
#
#     # 关闭摄像头
#     capture.camera_close()
#
#     # 关闭所有 OpenCV 窗口
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()

