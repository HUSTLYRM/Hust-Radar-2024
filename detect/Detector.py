import cv2
import torch
import datetime
from ruamel.yaml import YAML
from ultralytics import YOLO
import math
import time


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


def main():
    print("\nLoading right camera")
    # capture初始化
    capture = Capture(binocular_camera_cfg_path, 'new_cam')


    # print(ret.contents)
    print("Done")

    loop_times = 0

    while True:
        image_right = capture.get_frame()

        # 如果按下q，那么停止循环
        if cv2.waitKey(1) == ord('q'):
            break

        if image_right is not None:

            results = model_car.track(image_right, persist=True,tracker='../configs/bytetrack.yaml')

            if results is None:
                # 就算没有检测到，也要显示右摄像头的画面
                print("No results!")
                capture.show_img(image_right)
                #cv2.waitKey(1)
                continue

            if results[0].boxes.id is None:
                # 就算没有检测到，也要显示右摄像头的画面
                print("No detect!")
                capture.show_img(image_right)
                #cv2.waitKey(1)
                continue

            confidences = results[0].boxes.conf.cpu().numpy()
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                if conf < 0.01:
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

