import cv2
from ultralytics import YOLO
import math

def classify(frame,box):
    x,y,w,h = box
    x_left = x - w / 2
    y_left = y - h / 2  

    roi = frame[int(y_left) : int(y_left + h),int(x_left) : int(x_left + w)]
    
    results = classifier.predict(roi,conf=0.5,iou=0.7)
    maxConf = -1
    label = -1

    if len(results) == 0:   # no detect
        return -1
    for result in results:
        data = result.boxes.data
        for i in range(len(data)):
            if data[i][4] > maxConf:
                maxConf = data[i][4]
                label = data[i][5]
    
    return int(label),maxConf

    


def Track_Armor(detect_model,classifier,labels,cap,out,track_cfg):

    Track_value = {}
    for i in range(1000):
        Track_value[i] = [0] * 12

    loop_times = 0

    while cap.isOpened():
    # Read a frame from the video
        exist_armor = [-1] * 12
        draw_candidate = []
        id_candidate = [0] * 1000
        success, frame = cap.read()

        if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frame
            results = detect_model.track(frame, persist=True,tracker=track_cfg)

            if results[0].boxes.id != None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                continue
    
            
            index = 0
            for box, track_id in zip(boxes, track_ids):

                if loop_times % 20 == 0:
                    for i in range(12):
                        Track_value[int(track_id)][i] = math.floor(Track_value[int(track_id)][i] / 15)

                x,y,w,h = box
                classify_label,conf = classify(frame,box)


                if classify_label != -1:
                    Track_value[int(track_id)][int(float(classify_label))] += 0.5 + conf * 0.5


                label = Track_value[int(track_id)].index(max(Track_value[int(track_id)]))

                if exist_armor[label] != -1:
                    old_id = exist_armor[label]
                    if Track_value[int(track_id)][label] < Track_value[int(old_id)][label]:
                        Track_value[(int(track_id))][label] = 0
                        label = "NULL"
                    else:
                        Track_value[(int(old_id))][label] = 0
                        old_id_index = id_candidate[old_id]
                        draw_candidate[old_id_index][5] = "NULL"
                        exist_armor[label] = track_id
                else:
                    exist_armor[label] = track_id
            

                pd = Track_value[int(track_id)][0]
                same = True
                for i in range(11):
                    if pd != Track_value[int(track_id)][i + 1]:
                        same = False
                        break
                if same == False and label != "NULL":
                    label = str(labels[label])
                else:
                    label = "NULL"

                x_left = int(x - w / 2)
                y_left = int(y - h / 2)
                x_right = int(x + w / 2)
                y_right = int(y + h / 2)
                draw_candidate.append([track_id,x_left,y_left,x_right,y_right,label])
                id_candidate[track_id] = index
                index += 1
            # cv2.rectangle(frame,(x_left,y_left),(x_right,y_right),(255,128,0),3,8)
            # cv2.putText(frame,str(track_id),(int(x_left -5) ,int(y_left - 5)) ,cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122), 2)
            # cv2.putText(frame,label,(int(x_left + w - 5),int(y_left - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122), 2)
            # cv2.imwrite("/root/yolov8-20231114/yolov8/Tracker/" + str(num) + '.jpg',frame)

        
            for box in draw_candidate:
                track_id,x_left,y_left,x_right,y_right,label = box
                cv2.rectangle(frame,(x_left,y_left),(x_right,y_right),(255,128,0),3,8)
                # print(x_left,y_left,x_right,y_right)
                cv2.putText(frame,str(track_id),(int(x_left -5) ,int(y_left - 5)) ,cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122), 2)
                cv2.putText(frame,label,(int(x_left + w - 5),int(y_left - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122), 2)

            out.write(frame)
            loop_times += 1
        else:
            break
        

        # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Open the video file
    track_cfg = r'/root/yolov8/ultralytics/cfg/trackers/bytetrack.yaml'
    video_path = r"/root/raw_left.mp4"
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    fps = 15
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
    out = cv2.VideoWriter('result.mp4', fourcc, fps, (width, height))  # 写入视频

    # Load the YOLOv8 detect_model
    detect_model = YOLO("/root/yolov8/runs/train/stage_one/weights/best.pt")
    classifier = YOLO("/root/yolov8/runs/train/stage_two/weights/best.pt")


    labels = ["B1","B2","B3","B4","B5","B7","R1","R2","R3","R4","R5","R7"]

    Track_Armor(detect_model,classifier,labels,cap,out,track_cfg)