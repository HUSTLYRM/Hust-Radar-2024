
from detect.Detector import Detector
from detect.Video import Video
from detect.Capture import Capture
from Lidar.Lidar import Lidar
from Lidar.Converter import Converter
from Lidar.PointCloud import PcdQueue
from Car.Car import *
import numpy as np
import threading
import cv2
import time
import open3d as o3d
from collections import deque
from ruamel.yaml import YAML
import os

# 创建一个长度为N的队列

mode = "camera" # "video" or "camera"
save_video = False
round = 11 # 练赛第几轮

if __name__ == '__main__':
    video_path = "/home/nvidia/RadarWorkspace/code/Radar_Develop/data/tran_record/0505/ori_data/video10.mp4"
    detector_config_path = "configs/detector_config.yaml"
    binocular_camera_cfg_path = "configs/bin_cam_config.yaml"
    main_config_path = "configs/main_config.yaml"
    converter_config_path = "configs/converter_config.yaml"
    main_cfg = YAML().load(open(main_config_path, encoding='Utf-8', mode='r'))
    if save_video:
        save_video_folder_path = "data/train_record/" # 保存视频的文件夹
        # 今日日期，例如2024年5月6日则为20240506
        today = time.strftime("%Y%m%d", time.localtime())
        # 今日的视频文件夹
        today_video_folder_path = save_video_folder_path + today + "/"
        # 当天的视频文件夹不存在则创建
        if not os.path.exists(today_video_folder_path):
            os.makedirs(today_video_folder_path)
        # 视频名称，以时分秒命名，19：29：30则为192930
        video_name = time.strftime("%H%M%S", time.localtime())
        # 视频保存路径
        video_save_path = today_video_folder_path + video_name + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码器
        out = cv2.VideoWriter(video_save_path, fourcc, 6, (1920, 1280))  # 文件名，编码器，帧率，帧大小



    # 类初始化
    detector = Detector(detector_config_path)
    lidar = Lidar(main_cfg)
    converter = Converter(converter_config_path)  # 传入的是path
    carList = CarList(main_cfg)

    if mode == "video":
        capture = Video(video_path)
    elif mode == "camera":
        capture = Capture(binocular_camera_cfg_path,"new_cam")
    else:
        print("mode error")
        exit(1)

    start_time = time.time()
    # fps计算
    N = 10
    fps_queue = deque(maxlen=N)


    # 开启激光雷达线程
    lidar.start()
    threading.Thread(target=detector.detect_thread, args=(capture,), daemon=True).start()



    # 主循环
    while True:
        # print("main loop")
        # 读取frame
        # 开始计时
        # test_time_start = time.time()
        # frame = capture.get_frame()
        # # 结束计时
        # test_time_end = time.time()
        # # 计算用时多少ms
        # print("time:", (test_time_end - test_time_start) * 1000)

        # if frame is not None and frame.size > 0:
        #     # cv2.imshow("ori", frame)
        # else:
        #     print("end")

        # 计算fps
        now = time.time()
        fps = 1 / (now - start_time)
        start_time = now
        # 将FPS值添加到队列中
        fps_queue.append(fps)
        # 计算平均FPS
        avg_fps = sum(fps_queue) / len(fps_queue)

        print("fps:",avg_fps)

        # 获得推理结果
        infer_result = detector.get_results()
        # carList_results , # result in carList_results: [track_id , car_id , xywh , conf ,  camera_xyz , filed_xyz ]
        # 需要打包一份给carList
        carList_results = []
        result_img = None
        # 确保推理结果不为空且可以解包
        if infer_result is not None and len(infer_result) == 2:
            # print(infer_result)
            result_img, results = infer_result

            if results is not None:
                if lidar.pcdQueue.point_num == 0:
                    continue
                pc_all = lidar.get_all_pc()

                # 创建总体点云pcd
                pcd_all = o3d.geometry.PointCloud()
                pcd_all.points = o3d.utility.Vector3dVector(pc_all)

                # 将总体点云转到相机坐标系下
                converter.lidar_to_camera(pcd_all)

                # 检测框对应点云
                box_pcd = o3d.geometry.PointCloud()
                # 对每个结果进行分析 , 进行目标定位
                for result in results:
                    # 对每个检测框进行处理，获取对应点云
                    # 结果：[xyxy_box, xywh_box , track_id , label ]
                    xyxy_box, xywh_box ,  track_id , label = result # xywh的xy是中心点的xy

                    # 如果没有分类出是什么车，跳过
                    if label == "NULL":
                        continue
                    # print("xyxy",xyxy_box)
                    # print("xywh",xywh_box)

                    # 获取新xyxy_box , 原来是左上角和右下角，现在想要中心点保持不变，宽高设为原来的一半，再计算一个新的xyxy_box,可封装
                    div_times = 2
                    new_w = xywh_box[2] / div_times
                    new_h = xywh_box[3] / div_times
                    new_xyxy_box = [xywh_box[0] - new_w / 2, xywh_box[1] - new_h / 2, xywh_box[0] + new_w / 2, xywh_box[1] + new_h / 2]

                    # 获取检测框内numpy格式pc
                    box_pc = converter.get_points_in_box(pcd_all, new_xyxy_box)


                    # print(len(box_pc))
                    # 将box_pc存入o3d的pcd
                    box_pcd.points = o3d.utility.Vector3dVector(box_pc)

                    # 点云过滤
                    box_pcd = converter.filter(box_pcd)

                    # 获取box_pcd的中心点
                    cluster_result = converter.cluster(box_pcd)

                    # 如果聚类结果为空，则跳过，TODO：则用距离中值点取点
                    if cluster_result is None:
                        continue
                    _, center = cluster_result
                    # 取box_pcd 的距离中值点
                    # center = converter.get_center_mid_distance(box_pcd)


                    # 计算距离
                    distance = converter.get_distance(center)
                    print("xyz:",center,"distance:",distance)

                    # 将点转到赛场坐标系下 ， 此处未完成，返回的是[]
                    field_xyz = converter.camera_to_field(center)

                    # 在图像上写距离,位置为xyxy_box的左上角,可以去掉
                    cv2.putText(result_img, "distance: {:.2f}".format(distance), (int(xyxy_box[0]), int(xyxy_box[1]),), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 122), 2)
                    cv2.putText(result_img, "x: {:.2f}".format(center[0])+"y:{:.2f}".format(center[1])+"z:{:.2f}".format(center[2]), (int(xyxy_box[2]), int(xyxy_box[3]+10),),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 122), 2)

                    # 将结果打包
                    # carList_results , # result in carList_results: [track_id , car_id , xywh , conf ,  camera_xyz , filed_xyz ]
                    carList_results.append([track_id , carList.get_car_id(label) , xywh_box , 1 , center , field_xyz])

                # 将结果传入carList
                carList.update_car_info(carList_results)


        # 我方颜色
        my_color = carList.my_color
        # print(my_color)

        all_infos = carList.get_all_info()
        my_car_infos = []
        enemy_car_infos = []
        # result in results:[car_id , center_xy , camera_xyz , field_xyz]
        # 如果是我方车辆，找到所有敌方车辆，计算与每一台敌方车辆距离，并在图像两车辆中心点之间画线，线上写距离
        for all_info in all_infos:
            car_id , center_xy , camera_xyz , field_xyz , color = all_info
            print("car_id:",car_id,"center_xy:",center_xy,"camera_xyz:",camera_xyz,"field_xyz:",field_xyz ,"color:",color)
            print("mc",my_color,"mc","c",color,"c")
            # 将信息分两个列表存储
            if color == my_color:
                #print("same")
                my_car_infos.append(all_info)
            else:
                #print("dif")
                enemy_car_infos.append(all_info)

        # 画线
        for my_car_info in my_car_infos:
            my_car_id , my_center_xy , my_camera_xyz , my_field_xyz , my_color = my_car_info
            for enemy_car_info in enemy_car_infos:
                enemy_car_id , enemy_center_xy , enemy_camera_xyz , enemy_field_xyz , enemy_color = enemy_car_info
                # 计算距离
                distance = np.linalg.norm(np.array(my_field_xyz) - np.array(enemy_field_xyz))
                # 画线
                cv2.line(result_img, (int(my_center_xy[0]), int(my_center_xy[1])), (int(enemy_center_xy[0]), int(enemy_center_xy[1])), (0, 255, 122), 2)
                # 写距离
                cv2.putText(result_img, "distance: {:.2f}".format(distance), (int((my_center_xy[0] + enemy_center_xy[0]) / 2), int((my_center_xy[1] + enemy_center_xy[1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 122), 2)





        if result_img is None:
            continue

        cv2.putText(result_img, "fps: {:.2f}".format(avg_fps), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122),
                    2)

        result_img = cv2.resize(result_img, (1920, 1280))
        if save_video:
            out.write(result_img)
        cv2.imshow("frame", result_img) # 不加3帧
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    if save_video:
        out.release()
    lidar.stop()

    cv2.destroyAllWindows()