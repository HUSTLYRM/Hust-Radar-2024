
from detect.Detector import Detector
from detect.Video import Video
from detect.Capture import Capture
from Lidar.Lidar import Lidar
from Lidar.Converter import Converter
from Lidar.PointCloud import PcdQueue
from Car import *
import cv2
import time
import open3d as o3d
from collections import deque
from ruamel.yaml import YAML
import os

# 创建一个长度为N的队列

mode = "video" # "video" or "camera"
save_video = False
round = 11 # 练赛第几轮

if __name__ == '__main__':
    video_path = "/home/nvidia/RadarWorkspace/code/Radar_Develop/data/tran_record/0505/ori_data/video2.mp4"
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

    if mode == "video":
        capture = Video(video_path)
    elif mode == "camera":
        capture = Capture(binocular_camera_cfg_path,"new_cam")




    # fps计算
    N = 10
    fps_queue = deque(maxlen=N)
    start_time = time.time()

    # 开启激光雷达线程
    lidar.start()

    # 主循环
    while True:
        # print("main loop")
        # 读取frame
        # 开始计时
        test_time_start = time.time()
        frame = capture.get_frame()
        # 结束计时
        test_time_end = time.time()
        # 计算用时多少ms
        print("time:", (test_time_end - test_time_start) * 1000)

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

        # 读图失败，推出
        if frame is None:
            print("no frame")
            break

        # 目标检测部分
        ori_frame = frame.copy()
        # 获得推理结果=
        infer_result = detector.infer(frame)
        image = ori_frame

        if infer_result is not None:
            result_img ,results = infer_result
            if result_img is not None:
                # 用新图替代
                image = result_img
            if results is not None:
                if lidar.pcdQueue.point_num == 0:
                    continue
                pc_all = lidar.get_all_pc()

                # 创建总体点云pcd
                pcd_all = o3d.geometry.PointCloud()
                pcd_all.points = o3d.utility.Vector3dVector(pc_all)
                # 将总体点云转到相机坐标系下
                converter.lidar_to_camera(pcd_all)
                # 目前为止20帧
                # 待处理的点云
                box_pcd = o3d.geometry.PointCloud()
                # 对每个结果进行分析 , 进行目标定位
                for result in results:
                    # 对每个检测框进行处理，获取对应点云
                    xyxy_box, xywh_box ,  track_id , label = result # xywh的xy是中心点的xy
                    # if label == "NULL":
                    #     continue
                    # print("xyxy",xyxy_box)
                    # print("xywh",xywh_box)

                    # 获取新xyxy_box , 原来是左上角和右下角，现在想要中心点保持不变，宽高设为原来的一半，再计算一个新的xyxy_box
                    div_times = 2
                    new_w = xywh_box[2] / div_times
                    new_h = xywh_box[3] / div_times
                    new_xyxy_box = [xywh_box[0] - new_w / 2, xywh_box[1] - new_h / 2, xywh_box[0] + new_w / 2, xywh_box[1] + new_h / 2]


                    box_pc = converter.get_points_in_box(pcd_all, new_xyxy_box)


                    print(len(box_pc))
                    # 将box_pc转为o3d的pcd
                    box_pcd.points = o3d.utility.Vector3dVector(box_pc)

                    # 过滤
                    box_pcd = converter.filter(box_pcd)
                    # 开始计时
                    # test_time_start = time.time()
                    # ct = converter.get_center_mid_distance(box_pcd)
                    # 结束计时
                    #test_time_end = time.time()
                    # 计算用时多少ms
                    #print("time:", (test_time_end - test_time_start) * 1000, "point_num", len(box_pcd.points))
                    # 获取box_pcd的中心点
                    cluster_result = converter.cluster(box_pcd)

                    if cluster_result is None:
                        continue
                    _, center = cluster_result
                    # 取box_pcd 的距离中值点
                    # center = converter.get_center_mid_distance(box_pcd)


                    # 计算距离
                    distance = converter.get_distance(center)
                    print("xyz:",center,"distance:",distance)
                    # 在图像上写距离,位置为xyxy_box的左上角
                    cv2.putText(result_img, "distance: {:.2f}".format(distance), (int(xyxy_box[0]), int(xyxy_box[1]),), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 122), 2)
                    cv2.putText(result_img, "x: {:.2f}".format(center[0])+"y:{:.2f}".format(center[1])+"z:{:.2f}".format(center[2]), (int(xyxy_box[2]), int(xyxy_box[3]+10),),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 122), 2)






        cv2.putText(image, "fps: {:.2f}".format(avg_fps), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 122),
                    2)

        image = cv2.resize(image, (1920, 1280))
        if save_video:
            out.write(image)
        cv2.imshow("frame", image) # 不加3帧
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    if save_video:
        out.release()
    lidar.stop()

    cv2.destroyAllWindows()