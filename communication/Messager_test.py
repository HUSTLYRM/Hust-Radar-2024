# -*- coding: utf-8 -*-
import copy
import sys
import os

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

from Messager import Messager
from ruamel.yaml import YAML
from collections import deque
import serial
from Receiver import Receiver

if __name__ == '__main__':
    import random
    main_config_path = "../configs/main_config.yaml"
    main_cfg = YAML().load(open(main_config_path, encoding='Utf-8', mode='r'))

    image = cv2.imread("/home/nvidia/RadarWorkspace/code/Radar_Develop/data/map.jpg")


    # receiver = Receiver(main_cfg)
    # receiver.start()

    draw_queue = deque(maxlen=10)
    messager = Messager(main_cfg , draw_queue = draw_queue)
    messager.start()
    while True:
        # 模拟一个敌方车辆信息 , y在2-15 ，x在10-20 , y间隔0.1 ， x间隔0.2
        for x in range(10, 20):
            messager.sender.send_hero_alert_info(is_alert=True)
            for y in range(2, 15):
                messager.hero_is_dead = False
                messager.sender.send_hero_alert_info(is_alert=False)
                enemy_car_infos = [(1,1, (1, 1), (1, 1, 1), (17.38 ,  10.32 , 1), 1 ,True)]
                messager.update_enemy_car_infos(enemy_car_infos)
                time.sleep(0.01)
                messager.hero_is_dead = False
                # enemy_car_infos = [(1, 3, (1, 1), (1, 1, 1), (x+1, y, 1), 1, True)]
                # messager.update_enemy_car_infos(enemy_car_infos)
                # time.sleep(0.1)
                enemy_car_infos = [(1, 7, (1, 1), (1, 1, 1), (5.72, 7.12), 1, True)]
                messager.update_enemy_car_infos(enemy_car_infos)
                time.sleep(0.1)
                messager.hero_is_dead = False
                # send_info = ([1,1],[1,1],[1,1],[1,1],[1,1],[1,1])
                # messager.sender.send_enemy_location(send_info)      # print("time left",messager.time_left)
                # messager.sender.send_radar_double_effect_info(messager.already_activate_double_effect_times + 1)
                print("拥有双倍易伤次数" , messager.have_double_effect_times)
                print("已触发双倍易伤次数",messager.already_activate_double_effect_times)
                print("正在触发双倍易伤:" , bool(messager.is_activating_double_effect))
                print("飞镖目标" , messager.dart_target)
                # 清空前面 print的输出
                print("\r",end="")


                # jprint("blood",messager.enemy_health_info)
                show_image = copy.deepcopy(image)
                messager.hero_alert(image=show_image)
                # time.sleep(0.1)

        # enemy_car_infos = [(1,1, (1, 1), (1, 1, 1), (x, y , 1), 1 ,True)]
        # messager.update_enemy_car_infos(enemy_car_infos)
        # print(messager.is_in_areas(16,10.8))

        # cv2.waitKey(0)
        # messager.sender.send_radar_double_effect_info(1;g.)
        # 模拟一个哨兵预警信息
        # for i in range(0,7):
        #     sentinel_alert_info = [101, 1, i]
        #     messager.update_sentinel_alert_info(sentinel_alert_info)
        time.sleep(0.1)
