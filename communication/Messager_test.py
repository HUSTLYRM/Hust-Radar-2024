# -*- coding: utf-8 -*-
import copy
import sys
import os

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

from Messager import Messager
from ruamel.yaml import YAML
# from Lidar.Converter import Converter
from collections import deque
import serial
from Receiver import Receiver
'''
    def get_sentinel_patrol_area_field_xyz(self , my_color):
        # 传入颜色，返回预设的哨兵巡逻区的赛场坐标系的坐标
        if my_color == 'Red':
            return [22.63,9.42,0.5] # 蓝方哨兵巡逻区赛场中心坐标，
        else:
            return [5.68,6,54,0.5] # 红方哨兵巡逻区赛场中心坐标，

    def get_hero_highland_area_field_xyz(self , my_color):
        # 传入颜色，返回预设的英雄梯高区的赛场坐标系的坐标
        if my_color == 'Red':
            return [23.10,2.76,1]
        else:
            return [5.22,13.20,1]

'''
if __name__ == '__main__':
    import random
    main_config_path = "../configs/main_config.yaml"
    main_cfg = YAML().load(open(main_config_path, encoding='Utf-8', mode='r'))

    image = cv2.imread("/home/nvidia/RadarWorkspace/code/Radar_Develop/data/map.jpg")


    # receiver = Receiver(main_cfg)
    # receiver.start()

    draw_queue = deque(maxlen=10)
    messager = Messager(main_cfg , draw_queue = draw_queue)
    # converter = Converter(main_cfg)
    messager.start()
    while True:
        # 模拟一个敌方车辆信息 , y在2-15 ，x在10-20 , y间隔0.1 ， x间隔0.2
        # enemy_car_infos = [(1, 7, (1, 1), (1, 1, 1), filed_xyz_sentinel_red, 1, True)]
        # time.sleep(0.01)
        # enemy_car_infos = [(1, 107, (1, 1), (1, 1, 1), filed_xyz_sentinel_red, 1, True)]
        # enemy_car_infos = [(1, 7, (1, 1), (1, 1, 1), filed_xyz_sentinel_red, 1, True)]
        # messager.update_enemy_car_infos(enemy_car_infos)
        # time.sleep(0.1)
        for x in range(10, 20):
            # messager.sender.send_hero_alert_info(is_alert=True)
            # messager.sender.send_double_effect_times_to_car(107,1)
            for y in range(2, 15):
                # messager.sender.send_hero_alert_info(is_alert=True)
                # ime.sleep(0.1)
                # messager.hero_is_dead = False
                # messager.sender.send_hero_alert_info(is_alert=False)
                # enemy_car_infos = [(1,1, (1, 1), (1, 1, 1), (17.38 ,  10.32 , 1), 1 ,True)]
                # messager.update_enemy_car_infos(enemy_car_infos)
                # time.sleep(0.01)
                #
                # filed_xyz_sentinel_blue = [22.63,9.42,0.5] # 蓝方哨兵巡逻区赛场中心坐标，
                filed_xyz_sentinel_red = [5.68,6,54,0.5] # 红方哨兵巡逻区赛场中心坐标，
                filed_xyz_hero_highland_red = [23.10,2.76,1]
                # filed_xyz_hero_highland_blue = [5.22,13.20,1]
                # #
                # enemy_car_infos = [(1, 2, (1, 1), (1, 1, 1), filed_xyz_hero_highland_blue, 1, True)]
                # messager.update_enemy_car_infos(enemy_car_infos)
                # time.sleep(0.01)
                # #
                # enemy_car_infos = [(1, 3, (1, 1), (1, 1, 1), filed_xyz_hero_highland_red, 1, True)]
                # messager.update_enemy_car_infos(enemy_car_infos)
                # time.sleep(0.01)
                # #
                # enemy_car_infos = [(1, 4, (1, 1), (1, 1, 1), filed_xyz_sentinel_blue, 1, True)]
                # messager.update_enemy_car_infos(enemy_car_infos)
                # time.sleep(0.01)
                # #
                # enemy_car_infos = [(1, 7, (1, 1), (1, 1, 1), filed_xyz_sentinel_red, 1, True)]
                # messager.update_enemy_car_infos(enemy_car_infos)
                # time.sleep(0.01)
                messager.auto_send_double_effect_decision()
                # # essager.hero_is_dead = False
                # # enemy_car_infos = [(1, 3, (1, 1), (1, 1, 1), (x+1, y, 1), 1, True)]
                # # messager.update_enemy_car_infos(enemy_car_infos)
                time.sleep(0.1)
                enemy_car_infos = [(1, 7, (1, 1), (1, 1, 1), (5.72, 7.12), 1, True)]
                messager.update_enemy_car_infos(enemy_car_infos)
                # messager.sender.send_double_effect_times_to_car(107,1)
                time.sleep(0.1)
                # time.sleep(0.01)
                # # messager.hero_is_dead = False
                # # send_info = ([1,1],[1,1],[1,1],[1,1],[1,1],[1,1])
                # # messager.sender.send_enemy_location(send_info)      # print("time left",messager.time_left)
                # # messager.sender.send_radar_double_effect_info(messager.already_activate_double_effect_times + 1)
                # print("拥有双倍易伤次数" , messager.have_double_effect_times)
                # print("已触发双倍易伤次数",messager.already_activate_double_effect_times)
                # print("正在触发双倍易伤:" , bool(messager.is_activating_double_effect))
                # print("飞镖目标" , messager.dart_target)
                # print("剩余时间" , messager.time_left)
                # print("标记进度" , messager.mark_progress)
                # print("标记个数" , messager.marked_num)
                # print("车辆血量" , messager.enemy_health_info)
                print("预警英雄" , messager.is_alert_hero)

                # 清空前面 print的输出
                print("\r",end="")


                # jprint("blood",messager.enemy_health_info)
                # how_image = copy.deepcopy(image)
                # messager.hero_alert(image=show_image)
                # time.sleep(0.1)
        # time.sleep(5)
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
