# -*- coding: utf-8 -*-
import copy
import sys
import os

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

from Messager import Messager
from ruamel.yaml import YAML
import serial
from Receiver import Receiver

if __name__ == '__main__':
    import random
    main_config_path = "../configs/main_config.yaml"
    main_cfg = YAML().load(open(main_config_path, encoding='Utf-8', mode='r'))

    image = cv2.imread("/home/nvidia/RadarWorkspace/code/Radar_Develop/data/map.jpg")


    # receiver = Receiver(main_cfg)
    # receiver.start()


    messager = Messager(main_cfg)
    messager.start()
    while True:
        # 模拟一个敌方车辆信息 , y在2-15 ，x在10-20 , y间隔0.1 ， x间隔0.2
        for y in range(2, 15):
            for x in range(10 , 20):

                enemy_car_infos = [(1,1, (1, 1), (1, 1, 1), (x, y , 1), 1 ,True)]
                messager.update_enemy_car_infos(enemy_car_infos)
                show_image = copy.deepcopy(image)
                messager.show_areas(image=show_image)
                time.sleep(0.1)

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
