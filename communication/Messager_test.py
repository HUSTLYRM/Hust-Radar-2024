# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

from Messager import Messager
from ruamel.yaml import YAML
import serial
from Receiver import Receiver

if __name__ == '__main__':
    main_config_path = "../configs/main_config.yaml"
    main_cfg = YAML().load(open(main_config_path, encoding='Utf-8', mode='r'))



    # receiver = Receiver(main_cfg)
    # receiver.start()


    messager = Messager(main_cfg)
    messager.start()
    while True:
        # 模拟一个敌方车辆信息

        enemy_car_infos = [(1,1, (1, 1), (1, 1, 1), (18, 0.1 , 1), 1 ,True)]
        messager.update_enemy_car_infos(enemy_car_infos)
        # messager.sender.send_radar_double_effect_info(1;g.)
        # 模拟一个哨兵预警信息
        # for i in range(0,7):
        #     sentinel_alert_info = [101, 1, i]
        #     messager.update_sentinel_alert_info(sentinel_alert_info)
        time.sleep(0.1)
