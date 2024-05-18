# 测试通信子线程是否正确运行

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
        enemy_car_infos = [(101, (1, 1), (1, 1, 1), (1, 1, 1), 1)]
        messager.update_enemy_car_infos(enemy_car_infos)
        # 模拟一个哨兵预警信息
        sentinel_alert_info = (1, 1, 1)
        messager.update_sentinel_alert_info(sentinel_alert_info)
        pass