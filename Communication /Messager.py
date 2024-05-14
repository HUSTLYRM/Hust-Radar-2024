from Sender import Sender
from Receiver import Receiver
import threading
import time
class Message:
    def __init__(self , cfg):
        # 发送部分
        self.sender = Sender(cfg)
        self.double_effect_times = 1 # 第几次发送双倍易伤效果决策,第一次发送值为1，第二次发送值为2，每局最多只能发送到2,不能发送3

        # 接收部分
        self.receiver = Receiver(cfg)

        # 数据存储
        self.enemy_car_infos = [] # 敌方车辆信息，enemy_car_id , enemy_center_xy , enemy_camera_xyz , enemy_field_xyz , enemy_color = enemy_car_info
        self.time_left = -1 #剩余时间
        # 线程锁
        self.lock = threading.Lock()
        # 时间记录
        self.last_send_double_effect_time = time.time()
        self.last_send_map_time = time.time()

        # event
        self.send_double_effect_decision_event = threading.Event()

    # 更新敌方车辆信息
    def update_enemy_car_infos(self , enemy_car_infos):
        with self.lock:
            self.enemy_car_infos = enemy_car_infos

    # 发送敌方车辆位置
    def send_map(self, carID , x , y):
        self.sender.send_enemy_location(carID , x , y)

    # 发送自主决策信息
    def send_double_effect_decision(self):
        # 如果距离上次发送时间小于1s，不发送,time()的单位是s
        if time.time() - self.last_send_double_effect_time < 1:
            return

        if self.time_left > 160: # 还剩3分钟第一次大符，开始打大符20s后才开始发送
            return
        elif self.time_left <= 160 and self.time_left > 85:
            self.send_double_effect_decision(1)
            self.last_send_double_effect_time = time.time()
        elif self.time_left <= 85:
            self.send_double_effect_decision(2)
            self.last_send_double_effect_time = time.time()



    # 根据时间发送自主决策信息


    # 线程主函数
    def main_thread(self):

        while True:

            # 发送自主决策信息
            self.send_double_effect_decision()

            # 提取enemy_car_infos

            with self.lock:
                enemy_car_infos = self.enemy_car_infos

            for enemy_car_info in enemy_car_infos:
                # 提取car_id和field_xyz
                car_id , field_xyz = enemy_car_info[0] , enemy_car_info[3]
                # 提取x和y
                x , y = field_xyz[0] , field_xyz[1]
                # 发送
                self.send_map(car_id , x , y)





