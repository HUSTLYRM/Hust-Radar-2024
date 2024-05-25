from communication.Sender import Sender
from communication.Receiver import Receiver
import threading
import time
class Messager:
    def __init__(self , cfg):
        # 发送部分
        self.sender = Sender(cfg)
        self.double_effect_times = 0 # 第几次发送双倍易伤效果决策,第一次发送值为1，第二次发送值为2，每局最多只能发送到2,不能发送3

        # 特殊flag
        self.super_flag = cfg['others']['is_vs_qd'] # 打青岛大学的特殊flag
        self.is_vs_hg = cfg['others']['is_vs_hg']

        # 接收部分
        self.receiver = Receiver(cfg)

        # 数据存储
        self.enemy_car_infos = [] # 敌方车辆信息，enemy_car_id , enemy_center_xy , enemy_camera_xyz , enemy_field_xyz , enemy_color = enemy_car_info
        self.sentinel_alert_info = [] # 哨兵预警信息，匹配sender的generate_sentinel_alert_info(self , carID , distance , quadrant):
        self.time_left = -1 #剩余时间
        # 线程锁
        self.map_lock = threading.Lock() # 小地图敌方车辆信息锁
        self.sentinel_lock = threading.Lock() # 哨兵预警信息锁
        # 次数记录
        self.hero_enter_times = 0

        # 线程
        self.threading = threading.Thread(target=self.main_loop , daemon=True)
        # 时间记录
        self.last_send_double_effect_time = time.time()
        self.last_send_map_time = time.time()
        self.last_update_time_left_time = time.time()
        self.last_main_loop_time = time.time()
        self.first_big_buff_send = False
        self.second_big_buff_send = False
        # 创建一个1-5,7,101-105,107的上次发送时间的字典
        if self.sender.my_color == "Blue":
            self.last_send_time_map = {1:time.time(),2:time.time(),3:time.time(),4:time.time(),5:time.time(),7:time.time()}
        elif self.sender.my_color == "Red":
            self.last_send_time_map = {101:time.time(),102:time.time(),103:time.time(),104:time.time(),105:time.time(),107:time.time()}
        else:
            print("color error , check upper character")
            exit(0)





        # event
        # self.send_double_effect_decision_event = threading.Event()

        # flag
        self.working_flag = False



    # 开启线程
    def start(self):
        self.working_flag = True
        self.threading.start()

    # 关闭线程
    def stop(self):
        self.working_flag = False
        # self.threading.join()

    # 更新剩余时间
    # def update_time_left(self):
    #     self.time_left = self.receiver.get_time_left()

    # 更新敌方车辆信息
    def update_enemy_car_infos(self , enemy_car_infos):
        # 如果为空，直接返回
        with self.map_lock:
            self.enemy_car_infos = enemy_car_infos

    # 更新哨兵预警信息
    def update_sentinel_alert_info(self , sentinel_alert_info):
        with self.sentinel_lock:
            # print("update",sentinel_alert_info)
            self.sentinel_alert_info = sentinel_alert_info


    # 发送敌方车辆位置
    def send_map(self, carID , x , y):
        self.sender.send_enemy_location(carID , x , y)
        #

    # 发送哨兵预警角信息
    def send_sentry_alert_angle(self):

        sentinel_alert_info = []
        with self.sentinel_lock:
            sentinel_alert_info = self.sentinel_alert_info
        # print("alert info",sentinel_alert_info)
        if sentinel_alert_info == []:
            return
        carID , distance , quadrant = sentinel_alert_info
        self.sender.send_sentinel_alert_info(carID , distance , quadrant)
        # print("send_sentinel_alert_info")

    # 发送自主决策信息
    def send_double_effect_decision(self):
        # 如果距离上次发送时间小于1s，不发送,time()的单位是s
        if time.time() - self.last_send_double_effect_time < 1 or self.time_left == -1 :
            # print("send_double_pass")
            return
        # 手动触发，如果receiver里的self.send_double_flag为1，发送一次，如果为2，再发送一次
        if self.receiver.send_double_flag == self.double_effect_times+1:
            self.sender.send_radar_double_effect_info(self.double_effect_times+1)
            time.sleep(0.1)
            self.sender.send_radar_double_effect_info(self.double_effect_times+1)
            self.double_effect_times += 1
            self.last_send_double_effect_time = time.time()


        if self.time_left <= 160 and self.time_left > 100 and not self.first_big_buff_send:
            self.sender.send_radar_double_effect_info(self.double_effect_times + 1)
            time.sleep(0.1)
            self.sender.send_radar_double_effect_info(self.double_effect_times + 1)
            self.double_effect_times += 1
            self.last_send_double_effect_time = time.time()
            self.first_big_buff_send = True
        elif self.time_left <= 85  and not self.second_big_buff_send:
            self.sender.send_radar_double_effect_info(self.double_effect_times + 1)
            time.sleep(0.1)
            self.sender.send_radar_double_effect_info(self.double_effect_times + 1)
            self.double_effect_times += 1
            self.last_send_double_effect_time = time.time()
            self.second_big_buff_send = True
        if self.double_effect_times >= 2:
            self.double_effect_times = 1

    # 更新剩余时间
    def update_time_left(self):
        if time.time() - self.last_update_time_left_time < 1:
            return
        self.time_left = self.receiver.get_time_left()
        # print("update")
        self.last_update_time_left_time = time.time()



    # 根据时间发送自主决策信息


    # 线程主函数
    def main_loop(self):
        # 问题出在这里，阻塞导致效率很低
        self.receiver.start()


        while True:

            if not self.working_flag:
                # print("messager stop")
                break
            # print("time left",self.time_left)
            time_interval_cal = time.time() - self.last_main_loop_time
            if time_interval_cal < 0.1:
                time.sleep(0.1 - time_interval_cal)
            self.last_main_loop_time = time.time()

            # 更新剩余时间
            self.update_time_left()


            # 发送自主决策信息
            # self.send_double_effect_decision()

            # 发送哨兵通信信息
            self.send_sentry_alert_angle()

            # 提取enemy_car_infos
            enemy_car_infos = self.enemy_car_infos
            # print("ready to send map info",enemy_car_infos)

            # 不可信的车辆信息，也发送，但是要控制在0.45s发送一次，先记录，最后再统一发送
            # 不可信的车辆信息
            not_valid_enemy_car_infos = []


            for enemy_car_info in enemy_car_infos:
                # 提取car_id和field_xyz
                track_id , car_id , field_xyz , is_valid = enemy_car_info[0] ,enemy_car_info[1] , enemy_car_info[4] , enemy_car_info[6]
                if track_id == -1: # 如果track_id为-1，说明没有检测到，不发送
                    # print("not init continue")
                    continue


                # 将所有信息打印
                # print("car_id:",car_id , "field_xyz:",field_xyz , "is_valid:",is_valid)
                # 提取x和y
                x , y = field_xyz[0] , field_xyz[1]
                # x的控制边界，让他在[0,28]m , y控制在[0,15]m
                x = max(0,x)
                x = min(x,28)
                y= min(15,y)
                y = max(0,y)
                # 控制send_map的通信频率在10Hz
                time_interval = time.time() - self.last_send_map_time
                # print("time_interval:", time_interval)
                if is_valid:
                    if  time_interval < 0.1: # 如果时间间隔小于0.1s，等待至0.1s
                        # print("sleep time:",0.1 - time_interval)
                        time.sleep(0.1 - time_interval)
                else:
                    if time.time() - self.last_send_time_map[car_id] < 0.40:
                        # print("not valid ,sleep", 0.45 - (time.time() - self.last_send_time_map[car_id]) )
                        continue
                if self.super_flag:
                    if car_id == 1 or car_id == 101 and not is_valid:
                        self.send_map(car_id, 16.36, 1.80)
                        # print("send_map" ,  car_id , x , y)
                        # 更新时间
                        self.last_send_map_time = time.time()
                        self.last_send_time_map[car_id] = time.time()
                        continue
                if self.is_vs_hg:
                    if car_id == 107 and not is_valid :
                        self.send_map(car_id , 22 , 6)
                        self.last_send_map_time = time.time()
                        self.last_send_time_map[car_id] = time.time()
                        continue
                # 发送
                self.send_map(car_id , x , y)
                # print("send_map" ,  car_id , x , y)
                # 更新时间
                self.last_send_map_time = time.time()
                self.last_send_time_map[car_id] = time.time()
                if (car_id == 1 or car_id == 101 )  and is_valid: # 硬编码,要改
                    if time.time() - self.last_send_double_effect_time < 10:
                        continue
                    if 24>=x>=17:
                        self.hero_enter_times += 1
                    else:
                        self.hero_enter_times = 0

                    if self.hero_enter_times >= 3:
                        print("send double")
                        # self.sender.send_radar_double_effect_info(1)
                        # self.double_effect_times += 1
                        time.sleep(0.01)
                        self.sender.send_radar_double_effect_info(2)

                        time.sleep(0.01)
                        self.sender.send_radar_double_effect_info(1)
                        # self.double_effect_times += 1
                        self.last_send_double_effect_time = time.time()
            # # 发送不可信的车辆信息
            # for enemy_car_info in not_valid_enemy_car_infos:
            #     car_id , field_xyz = enemy_car_info[0] , enemy_car_info[3]
            #     x , y = field_xyz[0] , field_xyz[1]
            #     x = max(0,x)
            #     x = min(x,28)
            #     y= min(15,y)
            #     y = max(0,y)
            #     # 因为红方车辆id为1-5，7，蓝方为101-105，107，所以整除100，取余数，就可以得到1-5，7 , 再减1，就可以得到0-5
            #     time_interval = time.time() - self.last_send_time_list[(car_id % 100)-1]
            #     if  time_interval < 0.35:
            #         time.sleep(0.35 - time_interval)
            #     self.send_map(car_id , x , y)
            #     self.last_send_time_list[(car_id % 100)-1] = time.time()

        self.receiver.stop()





