import serial
import struct
import time
import threading
from ruamel.yaml import YAML

class Receiver:
    def __init__(self,cfg):
        # 串口配置
        port_list = list(serial.tools.list_ports.comports())
        port = port_list[0].device
        self.port = port
        self.bps = cfg['communication']['bps']
        self.timex = cfg['communication']['timex']
        self.ser = serial.Serial(self.port, self.bps, timeout=self.timex)
        # CRC表
        self.CRC8_TABLE = [
            0x00, 0x5e, 0xbc, 0xe2, 0x61, 0x3f, 0xdd, 0x83, 0xc2, 0x9c, 0x7e, 0x20, 0xa3, 0xfd, 0x1f, 0x41,
            0x9d, 0xc3, 0x21, 0x7f, 0xfc, 0xa2, 0x40, 0x1e, 0x5f, 0x01, 0xe3, 0xbd, 0x3e, 0x60, 0x82, 0xdc,
            0x23, 0x7d, 0x9f, 0xc1, 0x42, 0x1c, 0xfe, 0xa0, 0xe1, 0xbf, 0x5d, 0x03, 0x80, 0xde, 0x3c, 0x62,
            0xbe, 0xe0, 0x02, 0x5c, 0xdf, 0x81, 0x63, 0x3d, 0x7c, 0x22, 0xc0, 0x9e, 0x1d, 0x43, 0xa1, 0xff,
            0x46, 0x18, 0xfa, 0xa4, 0x27, 0x79, 0x9b, 0xc5, 0x84, 0xda, 0x38, 0x66, 0xe5, 0xbb, 0x59, 0x07,
            0xdb, 0x85, 0x67, 0x39, 0xba, 0xe4, 0x06, 0x58, 0x19, 0x47, 0xa5, 0xfb, 0x78, 0x26, 0xc4, 0x9a,
            0x65, 0x3b, 0xd9, 0x87, 0x04, 0x5a, 0xb8, 0xe6, 0xa7, 0xf9, 0x1b, 0x45, 0xc6, 0x98, 0x7a, 0x24,
            0xf8, 0xa6, 0x44, 0x1a, 0x99, 0xc7, 0x25, 0x7b, 0x3a, 0x64, 0x86, 0xd8, 0x5b, 0x05, 0xe7, 0xb9,
            0x8c, 0xd2, 0x30, 0x6e, 0xed, 0xb3, 0x51, 0x0f, 0x4e, 0x10, 0xf2, 0xac, 0x2f, 0x71, 0x93, 0xcd,
            0x11, 0x4f, 0xad, 0xf3, 0x70, 0x2e, 0xcc, 0x92, 0xd3, 0x8d, 0x6f, 0x31, 0xb2, 0xec, 0x0e, 0x50,
            0xaf, 0xf1, 0x13, 0x4d, 0xce, 0x90, 0x72, 0x2c, 0x6d, 0x33, 0xd1, 0x8f, 0x0c, 0x52, 0xb0, 0xee,
            0x32, 0x6c, 0x8e, 0xd0, 0x53, 0x0d, 0xef, 0xb1, 0xf0, 0xae, 0x4c, 0x12, 0x91, 0xcf, 0x2d, 0x73,
            0xca, 0x94, 0x76, 0x28, 0xab, 0xf5, 0x17, 0x49, 0x08, 0x56, 0xb4, 0xea, 0x69, 0x37, 0xd5, 0x8b,
            0x57, 0x09, 0xeb, 0xb5, 0x36, 0x68, 0x8a, 0xd4, 0x95, 0xcb, 0x29, 0x77, 0xf4, 0xaa, 0x48, 0x16,
            0xe9, 0xb7, 0x55, 0x0b, 0x88, 0xd6, 0x34, 0x6a, 0x2b, 0x75, 0x97, 0xc9, 0x4a, 0x14, 0xf6, 0xa8,
            0x74, 0x2a, 0xc8, 0x96, 0x15, 0x4b, 0xa9, 0xf7, 0xb6, 0xe8, 0x0a, 0x54, 0xd7, 0x89, 0x6b, 0x35,
        ]

        self.CRC16_TABLE = [
            0x0000, 0x1189, 0x2312, 0x329b, 0x4624, 0x57ad, 0x6536, 0x74bf,
            0x8c48, 0x9dc1, 0xaf5a, 0xbed3, 0xca6c, 0xdbe5, 0xe97e, 0xf8f7,
            0x1081, 0x0108, 0x3393, 0x221a, 0x56a5, 0x472c, 0x75b7, 0x643e,
            0x9cc9, 0x8d40, 0xbfdb, 0xae52, 0xdaed, 0xcb64, 0xf9ff, 0xe876,
            0x2102, 0x308b, 0x0210, 0x1399, 0x6726, 0x76af, 0x4434, 0x55bd,
            0xad4a, 0xbcc3, 0x8e58, 0x9fd1, 0xeb6e, 0xfae7, 0xc87c, 0xd9f5,
            0x3183, 0x200a, 0x1291, 0x0318, 0x77a7, 0x662e, 0x54b5, 0x453c,
            0xbdcb, 0xac42, 0x9ed9, 0x8f50, 0xfbef, 0xea66, 0xd8fd, 0xc974,
            0x4204, 0x538d, 0x6116, 0x709f, 0x0420, 0x15a9, 0x2732, 0x36bb,
            0xce4c, 0xdfc5, 0xed5e, 0xfcd7, 0x8868, 0x99e1, 0xab7a, 0xbaf3,
            0x5285, 0x430c, 0x7197, 0x601e, 0x14a1, 0x0528, 0x37b3, 0x263a,
            0xdecd, 0xcf44, 0xfddf, 0xec56, 0x98e9, 0x8960, 0xbbfb, 0xaa72,
            0x6306, 0x728f, 0x4014, 0x519d, 0x2522, 0x34ab, 0x0630, 0x17b9,
            0xef4e, 0xfec7, 0xcc5c, 0xddd5, 0xa96a, 0xb8e3, 0x8a78, 0x9bf1,
            0x7387, 0x620e, 0x5095, 0x411c, 0x35a3, 0x242a, 0x16b1, 0x0738,
            0xffcf, 0xee46, 0xdcdd, 0xcd54, 0xb9eb, 0xa862, 0x9af9, 0x8b70,
            0x8408, 0x9581, 0xa71a, 0xb693, 0xc22c, 0xd3a5, 0xe13e, 0xf0b7,
            0x0840, 0x19c9, 0x2b52, 0x3adb, 0x4e64, 0x5fed, 0x6d76, 0x7cff,
            0x9489, 0x8500, 0xb79b, 0xa612, 0xd2ad, 0xc324, 0xf1bf, 0xe036,
            0x18c1, 0x0948, 0x3bd3, 0x2a5a, 0x5ee5, 0x4f6c, 0x7df7, 0x6c7e,
            0xa50a, 0xb483, 0x8618, 0x9791, 0xe32e, 0xf2a7, 0xc03c, 0xd1b5,
            0x2942, 0x38cb, 0x0a50, 0x1bd9, 0x6f66, 0x7eef, 0x4c74, 0x5dfd,
            0xb58b, 0xa402, 0x9699, 0x8710, 0xf3af, 0xe226, 0xd0bd, 0xc134,
            0x39c3, 0x284a, 0x1ad1, 0x0b58, 0x7fe7, 0x6e6e, 0x5cf5, 0x4d7c,
            0xc60c, 0xd785, 0xe51e, 0xf497, 0x8028, 0x91a1, 0xa33a, 0xb2b3,
            0x4a44, 0x5bcd, 0x6956, 0x78df, 0x0c60, 0x1de9, 0x2f72, 0x3efb,
            0xd68d, 0xc704, 0xf59f, 0xe416, 0x90a9, 0x8120, 0xb3bb, 0xa232,
            0x5ac5, 0x4b4c, 0x79d7, 0x685e, 0x1ce1, 0x0d68, 0x3ff3, 0x2e7a,
            0xe70e, 0xf687, 0xc41c, 0xd595, 0xa12a, 0xb0a3, 0x8238, 0x93b1,
            0x6b46, 0x7acf, 0x4854, 0x59dd, 0x2d62, 0x3ceb, 0x0e70, 0x1ff9,
            0xf78f, 0xe606, 0xd49d, 0xc514, 0xb1ab, 0xa022, 0x92b9, 0x8330,
            0x7bc7, 0x6a4e, 0x58d5, 0x495c, 0x3de3, 0x2c6a, 0x1ef1, 0x0f78,
        ]

        # 数据存储
        self.time_left = -1 # 剩余时间

        # 线程
        self.threading = threading.Thread(target=self.parse_cmd_id, daemon=True)
        self.working_flag = False
        self.last_time_main_loop = time.time() # 保持一秒一帧

    # 线程创建
    # 线程开启
    def start(self):
        self.working_flag = True
        self.threading.start()
    # 线程关闭
    def stop(self):
        self.working_flag = False
        # self.threading.join()
    def get_crc16_check_byte(self,data):
        crc = 0xffff
        for byte in data:
            crc = ((crc >> 8) ^ self.CRC16_TABLE[(crc ^ byte & 0xff) & 0xff])
        return crc

    def get_crc8_check_byte(self,data):
        crc = 0xff
        for byte in data:
            crc_index = crc ^ byte
            crc = self.CRC8_TABLE[crc_index]
        return crc

    # 判断crc8是否正确，传入不含crc8的帧头部分和crc8
    def check_crc8(self,tx_buff,crc8):
        crc8_cal = self.get_crc8_check_byte(tx_buff)
        return crc8 == crc8_cal

    # 判断crc16是否正确，传入不含crc16的帧头部分和crc16
    def check_crc16(self,tx_buff,crc16):
        crc16_cal = self.get_crc16_check_byte(tx_buff)
        return crc16 == crc16_cal

    # 将帧头，cmd_id , data合成用来计算crc16的数据
    def get_tx_buff_to_cal_crc16(self,header,cmd_id,data):
        return header + struct.pack('H',cmd_id) + data

    # 定位帧头
    def find_sof(self):
        # 读取单个字节直至找到SOF
        find_time = 0
        while True:
            # 0.01s如果没有接收就返回空
            byte = self.ser.read()
            # print("finding sof")
            if byte == b'\xA5':
                # print("find SOF")
                return True
            # 整体再挂起0.01s,控制在50HZ
            time.sleep(0.01)
        return False

    # frame_header解析,返回data_length, crc8校验结果
    def parse_frame_header(self):
        # 找到SOF
        if not self.find_sof():
            print("not")
            return False

        # 读取帧头（SOF之后的4字节）
        header = self.ser.read(4)
        data_length, seq, crc8 = struct.unpack('<HBB', header)
        # 把SOF加回去，获得完整帧头
        full_header = struct.pack('B', 165) + header

        # 校验crc8是否正确
        _header = struct.pack('B', 165) + struct.pack('H', data_length) + struct.pack('B', seq)
        if self.check_crc8(_header,crc8):
            return data_length, True , full_header
        else:
            # print("check fail")
            return -1,False , full_header

    # 读取cmd_id
    def read_cmd_id(self):
        cmd_id = self.ser.read(2)
        return struct.unpack('H', cmd_id)[0]

    # 读取data
    def read_data(self,data_length):
        data = self.ser.read(data_length)
        return data

    # 读取frame_tail
    def read_frame_tail(self):
        frame_tail = self.ser.read(2)
        return frame_tail

    # 读取剩余比赛时间 , 返回是否可信，剩余时间
    def read_remaining_time(self):
        # 解析出data_length, 检查CRC8
        data_length, is_valid  , header= self.parse_frame_header()
        if not is_valid:
            print("Frame header CRC8 check failed")
            return False ,0

        # 读取cmd_id
        cmd_id = self.read_cmd_id()

        # 确保是我们需要的命令
        if cmd_id == 0x0001:
            # 1+2+8+2(crc16) , 第2-3字节是uint16_t stage_remain_time;
            remaining_time_data = self.read_data(data_length+2)
            remaining_time = remaining_time_data[1]+remaining_time_data[2]*256
            print(f"Remaining time: {remaining_time}")
            self.read_frame_tail()

            return True , remaining_time
        else:
            # print(f"Unknown cmd_id: {cmd_id}")
            # 514是0x

            return False

    # 解析帧头和cmd_id , 所有的东西都需要保留下来，因为要用来计算crc16
    def parse_cmd_id(self):

        while True:
            # 控制帧率为10fps


            if not self.working_flag:
                print("not")
                break


            if time.time() - self.last_time_main_loop < 0.02: # 主循环控制在50HZ
                time.sleep( 0.02-(time.time() - self.last_time_main_loop))
            self.last_time_main_loop = time.time()


            data_length, is_valid , header = self.parse_frame_header()
            if not is_valid:
                print("Frame header CRC8 check failed")
                continue

            rest_data = self.ser.read(2+data_length+2) # 包括命令码和CRC16

            # 重构帧头+命令码+数据以计算crc16
            tx_buff = header + rest_data[:-2]

            # 计算帧尾是否正确
            frame_tail_ori = rest_data[-2:]


            # if not self.check_crc16(tx_buff , frame_tail_ori):
                # print("CRC16 check failed")
                # continue

            cmd_id = rest_data[:2] # 读取命令码
            data = rest_data[2:-2] # 读取数据

            # 处理不同的命令
            self.switch_method(cmd_id,data)

    # 调用这个函数来处理不同的命令
    def switch_method(self, cmd_id, data):
        cmd_id_value = struct.unpack('<H', cmd_id)[0]
        # print(f"cmd_id: {cmd_id_value}")
        if cmd_id_value == 0x0001: # 比赛进行时间解析
            self.process_game_status(data)
            if time.time() - self.last_time_main_loop < 0.01:
                # print("receiver sleep")
                time.sleep(0.01 - ( time.time() - self.last_time_main_loop))
            self.last_time = time.time()
        elif cmd_id_value == 0x0003:
            self.parse_robot_status(data)
        elif cmd_id_value == 0x020C:
            self.parse_mark_process(data)
        elif cmd_id_value == 0x020E:
            self.parse_double_effect_chance(data)
        # Add conditions for other cmd_ids


    # 比赛进行时间时间解析
    def process_game_status(self,data):
        time_left = data[1]+data[2]*256
        self.time_left = time_left
        print(f"Time left: {time_left}")

    # 获取比赛剩余时间
    def get_time_left(self):
        return self.time_left

    # 己方机器人血量信息
    def parse_robot_status(self,data):
        pass

    # 标记进度
    def parse_mark_process(self,data):
        pass

    # 自主决策信息，是否拥有双倍易伤机会
    def parse_double_effect_chance(self,data):
        pass







    # 找到0x0305
    def parse_0x0305(self):
        # 找到SOF

        data_length , is_valid = self.parse_frame_header()

        if not is_valid:
            return False

        # 读取cmd_id
        cmd_id = self.read_cmd_id()
        if cmd_id[0] != 773:
            return False

        # 读取data
        data = self.read_data(data_length)
        carid =struct.unpack('H',data[:2])
        x= struct.unpack('f', data[2:6])
        y = struct.unpack('f', data[6:])
        print("carId:",carid,"x:",x,"y:",y)
        # 读取frame_tail

        frame_tail = self.read_frame_tail()

        # frame_tail是crc16，校验
        crc16 = struct.unpack('H',frame_tail)
        tx_buff = struct.pack('H', carid) + struct.pack('ff', x, y)


        return True

    # 关闭串口
    def close(self):
        self.ser.close()





def parse_frame(self,serial_port):
    # 找到SOF
    if not self.find_sof(serial_port):
        return False

    # 读取帧头（SOF之后的4字节）
    header = serial_port.read(4)
    # print(header)
    data_length, seq, crc8 = struct.unpack('<HBB', header)
    # print(data_length)
    # print(seq)
    # print(crc8)
    # print(seq)
    # print(crc8)

    # 根据data_length读取data和frame_tail
    data_and_tail = serial_port.read(2+data_length + 2)  # 包括命令码和CRC16

    # 解析出命令码和数据内容
    cmd_id = struct.unpack('H',data_and_tail[:2])
    if cmd_id[0] == 773:
        data = data_and_tail[2:-2]

        carid =struct.unpack('H',data[:2])
        x= struct.unpack('f', data[2:6])
        y = struct.unpack('f', data[6:])
        print("carId:",carid,"x:",x,"y:",y)


    # print(cmd_id)




    return True


# 打开串口

# receiver = Receiver()
# # 读取串口数据
# while True:
#     receiver.read_remaining_time()
#     # time.sleep(0.1)

# 测试

# main_config_path = "../configs/main_config.yaml"
# main_cfg = YAML().load(open(main_config_path, encoding='Utf-8', mode='r'))
# receiver = Receiver(main_cfg)
# receiver.start()
# while True:
#     # print('1')
#     print(receiver.get_time_left())
#     time.sleep(0.1)







