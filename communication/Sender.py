# 定义一个Communicator用于串口通信
import serial
import serial.tools.list_ports
import struct
from ruamel.yaml import YAML
import time
class Sender:
    def __init__(self , cfg ):
        # 我方颜色
        self.my_color = cfg['global']['my_color']

        if self.my_color == 'Red':
            self.my_id = 9
            self.my_sentinel_id = 7
            self.enemy_sentinel_id = 107
        else:
            self.my_id = 109
            self.my_sentinel_id = 107
            self.enemy_sentinel_id = 7

        port_list = list(serial.tools.list_ports.comports())
        port = port_list[1].device
        self.port = port
        self.bps = cfg['communication']['bps']
        self.timex = cfg['communication']['timex']
        # self.SOF = b'\xA5'
        self.SOF = struct.pack('B',0xa5)
        self.seq = 0  # 目前均为单包数据，且无重发机制?
        self.double_effect_times = 0
        self.ser = self.serial_init()
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


    # crc校验
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
    # 串口初始化
    def serial_init(self):

        port_list = list(serial.tools.list_ports.comports())

        if len(port_list) == 0:
            print('无可用串口!')
            # 停止程序
            exit()
        else:
            for i in range(0, len(port_list)):
                print(port_list[i])

        ser = serial.Serial(self.port, self.bps, timeout=self.timex)

        return ser

    # 帧尾获取 , 传入整包数据 , 通用方法
    def get_frame_tail(self , tx_buff):

        CRC16 = self.get_crc16_check_byte(tx_buff)
        frame_tail = bytes([CRC16 & 0x00ff, (CRC16 & 0xff00) >> 8])

        return frame_tail


    # 帧头获取 , 通用方法
    def get_frame_header(self,data_length=14):
        # frame header
        # +--------+--------------+--------+--------+
        # | SOF    | data_length  | seq    | CRC8   |
        # +--------+--------------+--------+--------+
        # | 1-byte | 2-byte       | 1-byte | 1-byte |
        # +--------+--------------+--------+--------+
        #
        # SOF: start of frame, a fixed byte at the beginnig of frame header
        #      the value is 0xA5 in v1.4 protocol
        #      单字节，接收的数据应为 A5
        #
        # data_length: 不包含 cmd_id 和 frame_tail
        #              (construct of a frame:
        #                   [ frame_head  | cmd_id  | data    | frame_tail ]
        #                     5-byte        2-byte    n-byte    2-byte
        #              )
        #              双字节，以data_length=14(10)为例，接收时表现为 0E 00，低字节在前
        #
        # seq: packet sequence number
        #      not used now?
        #
        # struct.py
        # https://docs.python.org/3.8/library/struct.html#struct-format-strings
        # format: struct member type -> size
        # 'H': unsigned short -> 2 bytes
        # 'h': short -> 2 bytes
        # 'B': unsigned char -> 1 byte
        # 'f': float -> 4 bytes
        # 'fff' or '3f' means continuous 3 float values
        # 'I': unsigned int -> 4 bytes
        # global SOF, seq

        _SOF = self.SOF
        _data_length = struct.pack('H', data_length)
        # print(_data_length)
        _seq = struct.pack('B', self.seq)
        _frame_header =  _SOF + _data_length + _seq
        frame_header = _frame_header + struct.pack('B', self.get_crc8_check_byte(_frame_header))
        return frame_header

    # 创建新地方车辆小地图信息，注意单位从m变为了cm
    # 新小地图信息包格式：
    '''
        雷达可通过常规链路向己方所有选手端发送对方机器人的坐标数据，该位置会在己方选手端小地图显示。
    表 3-2 命令码 ID：0x0305
    字节偏移量 大小 说明
    0 2 英雄机器人 x 位置坐标，单位：cm
    2 2 英雄机器人 y 位置坐标，单位：cm
    4 2 工程机器人 x 位置坐标，单位：cm
    6 2 工程机器人 y 位置坐标，单位：cm
    8 2 3 号步兵机器人 x 位置坐标，单位：cm
    10 2 3 号步兵机器人 y 位置坐标，单位：cm
    12 2 4 号步兵机器人 x 位置坐标，单位：cm
    14 2 4 号步兵机器人 y 位置坐标，单位：cm
    16 2 5 号步兵机器人 x 位置坐标，单位：cm
    18 2 5 号步兵机器人 y 位置坐标，单位：cm
    20 2 哨兵机器人 x 位置坐标，单位：cm
    22 2 哨兵机器人 y 位置坐标，单位：cm
    备注
    当 x、y 超出边界时显示在对应边缘处，
    当 x、y 均为 0 时，视为未发送此机器人坐标。
    typedef _packed struct
    {
    uint16_t hero_position_x;
    uint16_t hero_position_y;
    uint16_t engineer_position_x;
    uint16_t engineer_position_y;
    uint16_t infantry_3_position_x;
    uint16_t infantry_3_position_y;
    uint16_t infantry_4_position_x;
    uint16_t infantry_4_position_y;
    uint16_t infantry_5_position_x;
    uint16_t infantry_5_position_y;
    uint16_t sentry_position_x;
    uint16_t sentry_position_y;
    } map_robot_data_t;
    '''
    def generate_enemy_location_info(self , infos): # for info in infos 有6个, info是一个list，里面为[x , y] , 单位为m，需要转换为cm并以uint16_t形式打包
        cmd_id = struct.pack('H', 0x0305)
        # 初始化data
        data = b''
        for info in infos:
            data += struct.pack('HH', int(info[0]*100), int(info[1]*100)) # 单位转换为cm

        data_len = len(data)

        frame_head = self.get_frame_header(data_len)

        tx_buff = frame_head + cmd_id + data

        frame_tail = self.get_frame_tail(tx_buff)

        tx_buff += frame_tail

        return tx_buff



    # 创建敌方车辆小地图信息 , 中间方法
    # def generate_enemy_location_info(self, carID, x, y):
    #
    #     cmd_id = struct.pack('H', 0x0305)
    #
    #     data = struct.pack('H', carID) + struct.pack('ff', x, y)
    #
    #     data_len = len(data)
    #     # print("data_len",data_len)
    #     # print(data_len)
    #     frame_head = self.get_frame_header(data_len)
    #     # print("head:",frame_head)
    #
    #     tx_buff = frame_head + cmd_id + data
    #     # print("tx_buff",len(tx_buff))
    #
    #     frame_tail = self.get_frame_tail(tx_buff)
    #     # print("frame_tail",len(frame_tail))
    #
    #     tx_buff += frame_tail
    #     # print("all",len(tx_buff))
    #     print(len(tx_buff))
    #     return tx_buff

    # 发送Info , 通用方法
    def send_info(self,tx_buff):
        self.ser.write(tx_buff)

    # 发送敌方车辆位置信息 , 调用方法
    def send_enemy_location(self , infos):
        tx_buff = self.generate_enemy_location_info(infos)
        # print("send enemy location",tx_buff)

        self.send_info(tx_buff)



    # 构建机器人交互数据主cmd_id为0x0301,机器人交互数据目前只发给哨兵
    # (1).子内容ID为0x0201时，发送哨兵最近车辆的预警信息，在内容数据段开始，第一个H(unsigned short -> 2 bytes)为车辆ID，同附录，1-5,7为红1-5,7;101-105,107为蓝1-5,7
    # 第二个f(float -> 4 bytes)为距离，单位m,(测试版,如果效果好考虑改为角度，单位度)第三个H(unsigned short -> 2 bytes)为象限,值为0-7,分别对应正方向顺时针-22.5-22.5度,22.5-67.5度,67.5-112.5度,112.5-157.5度,157.5-202.5度,202.5-247.5度,247.5-292.5度,292.5-337.5度
    # (2).子内容ID为0x0202时，发送给哨兵在我的赛场坐标系下哨兵的坐标和所有检测到敌方车辆的赛场坐标系信息
    # 按照 哨兵 ， 敌方1号 ， 敌方2号 ， 敌方3号 ， 敌方4号 ， 敌方5号 ， 敌方7号的顺序发送
    # 每一个车辆的信息为一个B(unsigned char -> 1 byte)为信息是否有效 , 0x00为无效，0x01为有效，一个f(float -> 4 bytes)为x坐标，一个f(float -> 4 bytes)为y坐标
    # (3).子内容ID为0x0203时，发送英雄预警，没来发0x00，来了发0xff
    '''
字节偏移量 大小    说明             备注
0         2    子内容 ID   需为开放的子内容 ID
2         2    发送者 ID   需与自身 ID 匹配，ID 编号详见附录
4         2    接收者 ID   需为规则允许的多机通讯接收者，若接收者为选手端，则仅可发送至发送者对应的选手端，仅限己方通信，ID 编号详见附录，
6         x    内容数据段    x 最大为 112
    子内容 ID  内容数据段长度     功能说明
    0x0200~0x02FF     x≤112      机器人之间通信
    typedef _packed struct{
    uint16_t data_cmd_id;
    uint16_t sender_id;
    uint16_t receiver_id;
    uint8_t user_data[x];
    }robot_interaction_data_t;
    '''
    # (1)组织哨兵预警角信息 , 中间方法
    def generate_sentinel_alert_info(self , carID , distance , quadrant):
        cmd_id = struct.pack('H', 0x0301)
        data_cmd_id = struct.pack('H', 0x0201)
        sender_id = struct.pack('H', self.my_id)
        receiver_id = struct.pack('H', self.my_sentinel_id)
        data = data_cmd_id + sender_id + receiver_id + struct.pack('H', carID) + struct.pack('f', distance) + struct.pack('H', quadrant)
        data_len = len(data)
        frame_head = self.get_frame_header(data_len)

        tx_buff = frame_head + cmd_id + data

        frame_tail = self.get_frame_tail(tx_buff)

        tx_buff += frame_tail

        # print(tx_buff)

        return tx_buff

    # 机器人交互数据0x0301共通部分，后面的方法是data_cmd_id的不同
    def generate_robot_interact_info(self):
        pass

    # (1)发送哨兵预警角信息 , 调用方法
    def send_sentinel_alert_info(self , carID , distance , quadrant):
        tx_buff = self.generate_sentinel_alert_info(carID , distance , quadrant)
        # print(tx_buff)

        self.send_info(tx_buff)

    # (2)组织哨兵赛场坐标信息 , 中间方法 , 传入car_infos , len(car_infos) = 7 ,按顺序组织 , 必须补全7份信息
    # car_info in car_infos: [is_valid , [x , y]]
    def generate_sentinel_field_info(self , car_infos):
        cmd_id = struct.pack('H', 0x0301)
        data_cmd_id = struct.pack('H', 0x0202)
        sender_id = struct.pack('H', self.my_id)
        receiver_id = struct.pack('H', self.my_sentinel_id)
        data = data_cmd_id + sender_id + receiver_id
        for car_info in car_infos:
            if car_info[0]:
                data += struct.pack('B', 0x01) + struct.pack('ff', car_info[1][0], car_info[1][1])
            else:
                data += struct.pack('B', 0x00) + struct.pack('ff', 0, 0)
        data_len = len(data)
        frame_head = self.get_frame_header(data_len)

        tx_buff = frame_head + cmd_id + data

        frame_tail = self.get_frame_tail(tx_buff)

        tx_buff += frame_tail

        return tx_buff

    # (2)发送哨兵赛场坐标信息 , 调用方法 , 传入car_infos , len(car_infos) = 7 ,按顺序组织 , 必须补全7份信息
    # car_info in car_infos: [is_valid , [x , y]] , is_valid对应Car对象的trust属性
    def send_sentinel_field_info(self , car_infos):
        tx_buff = self.generate_sentinel_field_info(car_infos)

        self.send_info(tx_buff)

    # （3）组织雷达自主决策信息,中间方法
    def generate_radar_double_effect_info(self , times = 1):
        cmd_id = struct.pack('H', 0x0301)
        data_cmd_id = struct.pack('H', 0x0121)
        sender_id = struct.pack('H', self.my_id)
        receiver_id = struct.pack('H', 0x8080)
        times_data = struct.pack('H',times)
        data = data_cmd_id + sender_id + receiver_id + times_data

        data_len = len(data)
        frame_head = self.get_frame_header(data_len)

        tx_buff = frame_head + cmd_id + data

        frame_tail = self.get_frame_tail(tx_buff)

        tx_buff += frame_tail

        return tx_buff

    # (3) 发送雷达自主决策信息,调用方法
    def send_radar_double_effect_info(self,times = 1):
        tx_buff =  self.generate_radar_double_effect_info(times)
        print("send double",tx_buff)
        print("send double length",len(tx_buff))
        self.send_info(tx_buff)


    # 发送机器人交互数据0x0301
    '''
    机器人交互数据通过常规链路发送，其数据段包含一个统一的数据段头结构。数据段头结构包括内容 ID、
发送者和接收者的 ID、内容数据段。机器人交互数据包的总长不超过 127 个字节，减去 frame_header、
cmd_id 和 frame_tail 的 9 个字节以及数据段头结构的 6 个字节，故机器人交互数据的内容数据段最大
为 112 个字节。
    每 1000 毫秒，英雄、工程、步兵、空中机器人、飞镖能够接收数据的上限为 3720 字节，雷达和哨兵机器
人能够接收数据的上限为 5120 字节。
    由于存在多个内容 ID，但整个 cmd_id 上行频率最大为 30Hz，请合理安排带宽。
    '''


    # def send_enemy_location(self ,carID, x, y):
    #     tx_buff = self.get_frame_header(10)
    #
    #     # tx_buff += b'\x05\x03'  # cmd_id
    #     u16_num = 0x0305
    #     tx_buff += struct.pack('H', u16_num)
    #     tx_buff += struct.pack('H', carID) + struct.pack('ff', x, y)
    #
    #     CRC16 = self.get_crc16_check_byte(tx_buff)
    #     frame_tail = bytes([CRC16 & 0x00ff, (CRC16 & 0xff00) >> 8])
    #     tx_buff += frame_tail
    #     self.ser.write(tx_buff)
    #
    #     return tx_buff

# if __name__ == '__main__':
#     main_cfg_path = "../configs/main_config.yaml"
#     main_cfg = YAML().load(open(main_cfg_path, 'r'))
#
#     sender = Sender(main_cfg)
#
#     while True:
#         # 循环列表，x在[10,20]之间变化，每次变化0.1 8819.32mm,5706.98mm
#         for i in range(0,3000):
#             sender.send_enemy_location(3,0.1,0.1)
#             time.sleep(0.1)
#             if i <=600:
#                 sender.send_radar_double_effect_info(0)
#             if i > 600:
#                 sender.send_radar_double_effect_info(1)





        # print()
        # comm.send_sentinel_alert_info(101, 1, 1)
        # comm.send_sentinel_field_info([[1, [1, 1]] for _ in range(7)])
