import serial
import struct

class Receiver:
    def __init__(self,portx = '/dev/pts/4'):
        self.ser = serial.Serial(portx, 115200, timeout=0.5)

    # 定位帧头
    def find_sof(self):
        # 读取单个字节直至找到SOF
        while True:
            byte = self.ser.read()
            if byte == b'\xA5':
                return True
        return False

    # frame_header解析,返回data_length, crc8校验结果
    def parse_frame_header(self):
        # 找到SOF
        if not self.find_sof():
            return False

        # 读取帧头（SOF之后的4字节）
        header = self.ser.read(4)
        data_length, seq, crc8 = struct.unpack('<HBB', header)

        # 校验crc8是否正确
        _header = struct.pack('B', 165) + struct.pack('H', data_length) + struct.pack('B', seq)
        crc8_cal = self.get_crc8_check_byte(_header)
        if crc8 == crc8_cal:
            return data_length, True
        else:
            return -1,False






    # 找到0x0305
    def parse_0x0305(self):
        # 找到SOF
        if not self.find_sof():
            return False

        # 读取帧头（SOF之后的4字节）
        header = self.ser.read(4)
        # print(header)
        data_length, seq, crc8 = struct.unpack('<HBB', header)


        # 根据data_length读取data和frame_tail
        data_and_tail = self.ser.read(2+data_length + 2)  # 包括命令码和CRC16

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
ser = serial.Serial('/dev/pts/4', 115200, timeout=0.5)

# 读取串口数据
while True:
    flag = parse_frame(ser)


# 关闭串口
ser.close()