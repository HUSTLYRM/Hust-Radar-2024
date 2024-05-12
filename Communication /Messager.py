class Message:
    # 静态变量声明
    CRC8_TABLE = [
        # 这里是你的CRC8数据
    ]

    CRC16_TABLE = [
        # 这里是你的CRC16数据
    ]

    # 静态方法声明
    @staticmethod
    def get_crc16_check_byte(data):
        crc = 0xffff
        for byte in data:
            crc = ((crc >> 8) ^ Messager .CRC16_TABLE[(crc ^ byte & 0xff) & 0xff])
        return crc

    @staticmethod
    def get_crc8_check_byte(data):
        crc = 0xff
        for byte in data:
            crc_index = crc ^ byte
            crc = Messager.CRC8_TABLE[crc_index]
        return crc
