import os
import time

class RadarLog:
    def __init__(self, logger_name):
        self.logger_name = logger_name
        self.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_dir = '/home/nvidia/RadarWorkspace/code/Radar_Develop/logfile'
        self.log_path = f'{self.log_dir}/{self.logger_name}_{self.timestamp}.log'
        # 确保log目录存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def log(self, message):
        try:
            with open(self.log_path, 'a') as log_file:
                # 为每条日志添加时间戳
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                log_file.write(f'[{timestamp}] {message}\n')
        except Exception as e:
            print(f"Error: {e}")

# 使用示例
if __name__ == "__main__":
    logger = RadarLog("test_example")
    index = 0
    while index < 1000:
        if index % 100 == 0:
            logger.log(f"Processing {index}th image...")
        index += 1