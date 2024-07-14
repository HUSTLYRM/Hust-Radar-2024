import time

class Tools:
    @staticmethod
    def get_time_stamp():
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())