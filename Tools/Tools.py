import time

class Tools:
    @staticmethod
    def get_time_stamp():
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    @staticmethod
    # 帧率控制，传入帧率和上一帧的时间戳，自动进行sleep
    def frame_control(fps, last_time):
        current_time = time.time()
        sleep_time = 1 / fps - (current_time - last_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        return time.time()  # 返回当前时间戳 , last_time = frame_control(fps, last_time)