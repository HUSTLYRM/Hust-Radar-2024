import time

class Tools:
    @staticmethod
    def get_time_stamp():
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    @staticmethod
    # 帧率控制，传入帧率和上一帧的时间戳，自动进行sleep
    def frame_control_sleep(fps, last_time):
        '''

        :param fps: 期望控制帧率
        :param last_time: 上次执行的时间戳
        :return: 本次执行的时间戳，用于下次调用，用例：last_time = frame_control(fps, last_time)
        '''
        current_time = time.time()
        sleep_time = 1 / fps - (current_time - last_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        return time.time()  # 返回当前时间戳 , last_time = frame_control(fps, last_time)

    @staticmethod
    def frame_control_skip(fps, last_time):
        '''

        :param fps: 期望控制帧率
        :param last_time: 上次执行的时间戳
        :return: 是否跳过本次执行，用例：skip, last_time = frame_control(fps, last_time)
        if skip:
            continue
        '''
        if time.time() - last_time < 1 / fps:
            return True , last_time
        else:
            # 传入的last_time是一个不可变对象（如整数，字符串，元组），所以是副本，不会改变原来的值，需要返回新的时间戳
            return False , time.time()