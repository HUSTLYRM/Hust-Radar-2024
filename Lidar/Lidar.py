# 构建Lidar类，作为激光雷达接收类，构建一个ros节点持续订阅/livox/lidar话题，把点云信息写入PcdQueue,整个以子线程形式运行
from .PointCloud import *
import threading
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


class Lidar:
    def __init__(self,cfg):
        # 标志位
        self.flag = False  # 激光雷达接收启动标志
        self.init_flag = False # 激光雷达接收线程初始化标志
        self.working_flag = False  # 激光雷达接收线程启动标志
        self.threading = None  # 激光雷达接收子线程
        self.stop_event = threading.Event()  # 线程停止事件


        # 参数
        self.height_threshold = cfg["lidar"]["height_threshold"]  # 自身高度，用于去除地面点云
        self.min_distance = cfg["lidar"]["min_distance"]  # 最近距离，距离小于这个范围的不要
        self.max_distance = cfg["lidar"]["max_distance"]  # 最远距离，距离大于这个范围的不要
        self.lidar_topic_name = cfg["lidar"]["lidar_topic_name"] # 激光雷达话题名

        # 点云队列
        self.pcdQueue = PcdQueue(max_size=10) # 将激光雷达接收的点云存入点云队列中，读写上锁？

        # 激光雷达线程
        self.lock = threading.Lock()  # 线程锁

        if not self.init_flag:
            # 当雷达还未有一个对象时，初始化接收节点
            self.listener_begin(self.lidar_topic_name)
            # print("listener_begin")
            self.init_flag = True
            self.threading = threading.Thread(target=self.main_loop, daemon=True)


    # 线程启动
    def start(self):
        '''
        开始子线程，即开始spin
        '''
        if not self.working_flag:
            self.working_flag = True
            self.threading.start()

            # print("start@")

    # 线程关闭
    def stop(self):
        '''
        结束子线程
        '''
        if self.working_flag and self.threading is not None: # 关闭时写错了之前，写成了if not self.working_flag
            self.stop_event.set()
            rospy.signal_shutdown('Stop requested')
            self.working_flag = False
            print("stop")

    # 安全关闭子线程
    # def _async_raise(self,tid, exctype):
    #     """raises the exception, performs cleanup if needed"""
    #     tid = ctypes.c_long(tid)
    #     if not inspect.isclass(exctype):
    #         exctype = type(exctype)
    #     res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    #     if res == 0:
    #         raise ValueError("invalid thread id")
    #     elif res != 1:
    #         # """if it returns a number greater than one, you're in trouble,
    #         # and you should call it again with exc=NULL to revert the effect"""
    #         ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
    #         raise SystemError("PyThreadState_SetAsyncExc failed")
    #
    # # 停止线程，中间方法
    # def stop_thread(self,thread):
    #     self._async_raise(thread.ident, SystemExit)


    # 节点启动
    def listener_begin(self,laser_node_name="/livox/lidar"):
        rospy.init_node('laser_listener', anonymous=True)
        # print("init ")
        rospy.Subscriber(laser_node_name, PointCloud2, self.callback)
        # print("sub")

    # 订阅节点子线程
    def main_loop(self):
        # 通过将spin放入子线程来防止其对主线程的阻塞
        rospy.spin()



    def callback(self,data):
        '''
        子线程函数，对于/livox/lidar topic数据的处理 , data是传入的
        '''

        if self.stop_event.is_set():
            print("stop is set")
            return
        if self.working_flag:

            # 获取点云
            pc = np.float32(point_cloud2.read_points_list(data, field_names=("x", "y", "z"), skip_nans=True)).reshape(
                -1, 3)

            # 过滤点云
            dist = np.linalg.norm(pc, axis=1)  # 计算点云距离

            pc = pc[dist > self.min_distance]  # 雷达近距离滤除
            # 第二次需要重新计算距离，否则会出现维度不匹配
            dist = np.linalg.norm(pc, axis=1)  # 计算点云距离
            pc = pc[dist < self.max_distance]  # 雷达远距离滤除

            # 如果在地面+5cm以上，才保留，在地面的点为-height_threshold,
            # pc = pc[pc[:, 2] > (-1 * self.height_threshold)]


            with self.lock:
                # 存入点云队列
                self.pcdQueue.add(pc)

    # 获取所有点云
    def get_all_pc(self):
        with self.lock:
            return self.pcdQueue.get_all_pc()


    # del
    def __del__(self):
        self.stop()

