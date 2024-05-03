# 构建Lidar类，作为激光雷达接收类，构建一个ros节点持续订阅/livox/lidar话题，把点云信息写入PcdQueue,整个以子线程形式运行
from PointCloud import *
import threading
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from ruamel.yaml import YAML
import ctypes
import inspect

pcdQueue = PcdQueue()
main_cfg_path = "../configs/main_config.yaml"
main_cfg = YAML().load(open(main_cfg_path, encoding='Utf-8', mode='r'))

class Lidar:
    def __init__(self,cfg):
        # 标志位
        self.flag = False  # 激光雷达接收启动标志
        self.init_flag = False # 激光雷达接收线程初始化标志
        self.working_flag = False  # 激光雷达接收线程启动标志
        self.threading = None  # 激光雷达接收子线程


        # 参数
        self.height_threshold = cfg["lidar"]["height_threshold"]  # 自身高度，用于去除地面点云
        self.min_distance = cfg["lidar"]["min_distance"]  # 最近距离，距离小于这个范围的不要
        self.lidar_topic_name = cfg["lidar"]["lidar_topic_name"] # 激光雷达话题名

        # 点云队列
        self.pcdQueue = PcdQueue() # 将激光雷达接收的点云存入点云队列中，读写上锁？

        # 激光雷达线程
        self.lock = threading.Lock()  # 线程锁

        if not self.init_flag:
            # 当雷达还未有一个对象时，初始化接收节点
            self.listener_begin(self.lidar_topic_name)
            self.init_flag = True
            self.threading = threading.Thread(target=self.main_loop, daemon=True)


    # 线程启动
    def start(self):
        '''
        开始子线程，即开始spin
        '''
        if not self.working_flag:
            self.threading.start()
            self.working_flag = True

    # 线程关闭
    def stop(self):
        '''
        结束子线程
        '''
        if self.working_flag:
            self.stop_thread(self.threading)
            self.working_flag = False

    # 安全关闭子线程
    def _async_raise(self,tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    # 停止线程，中间方法
    def stop_thread(self,thread):
        self._async_raise(thread.ident, SystemExit)


    # 节点启动
    def listener_begin(self,laser_node_name="/livox/lidar"):
        rospy.init_node('laser_listener', anonymous=True)
        rospy.Subscriber(laser_node_name, PointCloud2, self.callback)

    # 订阅节点子线程
    def main_loop(self):
        # 通过将spin放入子线程来防止其对主线程的阻塞
        rospy.spin()
        # 当spin调用时，subscriber就会开始轮询接收所订阅的节点数据，即不断调用callback函数

    def callback(self,data):
        '''
        子线程函数，对于/livox/lidar topic数据的处理 , data是传入的
        '''
        if self.working_flag:
            # 获取点云
            pc = np.float32(point_cloud2.read_points_list(data, field_names=("x", "y", "z"), skip_nans=True)).reshape(
                -1, 3)

            # 过滤点云
            dist = np.linalg.norm(pc, axis=1)  # 计算点云距离
            pc = pc[dist > self.min_distance]  # 雷达近距离滤除
            # 如果在地面+5cm以上，才保留，在地面的点为-height_threshold,
            pc = pc[pc[:, 2] > (-1 * self.height_threshold)]
            with self.lock:
                # 存入点云队列
                self.pcdQueue.add(pc)

    # 获取所有点云
    def get_all_pc(self):
        with self.lock:
            return self.pcdQueue.get_all_pc()



    #



class Radar(object):

    # the global member of the Radar class
    __init_flag = False  # 雷达启动标志
    __working_flag = False  # 雷达接收线程启动标志
    __threading = None  # 雷达接收子线程

    __lock = threading.Lock()  # 线程锁
    __queue = []  # 一个列表，存放雷达类各个对象的Depth Queue

    __record_times = 0  # 已存点云的数量

    __record_list = []

    __record_max_times = 100  # 最大存点云数量

    def __init__(self, K_0, C_0, E_0, queue_size=200, imgsz=(3088, 2064)):
        '''
        雷达处理类，对每个相机都要创建一个对象

        :param K_0:相机内参
        :param C_0:畸变系数
        :param E_0:雷达到相机外参
        :param queue_size:队列最大长度
        :param imgsz:相机图像大小
        '''
        if not Radar.__init_flag:
            # 当雷达还未有一个对象时，初始化接收节点
            Radar.__laser_listener_begin(LIDAR_TOPIC_NAME)
            Radar.__init_flag = True
            Radar.__threading = threading.Thread(target=Radar.__main_loop, daemon=True)
        self._no = len(Radar.__queue)  # 该对象对应于整个雷达对象列表的序号
        self._K_0 = K_0
        self._C_0 = C_0
        Radar.__queue.append(DepthQueue(queue_size, imgsz, K_0, C_0, E_0))

    @staticmethod
    def start():
        '''
        开始子线程，即开始spin
        '''
        if not Radar.__working_flag:
            Radar.__threading.start()
            Radar.__working_flag = True

    @staticmethod
    def stop():
        '''
        结束子线程
        '''
        if Radar.__working_flag:
            stop_thread(Radar.__threading)
            Radar.__working_flag = False

    @staticmethod
    def __callback(data):
        '''
        子线程函数，对于/livox/lidar topic数据的处理
        '''
        if Radar.__working_flag:
            Radar.__lock.acquire()

            pc = np.float32(
                point_cloud2.read_points_list(data, field_names=("x", "y", "z"), skip_nans=True)).reshape(-1, 3)

            dist = np.linalg.norm(pc, axis=1)

            pc = pc[dist > 0.4]  # 雷达近距离滤除
            # do record
            if Radar.__record_times > 0:

                Radar.__record_list.append(pc)
                print("[INFO] recording point cloud {0}/{1}".format(Radar.__record_max_times - Radar.__record_times,
                                                                    Radar.__record_max_times))
                if Radar.__record_times == 1:
                    try:
                        if not os.path.exists(PC_STORE_DIR):
                            os.mkdir(PC_STORE_DIR)
                        with open("{0}/{1}.pkl"
                                          .format(PC_STORE_DIR,
                                                  datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M')),
                                  'wb') as f:
                            pkl.dump(Radar.__record_list, f)
                        Radar.__record_list.clear()
                        print("[INFO] record finished")
                    except:  # 当出现磁盘未挂载等情况，导致文件夹都无法创建
                        print("[ERROR] The point cloud save dir even doesn't exist on this computer!")
                Radar.__record_times -= 1
            # update every class object's queue
            for q in Radar.__queue:
                q.push_back(pc)

            Radar.__lock.release()

    @staticmethod
    def __laser_listener_begin(laser_node_name="/livox/lidar"):
        rospy.init_node('laser_listener', anonymous=True)
        rospy.Subscriber(laser_node_name, PointCloud2, Radar.__callback)

    @staticmethod
    def __main_loop():
        # 通过将spin放入子线程来防止其对主线程的阻塞
        rospy.spin()
        # 当spin调用时，subscriber就会开始轮询接收所订阅的节点数据，即不断调用callback函数

    @staticmethod
    def start_record():
        '''
        开始录制点云
        '''
        if Radar.__record_times == 0:
            Radar.__record_times = Radar.__record_max_times

    def detect_depth(self, rects):
        '''
        接口函数，传入装甲板bounding box返回对应（x0,y0,z_c)值
        ps:这个x0,y0是归一化相机坐标系中值，与下参数中指代bounding box左上方点坐标不同

        :param rects: armor bounding box, format: (x0,y0,w,h)
        '''
        Radar.__lock.acquire()
        # 通过self.no来指定该对象对应的深度队列
        results = Radar.__queue[self._no].detect_depth(rects)
        Radar.__lock.release()
        return results

    def read(self):
        '''
        debug用，返回深度队列当前的深度图
        '''
        Radar.__lock.acquire()
        depth = Radar.__queue[self._no].depth.copy()
        Radar.__lock.release()
        return depth

    def check_radar_init(self):
        '''
        检查该队列绑定队列置位符，来确定雷达是否正常工作
        '''
        if Radar.__queue[self._no].init_flag:
            Radar.__queue[self._no].init_flag = False
            return True
        else:
            return False

    def __del__(self):
        Radar.stop()
