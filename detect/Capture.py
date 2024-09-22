'''
注意，如果连接相机时mvs设置了MVS的相机曝光之类的为continues模式，那么这里的曝光时间设置是无效的，而且会报错，所以需要把自动曝光关掉

'''
import cv2
from ruamel.yaml import YAML
import sys
from ctypes import *
import numpy as np
from stereo_camera.MvImport import MvCameraControl_class as hk


# Capture类的封装
class Capture:
    def __init__(self, binocular_camera_cfg_path = "/home/nvidia/RadarWorkspace/code/Radar_Develop/configs/bin_cam_config.yaml",camera_name = 'new_cam'):
        cfg = YAML().load(open(binocular_camera_cfg_path, encoding='Utf-8', mode='r'))

        self.camera_name = camera_name
        self.camera_id = cfg['id'][self.camera_name]
        self.cfg = cfg
        self.width = cfg['param']['Width']
        self.height = cfg['param']['Height']

        # 初始化相机
        camera, ret = self.get_camera(cfg)  # 使用新的函数
        self.camera = camera
        self.ret = ret
        self.show_width = cfg['param']['show_width']
        self.show_height = cfg['param']['show_height']
        self.pyr_times = cfg['param']['pyr_times']


    # 展示图像
    def show_img(self,img):
        cv2.imshow(self.camera_name, cv2.resize(img, (self.show_width, self.show_height)))
        cv2.waitKey(1)

    # 用图像金字塔展示图像
    def show_img_pyramid(self,img):
        # 复制一份图像
        pyr_img = img.copy()
        for i in range(self.pyr_times):
            pyr_img = cv2.pyrDown(pyr_img)
        cv2.imshow(self.camera_name, pyr_img)
        cv2.waitKey(1)
    # 获取图像
    def get_frame(self):
        frame = hk.MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))
        self.ret.contents = False

        # 读取图像
        _ret = self.camera.MV_CC_GetImageBuffer(frame, 100)
        frame_info = frame.stFrameInfo

        if _ret == hk.MV_OK:
            self.ret.contents = True


            b1 = hk.MVCC_FLOATVALUE()

            self.camera.MV_CC_GetFloatValue('Brightness', b1)
            print(b1.fCurValue)

            # 像素格式转换
            channel_num = 3
            buffer = (c_ubyte * (frame_info.nWidth * frame_info.nHeight * channel_num))()
            size = frame_info.nWidth * frame_info.nHeight * channel_num
            st_convert_param = hk.MV_CC_PIXEL_CONVERT_PARAM()
            st_convert_param.nWidth = frame_info.nWidth  # 图像宽
            st_convert_param.nHeight = frame_info.nHeight  # 图像高
            st_convert_param.pSrcData = frame.pBufAddr  # 输入数据缓存
            st_convert_param.nSrcDataLen = frame_info.nFrameLen  # 输入数据大小
            st_convert_param.enSrcPixelType = frame_info.enPixelType  # 输入像素格式
            st_convert_param.enDstPixelType = hk.PixelType_Gvsp_BGR8_Packed  # 输出像素格式
            st_convert_param.pDstBuffer = buffer  # 输出数据缓存
            st_convert_param.nDstBufferSize = size  # 输出缓存大小
            self.camera.MV_CC_ConvertPixelType(st_convert_param)

            # 转为OpenCV可以处理的numpy数组 (Mat)
            image = np.asarray(buffer).reshape((frame_info.nHeight, frame_info.nWidth, 3))
            self.camera.MV_CC_FreeImageBuffer(frame)

            return image

        else:
            print("[%s] no data[0x%x]" % (self.camera_name, _ret))
            return None

    def get_camera(self,cfg=None):
        # cameras init
        if cfg is None:
            cfg = dict()
        cam = self.camera_init(cfg)

        print(f"{self.camera_name} camera connected")
        # print(cam.MV_CC_IsDeviceConnected()) # 没有这个

        # 设置参数
        self.set_parameters(cam, cfg)

        # 开始取流
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("right 开始取流失败! ret[0x%x]" % ret)
            sys.exit()

        ret_q = POINTER(c_bool)
        ret_q.contents = False

        return cam, ret_q

    def camera_init(self,cfg):
        device_list = hk.MV_CC_DEVICE_INFO_LIST()

        # Enumerate devices (only uses USB camera, so pass the first argument accordingly)
        _ret = hk.MvCamera.MV_CC_EnumDevices(hk.MV_USB_DEVICE, device_list)
        if _ret != 0:
            print("enum devices fail! ret[0x%x]" % _ret)
            sys.exit()

        if device_list.nDeviceNum == 0:
            print("find no device!")
            sys.exit()

        print("Find %d devices!" % device_list.nDeviceNum)

        n_connection_num = -1

        for i in range(0, device_list.nDeviceNum):
            mvcc_dev_info = cast(device_list.pDeviceInfo[i], POINTER(hk.MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == hk.MV_USB_DEVICE:

                # Output camera information
                str_serial_number = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    str_serial_number = str_serial_number + chr(per)

                # Determine left and right cameras based on hardware IDs
                if str_serial_number.endswith(cfg['id'][self.camera_name][-2:]):
                    n_connection_num = i
                    break

        if n_connection_num == -1:
            print("camera not found")
            sys.exit()

        # Create camera instance
        cam = hk.MvCamera()

        # Create handle for camera
        st_device_list = cast(device_list.pDeviceInfo[int(n_connection_num)], POINTER(hk.MV_CC_DEVICE_INFO)).contents
        _ret = cam.MV_CC_CreateHandle(st_device_list)
        if _ret != 0:
            print("create handle fail! ret[0x%x]" % _ret)
            sys.exit()

        # Open device (does not read input stream at this point)
        cam.MV_CC_OpenDevice(hk.MV_ACCESS_Exclusive, 0)

        # Clear any existing buffer (may exist)
        # cam.MV_CC_ClearImageBuffer() # 删除

        return cam

    # set
    def set_parameters(self, _cam=hk.MvCamera(), cfg=None):
        _name = self.camera_name
        # 设置触发模式为on 用于时间同步
        # 尝试设置为off
        if cfg is None:
            cfg = dict()
        _ret = _cam.MV_CC_SetEnumValue("TriggerMode",
                                       hk.MV_TRIGGER_MODE_OFF)  # hk.MV_TRIGGER_MODE_OFF hk.MV_TRIGGER_MODE_ON
        if _ret != 0:
            print("[%s] set trigger mode fail! ret[0x%x]" % (_name, _ret))
            sys.exit()
        # 设置触发源 设置触发源为软触发
        _ret = _cam.MV_CC_SetEnumValue("TriggerSource", hk.MV_TRIGGER_SOURCE_SOFTWARE)
        # LINE2 hk.MV_TRIGGER_SOURCE_SOFTWARE MV_TRIGGER_SOURCE_LINE2
        if _ret != 0:
            print("[%s] set trigger source fail! ret[0x%x]" % (_name, _ret))
            sys.exit()

        # 设置尺寸
        _ret = _cam.MV_CC_SetIntValue("Width", cfg['param']['Width'])
        if _ret != 0:
            print("[%s] set width fail! ret[0x%x]" % (_name, _ret))
            sys.exit()
        _ret = _cam.MV_CC_SetIntValue("Height", cfg['param']['Height'])
        if _ret != 0:
            print("[%s] set height fail! ret[0x%x]" % (_name, _ret))
            sys.exit()

        # 设置垂直偏移 (1280*1024总区域相对于采样范围的上移，0时采样范围大概处于顶端部分)
        _ret = _cam.MV_CC_SetIntValue("OffsetY", cfg['param']['OffsetY'])
        if _ret != 0:
            print("[%s] set offset(y) fail! ret[0x%x]" % (_name, _ret))
            sys.exit()
        """if _name == 'left_camera':
            _ret = _cam.MV_CC_SetIntValue("OffsetX", 64)
            if _ret != 0:
                print("[%s] set offset(y) fail! ret[0x%x]" % (_name, _ret))
                sys.exit()"""

        # 设置gamma矫正
        _ret = _cam.MV_CC_SetBoolValue("GammaEnable", cfg['param']['GammaEnable'])
        if _ret != 0:
            print("[%s] set gammaEnable fail! ret[0x%x]" % (_name, _ret))
            sys.exit()
        _ret = _cam.MV_CC_SetFloatValue("Gamma", cfg['param']['Gamma'])
        if _ret != 0:
            print("[%s] set gamma fail! ret[0x%x]" % (_name, _ret))
            sys.exit()



        # 设置增益
        _ret = _cam.MV_CC_SetFloatValue("Gain", cfg['param']['Gain'])
        if _ret != 0:
            print("[%s] set gamma fail! ret[0x%x]" % (_name, _ret))
            sys.exit()

        # 设置 白平衡，用于图像颜色“不正”时
        _ret = _cam.MV_CC_SetEnumValueByString("BalanceWhiteAuto", cfg['param']['BalanceWhiteAuto'])
        if _ret != 0:
            print("[%s] set balanceWhiteAuto fail! ret[0x%x]" % (_name, _ret))
            sys.exit()
        _ret = _cam.MV_CC_SetEnumValueByString("BalanceRatioSelector", "Red")
        if _ret != 0:
            print("[%s] set balanceWhiteRatio:Red fail! ret[0x%x]" % (_name, _ret))
            sys.exit()
        _ret = _cam.MV_CC_SetIntValue("BalanceRatio", cfg['param']['BalanceRatioR'])
        if _ret != 0:
            print("[%s] set balanceRatio fail! ret[0x%x]" % (_name, _ret))
            sys.exit()
        _ret = _cam.MV_CC_SetEnumValueByString("BalanceRatioSelector", "Green")
        if _ret != 0:
            print("[%s] set balanceWhiteRatio:Green fail! ret[0x%x]" % (_name, _ret))
            sys.exit()
        _ret = _cam.MV_CC_SetIntValue("BalanceRatio", cfg['param']['BalanceRatioG'])
        if _ret != 0:
            print("[%s] set balanceRatio fail! ret[0x%x]" % (_name, _ret))
            sys.exit()
        _ret = _cam.MV_CC_SetEnumValueByString("BalanceRatioSelector", "Blue")
        if _ret != 0:
            print("[%s] set balanceWhiteRatio:Blue fail! ret[0x%x]" % (_name, _ret))
            sys.exit()
        _ret = _cam.MV_CC_SetIntValue("BalanceRatio", cfg['param']['BalanceRatioB'])
        if _ret != 0:
            print("[%s] set balanceRatio fail! ret[0x%x]" % (_name, _ret))
            sys.exit()

        # 设置曝光时间
        _ret = _cam.MV_CC_SetFloatValue("ExposureTime", cfg['param']['ExposureTime'])
        if _ret != 0:
            print("[%s] set exposure fail! ret[0x%x]" % (_name, _ret))
            sys.exit()

    # 关闭相机
    def release(self):
        if self.camera is not None:
            self.camera.MV_CC_StopGrabbing()

            # 关闭设备
            _ret = self.camera.MV_CC_CloseDevice()
            if _ret != 0:
                print("close device fail! ret[0x%x]" % _ret)
            else:
                print(self.camera_name + ' closed')

            # 销毁句柄
            _ret = self.camera.MV_CC_DestroyHandle()
            if _ret != 0:
                print("destroy handle fail! ret[0x%x]" % _ret)

            self.camera = None

    # del
    def __del__(self):
        self.release()
        # cv2.destroyAllWindows()



if __name__ == '__main__':
    import time
    capture = Capture()
    last_time = time.time()
    while True:

        image = capture.get_frame()
        capture.show_img(image)
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        print(f"fps: {fps}")
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
