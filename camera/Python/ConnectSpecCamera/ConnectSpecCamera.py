# -- coding: utf-8 --

import time
import sys
import threading
import termios

from ctypes import *

sys.path.append("../MvImport")
from MvCameraControl_class import *

g_bExit = False

# 为线程定义一个函数
def work_thread(cam, pData, nDataSize):
	stOutFrame = MV_FRAME_OUT()
	memset(byref(stOutFrame), 0, sizeof(stOutFrame))
	while True:
		ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
		if ret == 0:
			print ("get one frame: Width[%d], Height[%d], nFrameNum[%d]"  % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
			cam.MV_CC_FreeImageBuffer(stOutFrame)
		else:
			print ("no data[0x%x]" % ret)
		if g_bExit == True:
			break

def press_any_key_exit():
	fd = sys.stdin.fileno()
	old_ttyinfo = termios.tcgetattr(fd)
	new_ttyinfo = old_ttyinfo[:]
	new_ttyinfo[3] &= ~termios.ICANON
	new_ttyinfo[3] &= ~termios.ECHO
	#sys.stdout.write(msg)
	#sys.stdout.flush()
	termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
	try:
		os.read(fd, 7)
	except:
		pass
	finally:
		termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)
	
if __name__ == "__main__":

	stDevInfo = MV_CC_DEVICE_INFO()
	stGigEDev = MV_GIGE_DEVICE_INFO()

	if sys.version >= '3':
		deviceIp = input("please input current camera ip : ")
		netIp = input("please input your PC ip : ")
	else:
		deviceIp = raw_input("please input current camera ip : ")
		netIp = raw_input("please input your PC ip : ")
	
	deviceIpList = deviceIp.split('.')
	stGigEDev.nCurrentIp = (int(deviceIpList[0]) << 24) | (int(deviceIpList[1]) << 16) | (int(deviceIpList[2]) << 8) | int(deviceIpList[3])

	netIpList = netIp.split('.')
	stGigEDev.nNetExport =  (int(netIpList[0]) << 24) | (int(netIpList[1]) << 16) | (int(netIpList[2]) << 8) | int(netIpList[3])

	stDevInfo.nTLayerType = MV_GIGE_DEVICE
	stDevInfo.SpecialInfo.stGigEInfo = stGigEDev

	# ch:创建相机实例 | en:Creat Camera Object
	cam = MvCamera()

	# ch:选择设备并创建句柄 | en:Select device and create handle
	ret = cam.MV_CC_CreateHandle(stDevInfo)
	if ret != 0:
		print ("create handle fail! ret[0x%x]" % ret)
		sys.exit()

	# ch:打开设备 | en:Open device
	ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
	if ret != 0:
		print ("open device fail! ret[0x%x]" % ret)
		sys.exit()
	
	# ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
	if stDevInfo.nTLayerType == MV_GIGE_DEVICE:
		nPacketSize = cam.MV_CC_GetOptimalPacketSize()
		if int(nPacketSize) > 0:
			ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
			if ret != 0:
				print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
		else:
			print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

	# ch:设置触发模式为off | en:Set trigger mode as off
	ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
	if ret != 0:
		print ("set trigger mode fail! ret[0x%x]" % ret)
		sys.exit()

	#ch:获取数据包大小 | en:Get payload size
	stParam =  MVCC_INTVALUE()
	memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
	ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
	if ret != 0:
		print ("get payload size fail! ret[0x%x]" % ret)
		sys.exit()
	nPayloadSize = stParam.nCurValue

	# ch:开始取流 | en:Start grab image
	ret = cam.MV_CC_StartGrabbing()
	if ret != 0:
		print ("start grabbing fail! ret[0x%x]" % ret)
		sys.exit()

	data_buf = (c_ubyte * nPayloadSize)()
	try:
		hThreadHandle = threading.Thread(target=work_thread, args=(cam, byref(data_buf), nPayloadSize))
		hThreadHandle.start()
	except:
		print ("error: unable to start thread")
		
	print ("press a key to stop grabbing.")
	press_any_key_exit()

	g_bExit = True
	hThreadHandle.join()

	# ch:停止取流 | en:Stop grab image
	ret = cam.MV_CC_StopGrabbing()
	if ret != 0:
		print ("stop grabbing fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	# ch:关闭设备 | Close device
	ret = cam.MV_CC_CloseDevice()
	if ret != 0:
		print ("close deivce fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	# ch:销毁句柄 | Destroy handle
	ret = cam.MV_CC_DestroyHandle()
	if ret != 0:
		print ("destroy handle fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	del data_buf

