# -- coding: utf-8 --

import sys
import threading
import termios

from ctypes import *

sys.path.append("../MvImport")
from MvCameraControl_class import *

g_bExit = False

# 为线程定义一个函�?
def work_thread(cam=0, pData=0, nDataSize=0):
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

	deviceList = MV_CC_DEVICE_INFO_LIST()
	tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
	
	# ch:枚举设备 | en:Enum device
	ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
	if ret != 0:
		print ("enum devices fail! ret[0x%x]" % ret)
		sys.exit()

	if deviceList.nDeviceNum == 0:
		print ("find no device!")
		sys.exit()

	print ("find %d devices!" % deviceList.nDeviceNum)

	for i in range(0, deviceList.nDeviceNum):
		mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
		if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
			print ("\ngige device: [%d]" % i)
			strModeName = ""
			for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
				strModeName = strModeName + chr(per)
			print ("device model name: %s" % strModeName)

			nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
			nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
			nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
			nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
			print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
		elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
			print ("\nu3v device: [%d]" % i)
			strModeName = ""
			for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
				if per == 0:
					break
				strModeName = strModeName + chr(per)
			print ("device model name: %s" % strModeName)

			strSerialNumber = ""
			for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
				if per == 0:
					break
				strSerialNumber = strSerialNumber + chr(per)
			print ("user serial number: %s" % strSerialNumber)

	if sys.version >= '3':
		nConnectionNum = input("please input the number of the device to connect:")
	else:
		nConnectionNum = raw_input("please input the number of the device to connect:")

	if int(nConnectionNum) >= deviceList.nDeviceNum:
		print ("intput error!")
		sys.exit()

	# ch:创建相机实例 | en:Creat Camera Object
	cam = MvCamera()

	# ch:选择设备并创建句�?| en:Select device and create handle
	stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

	ret = cam.MV_CC_CreateHandle(stDeviceList)
	if ret != 0:
		print ("create handle fail! ret[0x%x]" % ret)
		sys.exit()

	#ch:询问用户启动多播控制应用程序或多播监控应用程�?
	#en:Ask the user to launch: the multicast controlling application or the multicast monitoring application.
	if sys.version >= '3':
		key = input("start multicast sample in (c)ontrol or in (m)onitor mode? (c/m): ")
	else:
		key = raw_input("start multicast sample in (c)ontrol or in (m)onitor mode? (c/m): ")

	#ch:查询用户使用的模�?| en:Query the user for the mode to use.
	monitor = False
	if key == 'm' or key == 'M':
		monitor = True
	elif key == 'c' or key == 'C':
		monitor = False
	else:
		print ("intput error!")
		sys.exit()

	if monitor:
		ret = cam.MV_CC_OpenDevice(MV_ACCESS_Monitor, 0)
		if ret != 0:
			print ("open device fail! ret[0x%x]" % ret)
			sys.exit()
	else:
		ret = cam.MV_CC_OpenDevice(MV_ACCESS_Control, 0)
		if ret != 0:
			print ("open device fail! ret[0x%x]" % ret)
			sys.exit()

	# ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
	if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
		nPacketSize = cam.MV_CC_GetOptimalPacketSize()
		if int(nPacketSize) > 0:
			ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
			if ret != 0:
				print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
		else:
			print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

	#ch:获取数据包大�?| en:Get payload size
	stParam =  MVCC_INTVALUE()
	memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

	ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
	if ret != 0:
		print ("get payload size fail! ret[0x%x]" % ret)
		sys.exit()
	nPayloadSize = stParam.nCurValue

	#ch:指定组播ip | en:multicast IP
	strIp = "239.192.1.1"
	device_ip_list = strIp.split('.')
	dest_ip = (int(device_ip_list[0]) << 24) | (int(device_ip_list[1]) << 16) | (int(device_ip_list[2]) << 8) | int(device_ip_list[3])
	print ("dest ip: %s" % strIp)

	#ch:可指定端口号作为组播组端�?| en:multicast port
	stTransmissionType = MV_TRANSMISSION_TYPE() 
	memset(byref(stTransmissionType), 0, sizeof(MV_TRANSMISSION_TYPE))

	stTransmissionType.enTransmissionType = MV_GIGE_TRANSTYPE_MULTICAST
	stTransmissionType.nDestIp = dest_ip
	stTransmissionType.nDestPort = 1042

	ret = cam.MV_GIGE_SetTransmissionType(stTransmissionType)
	if MV_OK != ret:
		print ("set transmission type fail! ret [0x%x]" % ret)

	# ch:开始取�?| en:Start grab image
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

	# ch:销毁句�?| Destroy handle
	ret = cam.MV_CC_DestroyHandle()
	if ret != 0:
		print ("destroy handle fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	del data_buf
