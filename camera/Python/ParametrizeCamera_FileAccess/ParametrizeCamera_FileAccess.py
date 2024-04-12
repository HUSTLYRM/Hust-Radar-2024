# -- coding: utf-8 --

import time
import sys
import threading
import termios

from ctypes import *

sys.path.append("../MvImport")
from MvCameraControl_class import *

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

# 为ProgressThread线程定义一个函数
def progress_thread(cam=0, nMode=0):
	stFileAccessProgress = MV_CC_FILE_ACCESS_PROGRESS()
	memset(byref(stFileAccessProgress), 0, sizeof(stFileAccessProgress))
	while True:
		#ch:获取文件存取进度 |en:Get progress of file access
		ret = cam.MV_CC_GetFileAccessProgress(stFileAccessProgress)
		print ("State = [%x],Completed = [%d],Total = [%d]" % (ret, stFileAccessProgress.nCompleted, stFileAccessProgress.nTotal))
		if (ret != MV_OK or (stFileAccessProgress.nCompleted != 0 and stFileAccessProgress.nCompleted == stFileAccessProgress.nTotal)):
			print('press a key to continue.')
			break

# 为FileAccessThread线程定义一个函数
def file_access_thread(cam=0, nMode=0):
	stFileAccess = MV_CC_FILE_ACCESS()
	memset(byref(stFileAccess), 0, sizeof(stFileAccess))
	stFileAccess.pUserFileName = 'UserSet1.bin'.encode('ascii')
	stFileAccess.pDevFileName = 'UserSet1'.encode('ascii')
	if 1 == nMode:
		#ch:读模式 |en:Read mode
		ret = cam.MV_CC_FileAccessRead(stFileAccess)
		if MV_OK != ret:
			print ("file access read fail ret [0x%x]\n" % ret)
	elif 2 == nMode:
		#ch:写模式 |en:Write mode
		ret = cam.MV_CC_FileAccessWrite(stFileAccess)
		if MV_OK != ret:
			print ("file access write fail ret [0x%x]\n" % ret)

# 打印设备详情
def PrintDeviceInfo(deviceList):
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

if __name__ == "__main__":

	deviceList = MV_CC_DEVICE_INFO_LIST()
	tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
	
	# ch:枚举设备 | en:Enum device
	ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
	if ret != 0:
		print ("enum devices fail! ret[0x%x]" % ret)
		sys.exit()

	if deviceList.nDeviceNum == 0:
		print ("find no Device!")
		sys.exit()

	print ("find %d devices!" % deviceList.nDeviceNum)

	# 打印设备详情
	PrintDeviceInfo(deviceList)

	if sys.version >= '3':
		nConnectionNum = input("please input the number of the device to connect:")
	else:
		nConnectionNum = raw_input("please input the number of the device to connect:")

	if int(nConnectionNum) >= deviceList.nDeviceNum:
		print ("intput error!")
		sys.exit()

	# ch:创建相机实例 | en:Creat Camera Object
	cam = MvCamera()
	
	# ch:选择设备并创建句柄 | en:Select device and create handle
	stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

	ret = cam.MV_CC_CreateHandle(stDeviceList)
	if ret != 0:
		print ("create handle fail! ret[0x%x]" % ret)
		sys.exit()

	# ch:打开设备 | en:Open device
	ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
	if ret != 0:
		print ("open device fail! ret[0x%x]" % ret)
		sys.exit()

	#ch:读模式 |en:Read mode
	print ("read to file.")
	print('press a key to start.')
	press_any_key_exit()

	try:
		hReadThreadHandle = threading.Thread(target=file_access_thread, args=(cam, 1))
		hReadThreadHandle.start()
		time.sleep(0.005)
		hProgress1ThreadHandle = threading.Thread(target=progress_thread, args=(cam, 1))
		hProgress1ThreadHandle.start()
	except:
		print ("error: unable to start thread")

	print ("waiting.")
	press_any_key_exit()

	hReadThreadHandle.join()
	hProgress1ThreadHandle.join()

	#ch:写模式 |en:Write mode
	print ("write from file.")
	print('press a key to start.')
	press_any_key_exit()

	try:
		hWriteThreadHandle = threading.Thread(target=file_access_thread, args=(cam, 2))
		hWriteThreadHandle.start()
		time.sleep(0.005)
		hProgress2ThreadHandle = threading.Thread(target=progress_thread, args=(cam, 2))
		hProgress2ThreadHandle.start()
	except:
		print ("error: unable to start thread")

	print ("waiting.")
	press_any_key_exit()

	hWriteThreadHandle.join()
	hProgress2ThreadHandle.join()

	# ch:关闭设备 | Close device
	ret = cam.MV_CC_CloseDevice()
	if ret != 0:
		print ("close deivce fail! ret[0x%x]" % ret)
		sys.exit()

	# ch:销毁句柄 | Destroy handle
	ret = cam.MV_CC_DestroyHandle()
	if ret != 0:
		print ("destroy handle fail! ret[0x%x]" % ret)
		sys.exit()
