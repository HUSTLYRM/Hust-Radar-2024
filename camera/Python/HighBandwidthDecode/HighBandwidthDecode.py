# -- coding: utf-8 --

import sys
import threading
import os
import termios

from ctypes import *

sys.path.append("../MvImport")
from MvCameraControl_class import *

g_bExit = False

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

	SDKVersion = MvCamera.MV_CC_GetSDKVersion()
	print ("SDKVersion[0x%x]" % SDKVersion)

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

	print ("Find %d devices!" % deviceList.nDeviceNum)

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
	
	# ch:选择设备并创建句柄| en:Select device and create handle
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
	
	# ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
	if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
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

	# ch:获取数据包大小 | en:Get payload size
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

	nImageNum=10
	stOutFrame = MV_FRAME_OUT()
	stDecodeParam = MV_CC_HB_DECODE_PARAM()
	memset(byref(stOutFrame), 0, sizeof(stOutFrame))
	for i in range(0,nImageNum):
		ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
		if None != stOutFrame.pBufAddr and 0 == ret:
			print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
			stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))

			stDecodeParam.pSrcBuf = stOutFrame.pBufAddr
			stDecodeParam.nSrcLen = stOutFrame.stFrameInfo.nFrameLen
			stDecodeParam.pDstBuf = (c_ubyte * nPayloadSize)()
			stDecodeParam.nDstBufSize = nPayloadSize
			ret = cam.MV_CC_HBDecode(stDecodeParam)
			if ret != 0:
				print("HB Decode fail! ret[0x%x]" % ret)
				break
			else:
				print("Decode succeed!")

			nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)

			file_path = "Image_w%d_h%d_fn%d.raw" %(stOutFrame.stFrameInfo.nWidth,stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum)
			file_open = open(file_path.encode('ascii'),'wb+')
			try:
				error_save = (c_ubyte * stDecodeParam.nDstBufLen)()
				memmove(byref(error_save),stDecodeParam.pDstBuf,stDecodeParam.nDstBufLen)
				file_open.write(error_save)
			except:
				raise Exception("save error raw file executed failed!")
			finally:
                		file_open.close()
		else:
			print("no data[0x%x]" % ret)
		if g_bExit == True:
            		break
		
	print ("press a key to exit.")
	press_any_key_exit()

	g_bExit = True

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
