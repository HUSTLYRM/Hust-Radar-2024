# !/usr/bin/python3
# coding=utf-8

import numpy as np

CAM_WID, CAM_HGT = 640, 480  # 重投影到的深度图尺寸
CAM_FX, CAM_FY = 795.209, 793.957  # fx/fy
CAM_CX, CAM_CY = 332.031, 231.308  # cx/cy

EPS = 1.0e-16

# 加载点云数据
pc = np.genfromtxt('pc_rot.csv', delimiter=',').astype(np.float32)

# 滤除镜头后方的点
valid = pc[:, 2] > EPS
z = pc[valid, 2]

# 点云反向映射到像素坐标位置
u = np.round(pc[valid, 0] * CAM_FX / z + CAM_CX).astype(int)
v = np.round(pc[valid, 1] * CAM_FY / z + CAM_CY).astype(int)

# 滤除超出图像尺寸的无效像素
valid = np.bitwise_and(np.bitwise_and((u >= 0), (u < CAM_WID)),
                       np.bitwise_and((v >= 0), (v < CAM_HGT)))
u, v, z = u[valid], v[valid], z[valid]

# 按距离填充生成深度图，近距离覆盖远距离
img_z = np.full((CAM_HGT, CAM_WID), np.inf)
for ui, vi, zi in zip(u, v, z):
    img_z[vi, ui] = min(img_z[vi, ui], zi)  # 近距离像素屏蔽远距离像素

# 小洞和“透射”消除
img_z_shift = np.array([img_z, \
                        np.roll(img_z, 1, axis=0), \
                        np.roll(img_z, -1, axis=0), \
                        np.roll(img_z, 1, axis=1), \
                        np.roll(img_z, -1, axis=1)])
img_z = np.min(img_z_shift, axis=0)

# 保存重新投影生成的深度图dep_rot
np.savetxt('dep_rot.csv', img_z, fmt='%.12f', delimiter=',', newline='\n')

# 加载刚保存的深度图dep_rot并显示
import matplotlib.pyplot as plt

img = np.genfromtxt('dep_rot.csv', delimiter=',').astype(np.float32)
plt.imshow(img, cmap='jet')
plt.show()

