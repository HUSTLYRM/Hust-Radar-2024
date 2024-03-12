import numpy as np
import open3d as o3d
import cv2

# set param
fx = 1269.8676
fy = 1276.0659
cx = 646.6841
cy = 248.7859

CAM_WID, CAM_HGT = 1280, 640  # 重投影到的深度图尺寸
CAM_FX, CAM_FY = fx, fy  # fx/fy
CAM_CX, CAM_CY = cx, cy  # cx/cy

EPS = 1.0e-16
MAX_DEPTH = 50.0  # 最大深度值

# pcd文件读取
pcd = o3d.io.read_point_cloud('/home/nvidia/RadarWorkspace/pcd/outdoor.pcd')
print(pcd)
pc = np.asarray(pcd.points)
print(pc)
#相机坐标系：这是一个以相机为中心的坐标系。在这个坐标系中，原点是相机的光心（即镜头的中心），Z轴通常指向相机的前方，X轴和Y轴分别指向相机的右方和下方。在这个坐标系中，一个点的坐标值表示这个点相对于相机的位置。
# 如果我认为相机坐标系和激光雷达坐标系的原点重合，朝向一致，那么是否可以认为激光雷达的x是相机坐标系的z，激光雷达的y是相机坐标系的-x，激光雷达的z是相机坐标系的-y
#激光雷达看的正方向，向前为x，向左为y，向上为z
# Convert lidar coordinates to camera coordinates

# pc[:, [0, 1, 2]] = pc[:, [2, 0, 1]]  # Swap x, y, z to z, x, y
# pc[:, [1, 2]] = -pc[:, [1, 2]]  # Invert x and y
#
print(pc)
z = np.copy(pc[:, 0])
y = -np.copy(pc[:, 2])
x = -np.copy(pc[:, 1])
pc[:, 0] = x
pc[:, 1] = y
pc[:, 2] = z
# Print out the minimum, maximum, and average depth values in the point cloud data

min_val_pc = np.min(pc[:, 2])
max_val_pc = np.max(pc[:, 2])
mean_val_pc = np.mean(pc[:, 2])
'''
min_val_pc = np.min(z)
max_val_pc = np.max(z)
mean_val_pc = np.mean(z)
'''

print(f'Min depth value in point cloud: {min_val_pc}')
print(f'Max depth value in point cloud: {max_val_pc}')
print(f'Mean depth value in point cloud: {mean_val_pc}')

# 滤除镜头后方的点
valid = pc[:, 2] > EPS
z = pc[valid, 2]

print(f'Number of points after filtering: {len(z)}')

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

# 将超过最大深度值的深度值设为最大深度值
img_z = np.where(img_z > MAX_DEPTH, MAX_DEPTH, img_z)

print(f'Number of inf pixels: {np.sum(np.isinf(img_z))}')

# Before normalization
min_val = np.min(img_z[~np.isinf(img_z)])  # Minimum value excluding inf
max_val = np.max(img_z[~np.isinf(img_z)])  # Maximum value excluding inf
mean_val = np.mean(img_z[~np.isinf(img_z)])  # Mean value excluding inf

print(f'Min depth value: {min_val}')
print(f'Max depth value: {max_val}')
print(f'Mean depth value: {mean_val}')
# 归一化深度值到0-255
img_z = cv2.normalize(img_z, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# 显示深度图
cv2.imshow('Depth Image', img_z)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 定义一个类，把上面的代码封装起来
class depth:
    def __init__(self, fx, fy, cx, cy, EPS, MAX_DEPTH, CAM_WID, CAM_HGT):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.EPS = EPS
        self.MAX_DEPTH = MAX_DEPTH
        self.CAM_WID = CAM_WID
        self.CAM_HGT = CAM_HGT

    def pcd_to_depth(self, pcd):
        pc = np.asarray(pcd.points)
        z = np.copy(pc[:, 0])
        y = -np.copy(pc[:, 2])
        x = -np.copy(pc[:, 1])
        pc[:, 0] = x
        pc[:, 1] = y
        pc[:, 2] = z
        valid = pc[:, 2] > self.EPS
        z = pc[valid, 2]
        u = np.round(pc[valid, 0] * self.fx / z + self.cx).astype(int)
        v = np.round(pc[valid, 1] * self.fy / z + self.cy).astype(int)
        valid = np.bitwise_and(np.bitwise_and((u >= 0), (u < self.CAM_WID)),
                               np.bitwise_and((v >= 0), (v < self.CAM_HGT)))
        u, v, z = u[valid], v[valid], z[valid]
        img_z = np.full((self.CAM_HGT, self.CAM_WID), np.inf)
        for ui, vi, zi in zip(u, v, z):
            img_z[vi, ui] = min(img_z[vi, ui], zi)
        img_z_shift = np.array([img_z, \
                                np.roll(img_z, 1, axis=0), \
                                np.roll(img_z, -1, axis=0), \
                                np.roll(img_z, 1, axis=1), \
                                np.roll(img_z, -1, axis=1)])
        img_z = np.min(img_z_shift, axis=0)
        img_z = np.where(img_z > self.MAX_DEPTH, self.MAX_DEPTH, img_z)
        img_z = cv2.normalize(img_z, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return img_z

