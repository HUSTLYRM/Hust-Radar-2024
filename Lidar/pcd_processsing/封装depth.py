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
MAX_DEPTH = 30.0  # 最大深度值

# pcd文件读取
pcd = o3d.io.read_point_cloud('/home/nvidia/RadarWorkspace/code/ros_receive_test/pcd_data/0.pcd')

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

# main
if __name__ == '__main__':
    d = depth(fx, fy, cx, cy, EPS, MAX_DEPTH, CAM_WID, CAM_HGT)
    img_z = d.pcd_to_depth(pcd)
    print('1')
    cv2.imshow('img_z', img_z)
    print('2')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('img_z.png', img_z)
    print('depth image saved as img_z.png')