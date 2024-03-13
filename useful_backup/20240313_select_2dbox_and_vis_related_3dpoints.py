import cv2
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

global count_msg
global first
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
        center_u = 640
        center_v = 320
        width_u = 100
        width_v = 100
        # 目标框，(min_u, min_v, max_u, max_v)
        self.detect_box =  (center_u - width_u, center_v - width_v, center_u + width_u, center_v + width_v)

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
        img_jet = cv2.applyColorMap(img_z, cv2.COLORMAP_JET)
        return img_jet

    # 获取投影后落在深度图矩形框内的点云
    def get_box_points(self, pcd):
        # box是一个元组，包含了矩形框的左上角和右下角的坐标：(min_u, min_v, max_u, max_v)
        min_u, min_v, max_u, max_v = self.detect_box

        pc = np.asarray(pcd.points)
        z = np.copy(pc[:, 0])
        y = -np.copy(pc[:, 2])
        x = -np.copy(pc[:, 1])
        pc[:, 0] = x
        pc[:, 1] = y
        pc[:, 2] = z

        u = np.round(pc[:, 0] * self.fx / pc[:, 2] + self.cx).astype(int)
        v = np.round(pc[:, 1] * self.fy / pc[:, 2] + self.cy).astype(int)

        # 创建一个mask，标记落在矩形框中的点云,因为bitwise_and每次只能操作两个数组，所以需要分开操作
        mask1 = np.bitwise_and(u >= min_u, u <= max_u)
        mask2 = np.bitwise_and(v >= min_v, v <= max_v)
        mask3 = np.bitwise_and(mask1, mask2)
        mask = np.bitwise_and(mask3, pc[:, 2] <= self.MAX_DEPTH)

        # 获得落在矩形框中的点云
        box_points = pc[mask]

        # 转为open3d需要的
        pcd_box = o3d.geometry.PointCloud()
        pcd_box.points = o3d.utility.Vector3dVector(box_points)

        return pcd_box

    def set_box(self, img):
        # 调用 selectROI 函数
        r = cv2.selectROI(img)
        # 将 ROI 的坐标转换为元组（min_u, min_v, max_u, max_v）
        self.detect_box = (int(r[0]), int(r[1]), int(r[0] + r[2]), int(r[1] + r[3]))
        # 按任意键可以）
        cv2.waitKey(0)
        return self.detect_box
# 创建点云队列
class PcdQueue(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)

    def add(self, pcd):
        self.queue.append(pcd)

    def get_all(self):
        return list(self.queue)

    # 获得队列中点的数量，而非队列的大小
    def point_num(self):
        num = 0
        for pcd in self.queue:
            num += len(pcd.points)
        return num




def main():
    global count_msg
    global first
    # set param
    fx = 1269.8676
    fy = 1276.0659
    cx = 646.6841
    cy = 248.7859
    count_msg = 0
    first = True

    CAM_WID, CAM_HGT = 1280, 640  # 重投影到的深度图尺寸
    CAM_FX, CAM_FY = fx, fy  # fx/fy
    CAM_CX, CAM_CY = cx, cy  # cx/cy

    EPS = 1.0e-16
    MAX_DEPTH = 20.0  # 最大深度值
    # 创建深度图对象
    d = depth(fx, fy, cx, cy, EPS, MAX_DEPTH, CAM_WID, CAM_HGT)
    # 创建可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])


    # 创建点云队列
    pcd_queue = PcdQueue(max_size=15)

    def callback(msg):
        global count_msg
        global first

        # 将 ROS PointCloud2 消息转换为表示每个点的生成器对象
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        # 查看获取生成器对象含有什么
        # print()
        # 总点云
        pcd_merge = o3d.geometry.PointCloud()



        # 将点生成器转换为 numpy 数组，并将其转换为 Open3D 点云
        point_cloud = np.array(list(gen))
        # print(point_cloud)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        # pc = np.asarray(pcd.points)
        # print(pc)

        # 添加点云到队列
        pcd_queue.add(pcd)

        # 清除所有的几何体
        vis.clear_geometries()

        # 把所有点云放入一个merge中


        # 添加所有的点云到可视化窗口
        for pcd in pcd_queue.get_all():
            #vis.add_geometry(pcd)
            # 把所有点云放入一个merge中
            pcd_merge += pcd

        # 备份一份merge
        pcd_merge_backup = o3d.geometry.PointCloud()
        pcd_merge_backup.points = o3d.utility.Vector3dVector(np.asarray(pcd_merge.points))

        img_z = d.pcd_to_depth(pcd_merge)
        select_pcd = d.get_box_points(pcd_merge_backup)

        vis.add_geometry(select_pcd)



        # 把深度图转换为伪彩色图,封装了之后在里面了
        # img_jet = cv2.applyColorMap(img_z, cv2.COLORMAP_JET)

        # 增加计数，并判断是否够十次，如果够则手绘一次roi
        count_msg += 1
        if count_msg == 10:
            # 假如你的图像变量名为 img
            d.set_box(img_z)

        # 把检测框画在深度图上
        cv2.rectangle(img_z, (d.detect_box[0], d.detect_box[1]), (d.detect_box[2], d.detect_box[3]), (0, 0, 255), 2)

        cv2.imshow('depth',img_z)
        cv2.waitKey(1)
        # print('1')
        # cv2.imshow('img_z', img_z)
        # cv2.waitKey(1)
        # print('2')


        # print(pcd_queue.point_num())

        vis.poll_events()
        vis.update_renderer() # 重新渲染

    rospy.init_node('open3d_visualize_node', anonymous=True)
    rospy.Subscriber('livox/lidar', PointCloud2, callback)

    try:
        while rospy.is_shutdown() == False:
            rospy.spin()
            rospy.sleep(0.01)
    except KeyboardInterrupt:
        print("Shutting down")

    vis.destroy_window()

if __name__ == '__main__':
    main()