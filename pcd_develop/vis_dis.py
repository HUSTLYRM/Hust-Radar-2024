import cv2
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import copy

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
        img_z = cv2.normalize(img_z, None, 0, 200, cv2.NORM_MINMAX, cv2.CV_8U) # 远的看不到，就把最大值调小
        img_jet = cv2.applyColorMap(img_z, cv2.COLORMAP_JET)
        return img_jet

    # 获取投影后落在深度图矩形框内的点云,返回的是一个numpy数组
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
        # pcd_box = o3d.geometry.PointCloud()
        # pcd_box.points = o3d.utility.Vector3dVector(box_points)

        return box_points

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
    def __init__(self, max_size,voxel_size = 0.05):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)
        # 创建一个空的voxel
        self.voxel = o3d.geometry.VoxelGrid()
        self.voxel_size = voxel_size
        self.pcd_all = o3d.geometry.PointCloud()

    # 目前是每次添加都会把所有点云转为voxel，然后更新voxel
    def add(self, pcd):
        self.queue.append(pcd)
        self.pcd_all = self.get_all_pcd()
        self.update_voxel(self.voxel_size)

    def get_all(self):
        return list(self.queue)

    # 获得队列中点的数量，而非队列的大小
    def point_num(self):
        num = 0
        for pcd in self.queue:
            num += len(pcd.points)
        return num

    # 获得队列中的所有点云，以o3d的geometry的PointCloud的形式
    def get_all_pcd(self):
        pcd_all = o3d.geometry.PointCloud()
        for pcd in self.queue:
            pcd_all += pcd
        return pcd_all

    # 将队列中的点云转为voxel格式
    def update_voxel(self,voxel_size = 0.05):
        self.voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd_all, voxel_size=voxel_size)


    # 获取一份新的可处理的voxel
    def get_voxel_copy(self):
        voxel = o3d.geometry.VoxelGrid(self.voxel)
        return voxel

    # 获取只读voxel
    def get_voxel(self):
        return self.voxel

    # 聚类并返回最大簇的点云和中心点坐标
    def cluster(self,pcd):
        # 使用DBSCAN进行聚类
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=0.10, min_points=10, print_progress=True))

        # 如果没有找到任何簇，返回一个空的点云和中心点
        if labels.max() == -1:
            return np.array([]), np.array([0, 0, 0])


        # 计算每个簇的大小
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        cluster_sizes = [len(np.where(labels == i)[0]) for i in range(max_label + 1)]

        # 找到最大簇的索引
        max_cluster_idx = np.argmax(cluster_sizes)

        # 找到最大簇的所有点
        max_cluster_points = np.asarray(pcd.points)[np.where(labels == max_cluster_idx)]

        # 计算最大簇的中心点
        centroid = max_cluster_points.mean(axis=0)

        return max_cluster_points, centroid

    # 获取点云的距离中值的点,返回的是一个numpy数组
    def get_median_point(self,pcd):
        # 计算每个点到原点的距离
        distance = np.sqrt(np.sum(np.asarray(pcd.points) ** 2, axis=1))

        # 取中值点
        median_distance = np.median(distance)
        # print("测据:", median_distance)
        # 取中值点的索引
        median_index = np.where(distance == median_distance)
        # 取中值点
        median_point = np.asarray(pcd.points)[median_index]

        return median_point

    # 传入一个点（numpy数组），和一个点云，返回一个高亮了这个点的点云
    def highlight_point(self,point,pcd):
        # 将点转换为numpy数组
        points = np.asarray(pcd.points)

        # 添加新的点
        points = np.vstack([points, point])

        # 转换回Vector3dVector并赋值给show_pcd
        pcd.points = o3d.utility.Vector3dVector(points)
        # 创建一个颜色数组，对应于show_pcd中的每个点，将所有的点设置为绿色
        colors = np.ones((len(pcd.points), 3)) * [0, 1, 0]  # 所有点默认为绿色

        # 将最后一个颜色设置为红色
        colors[-1] = [1, 0, 0]  # 最后一个点为红色

        # 设置点云的颜色
        pcd.colors = o3d.utility.Vector3dVector(colors)







def main():
    global count_msg
    global first
    global show_pcd
    global add_flag
    global show_pcd_ori
    global text
    global dis
    # set param
    fx = 1246.79200717189
    fy = 1243.23027688354
    cx = 637.846976999981
    cy = 506.588375264748
    count_msg = 0
    first = True
    add_flag = False
    dis = 0.0

    CAM_WID, CAM_HGT = 1280, 640  # 重投影到的深度图尺寸
    CAM_FX, CAM_FY = fx, fy  # fx/fy
    CAM_CX, CAM_CY = cx, cy  # cx/cy

    EPS = 1.0e-16
    MAX_DEPTH = 10.0  # 最大深度值
    # 创建深度图对象
    d = depth(fx, fy, cx, cy, EPS, MAX_DEPTH, CAM_WID, CAM_HGT)
    # 创建可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # 创建原始点云可视化对象
    vis_ori = o3d.visualization.Visualizer()
    vis_ori.create_window()

    # 创建pcd
    select_pcd = o3d.geometry.PointCloud()
    # 创建一个专门用来展示的pointcloud
    show_pcd = o3d.geometry.PointCloud()
    # 创建一个专门用来展示原始点云的pointcloud
    show_pcd_ori = o3d.geometry.PointCloud()
    # 创立text
    text = o3d.geometry.Text3D(str(dis), centroid, orientation=[0.0, 0.0, 1.0], size=0.1, color=[1, 0, 0])
    vis.add_geometry(text)
    # 创建一个高亮的点云
    highlight_pcd = o3d.geometry.PointCloud()

    # 创建voxel
    voxel_all = o3d.geometry.VoxelGrid()
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])


    # 创建点云队列
    pcd_queue = PcdQueue(max_size=10)

    def callback(msg):
        global count_msg
        global show_pcd
        global show_pcd_ori
        global first
        global add_flag

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
        # vis.clear_geometries()

        # 把所有点云放入一个merge中


        # 添加所有的点云到可视化窗口
        pcd_merge = pcd_queue.get_all_pcd()

        # 备份一份merge
        pcd_merge_backup = o3d.geometry.PointCloud()
        pcd_merge_backup.points = o3d.utility.Vector3dVector(np.asarray(pcd_merge.points))

        img_z = d.pcd_to_depth(pcd_merge)
        # 每次不是重新创建一个点云，而是更新点云的点，所以可以用update
        select_pcd_points = d.get_box_points(pcd_merge_backup)
        select_pcd.points = o3d.utility.Vector3dVector(select_pcd_points)
        show_pcd_ori.points = o3d.utility.Vector3dVector(select_pcd_points)
        # print(len(select_pcd.points))
        # 对select_pcd进行统计离群值滤波
        cl , ind = select_pcd.remove_radius_outlier(nb_points=16, radius=0.1)
        # final_cloud = select_pcd.select_by_index(ind) # 选择去除离群值后的点云,如果想要保留离群值，可以用select_by_index(ind, invert=True)

        print("before:",len(cl.points))
        # 用dbscan聚类，返回最大簇的点云和中心点坐标
        cl_pt, centroid = pcd_queue.cluster(cl)
        print("after:",len(cl_pt))

        # 把points:cl_pt传入cl中
        cl.points = o3d.utility.Vector3dVector(cl_pt)

        # 打印中心点坐标,不用科学技术法,计算中心点距离
        # print(centroid)
        print("距离:",np.sqrt(np.sum(centroid ** 2)))






        show_pcd.points = o3d.utility.Vector3dVector(cl.points)


        # 计算cl点云的距离，取中值点，加入show_pcd的points中并高亮展示
        # 调用方法计算
        median_point = pcd_queue.get_median_point(cl)
        # 调用方法高亮
        pcd_queue.highlight_point(median_point,show_pcd)

        print(len(show_pcd.points))


        # print(len(show_pcd.points))
        # voxel_all = pcd_queue.get_voxel()
        # 加个如果第一次点云判空如果是空的就跳过这一步，不把first变换
        if add_flag:
            if first:
                # 对show_pcd进行判空,如果不为空才添加
                if len(show_pcd.points):
                    first = False
                    # print("add")

                    vis.add_geometry(show_pcd)
                    vis_ori.add_geometry(show_pcd_ori)
            else:
                # print("update")
                print("pcd_ori:", len(show_pcd_ori.points))
                print("pcd:",len(show_pcd.points))
                vis.update_geometry(show_pcd) # select_pcd
                vis_ori.update_geometry(show_pcd_ori)



        # 把深度图转换为伪彩色图,封装了之后在里面了
        # img_jet = cv2.applyColorMap(img_z, cv2.COLORMAP_JET)

        # 增加计数，并判断是否够十次，如果够则手绘一次roi
        count_msg += 1
        if count_msg == 20:
            # 假如你的图像变量名为 img
            d.set_box(img_z)
            add_flag = True

        # 把检测框画在深度图上
        cv2.rectangle(img_z, (d.detect_box[0], d.detect_box[1]), (d.detect_box[2], d.detect_box[3]), (0, 0, 255), 2)
        # 把检测框中心点像素坐标计算出来并print
        center_u = (d.detect_box[0] + d.detect_box[2]) / 2
        center_v = (d.detect_box[1] + d.detect_box[3]) / 2
        print("center_u:",center_u,"center_v:",center_v)


        cv2.imshow('depth',img_z)
        cv2.waitKey(1)
        # print('1')
        # cv2.imshow('img_z', img_z)
        # cv2.waitKey(1)
        # print('2')


        # print(pcd_queue.point_num())

        vis.poll_events()
        vis_ori.poll_events()
        vis.update_renderer() # 重新渲染
        vis_ori.poll_events()

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