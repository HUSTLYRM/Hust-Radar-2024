# 将激光雷达坐标系下的点云转换到相机坐标系下
# 关于点云处理，过程中pcd可以变换，只要vis的时候更改pcd的points属性就可以了
'''
理论上坐标系之间轴的关系：
激光雷达的x是相机坐标系的z，激光雷达的y是相机坐标系的-x，激光雷达的z是相机坐标系的-y
'''
# 外参矩阵R:
'''
   0.0092749    -0.999957  0.000449772
  0.00118781 -0.000438773    -0.999999
    0.999956   0.00927542   0.00118369
'''
# 外参矩阵T:
'''
0.00529624
 0.0306859
 -0.135507
'''
# 内参矩阵
'''
instrinsic
1246.7920	0	637.8469
0	1243.23027688354	506.5883
0	0	1
'''
# 去畸变
'''
distortion
-0.100813 0.58183 0.0031347 0.00040115 0
'''
# open3的的pcd的pcd.points在np.asarray(pcd.points)的情况下格式是这样的
'''
array([[x1, y1, z1],
       [x2, y2, z2],
       [x3, y3, z3],
       ...,
       [xn, yn, zn]])
'''
import cupy as cp
import numpy as np
import open3d as o3d
from camera_locator.anchor import Anchor
from camera_locator.point_picker import PointsPicker
import cv2
import yaml

# 从YAML文件中读取字典

# 定义一个坐标系转换的类，包含成员有外参矩阵extrinsic_matrix和相机内参矩阵intrinsic_matrix,以及内参矩阵的内容fx，fy,cx,cy,外参矩阵的内容R,T
# 方法1为将激光雷达坐标系下的点云转换到相机坐标系下
# 方法2为将相机坐标系下的点云转换到激光雷达坐标系下
# 方法3为将相机坐标系下的点云转换到图像坐标系下
# 方法4为将图像坐标系下的点云转换到相机坐标系下
class Converter:
    # 传入data_loader,用data_loader初始化类
    # def __init__(self, R, T, fx, fy, cx, cy , max_depth = 40):
    #     self.R = R # 外参矩阵的旋转矩阵，传入的是一个3*3的矩阵
    #     self.T = T # 外参矩阵的平移矩阵，传入的是一个3*1的矩阵
    #     self.fx = fx # 相机内参矩阵的fx
    #     self.fy = fy # 相机内参矩阵的fy
    #     self.cx = cx # 相机内参矩阵的cx
    #     self.cy = cy # 相机内参矩阵的cy
    #     self.max_depth = max_depth # 最大深度值
    #     # 相机坐标系到图像坐标系的内参矩阵，3*3的矩阵
    #     self.intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    #     # 图像坐标系到相机坐标系的内参矩阵，3*3的矩阵
    #     self.intrinsic_matrix_inv = np.linalg.inv(self.intrinsic_matrix) # 图像坐标系到相机坐标系的内参矩阵的逆矩阵，自动生成的，得检查一下
    #     # 激光雷达到相机的外参矩阵，4*4的矩阵，前三列为旋转矩阵，第四列为平移矩阵
    #     self.extrinsic_matrix = np.array([[R[0, 0], R[0, 1], R[0, 2], T[0]], [R[1, 0], R[1, 1], R[1, 2], T[1]], [R[2, 0], R[2, 1], R[2, 2], T[2]], [0, 0, 0, 1]])
    #     # 相机到激光雷达的外参矩阵，4*4的矩阵，前三列为旋转矩阵，第四列为平移矩阵
    #     self.extrinsic_matrix_inv = np.linalg.inv(self.extrinsic_matrix) # 相机到激光雷达的外参矩阵的逆矩阵，自动生成的，得检查一下
    def __init__(self, my_color ,data_loader_path = 'parameters.yaml' ):
        # 传入data_loader路径,用data_loader初始化类
        with open(data_loader_path, 'r') as file:
            data_loader = yaml.safe_load(file)
        self.left_up = data_loader['field'][my_color]['left_up']
        self.left_down = data_loader['field'][my_color]['left_down']
        self.right_down = data_loader['field'][my_color]['right_down']
        self.right_up = data_loader['field'][my_color]['right_up']
        # 获取R和T，并将它们转换为NumPy数组
        self.R = np.array(data_loader['calib']['extrinsic']['R']['data']).reshape(
            (data_loader['calib']['extrinsic']['R']['rows'], data_loader['calib']['extrinsic']['R']['cols']))
        self.T = np.array(data_loader['calib']['extrinsic']['T']['data']).reshape(
            (data_loader['calib']['extrinsic']['T']['rows'], data_loader['calib']['extrinsic']['T']['cols']))
        # 获取相机内参
        self.cx = data_loader['calib']['intrinsic']['cx']
        self.cy = data_loader['calib']['intrinsic']['cy']
        self.fx = data_loader['calib']['intrinsic']['fx']
        self.fy = data_loader['calib']['intrinsic']['fy']
        self.max_depth = data_loader['params']['max_depth']
        self.width = data_loader['params']['width']
        self.height = data_loader['params']['height']
        # 获取聚类参数
        self.eps = data_loader['cluster']['eps']
        self.min_points = data_loader['cluster']['min_points']
        self.print_cluster_progress = data_loader['cluster']['print_progress']
        # 获取滤波参数
        self.nb_neighbors = data_loader['filter']['nb_neighbors']
        self.std_ratio = data_loader['filter']['std_ratio']
        self.voxel_size = data_loader['filter']['voxel_size']
        # 去畸变参数
        self.distortion_matrix = np.array(data_loader['calib']['distortion']['data'])
        # 相机坐标系到图像坐标系的内参矩阵，3*3的矩阵
        self.intrinsic_matrix = cp.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        # 图像坐标系到相机坐标系的内参矩阵，3*3的矩阵
        self.intrinsic_matrix_inv = cp.linalg.inv(self.intrinsic_matrix)
        # 激光雷达到相机的外参矩阵，4*4的矩阵，前三列为旋转矩阵，第四列为平移矩阵
        self.extrinsic_matrix = cp.hstack((self.R, self.T))
        self.extrinsic_matrix = cp.vstack((self.extrinsic_matrix, [0, 0, 0, 1]))
        # 相机到激光雷达的外参矩阵，4*4的矩阵，前三列为旋转矩阵，第四列为平移矩阵
        self.extrinsic_matrix_inv = cp.linalg.inv(self.extrinsic_matrix)
        # 相机到赛场坐标系的外参矩阵，4*4的矩阵，前三列为旋转矩阵，第四列为平移矩阵
        self.camera_to_field_R = None # 后面初始化
        self.camera_to_field_T = None # 后面初始化
        self.camera_to_field_matrix = None # 后面初始化
        self.field_to_camera_R = None # 后面初始化
        print(self.extrinsic_matrix)
        print(self.intrinsic_matrix)

    # 相机坐标系到赛场坐标系的初始化
    def camera_to_field_init(self,capture):
        # 初始化要用的类
        anchor = Anchor()
        pp = PointsPicker()

        while True:
            # 获得一张图片
            image = capture.get_frame()
            # 把image resize为1920*1080
            show_image = cv2.resize(image, (1920, 1080))
            cv2.imshow("clear press y else n",show_image)
            # 接收按键，如果y则进入下一步，否则重选一张
            key = cv2.waitKey(0)
            if key == ord('y'):
                pp.caller(image, anchor)
                true_points = np.array([self.left_up, self.left_down, self.right_down, self.right_up], dtype=np.float32)
                pixel_points = np.array(anchor.vertexes, dtype=np.float32)
                _, rotation_vector, translation_vector = cv2.solvePnP(true_points, pixel_points,
                                                                      self.intrinsic_matrix.get(),
                                                                      self.distortion_matrix)
                rotation_matrix = cv2.Rodrigues(rotation_vector)[0]  # 从赛场到相机的旋转矩阵

                # 将旋转矩阵R和平移向量T合并成一个4x4的齐次坐标变换矩阵
                # 注意这里使用 rotation_matrix 和 translation_vector，前者是赛场到相机的旋转矩阵，后者是对应的平移向量
                transformation_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))  # 创建包含R和T的3x4矩阵

                transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))  # 添加一个[0, 0, 0, 1]行向量
                print("transformation_matrix",transformation_matrix)
                # 获得从相机坐标系到赛场坐标系的矩阵，通过求逆
                self.camera_to_field_matrix = np.linalg.inv(transformation_matrix)
                # 将赛场坐标系的平移部分转为m
                self.camera_to_field_matrix[:3, 3] /= 1000

                print(self.camera_to_field_matrix)

                break
            else:
                continue



    # 从numpy转到cupy
    def np2cp(self, np_array):
        return cp.array(np_array)
    # 从cupy转到numpy
    def cp2np(self, cp_array):
        return cp.asnumpy(cp_array)

    # 把open3d的pcd格式的点云的points修改为pc
    def update_pcd(self, pcd, pc):
        pcd.points = o3d.utility.Vector3dVector(pc)
        return pcd

    # 提取open3d的pcd格式的点云的points，转为np.array格式
    def get_points(self, pcd):
        pc = np.asarray(pcd.points)
        return pc
    def lidar_to_camera(self, pcd): # 对pcd直接修改，不用返回
        # 修改pcd对象的points属性，保持原有的pcd对象不变
        # 激光雷达坐标系下的点云转换到相机坐标系下,传入的是一个open3d的pcd格式的点云，在里面直接修改pcd的points属性,返回修改好的pcd
        # 从open3d的pcd格式的点云中提取点云的坐标
        pc = self.get_points(pcd)
        # 从numpy变为cupy
        pc = self.np2cp(pc)
        # Add a column of ones to the points
        pc = cp.hstack((pc, np.ones((pc.shape[0], 1))))
        pc = cp.dot(pc, self.extrinsic_matrix.T)
        # 提取前三列
        pc = pc[:, :3]
        # 从cupy变为numpy
        pc = self.cp2np(pc)

        self.update_pcd(pcd, pc)

    # 展示点云的基本信息,x,y,z的范围
    def show_pcd_info(self, pcd):
        pc = self.get_points(pcd)
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        print('x: ', x.min(), x.max())
        print('y: ', y.min(), y.max())
        print('z: ', z.min(), z.max())




    def camera_to_lidar(self, pcd): # 对pcd直接修改，不用返回
        # 相机坐标系下的点云转换到激光雷达坐标系下
        pc = self.get_points(pcd)
        pc = self.np2cp(pc)
        # Add a column of ones to the points
        pc = cp.hstack((pc, np.ones((pc.shape[0], 1))))
        pc = cp.dot(pc, self.extrinsic_matrix_inv.T)
        # 提取前三列
        pc = pc[:, :3]
        pc = self.cp2np(pc)
        self.update_pcd(pcd, pc)


    def camera_to_image(self, pcd): # 传入的是一个open3d的pcd格式的点云，返回的是一个n*3的矩阵，n是点云的数量，是np.array格式的
        # 相机坐标系下的点云批量乘以内参矩阵，得到图像坐标系下的u,v和z,类似于深度图的生成
        pc = self.get_points(pcd)
        # 从numpy变为cupy
        pc = self.np2cp(pc)



        xyz = cp.dot(pc, self.intrinsic_matrix.T) # 得到的uvz是一个n*3的矩阵，n是点云的数量，是np.array格式的
        # 之前深度图没正确生成是因为没有提取z出来，导致原来的uv错误过大了
        # 要获得u,v,z，需要将xyz的第三列除以第三列
        uvz = cp.zeros(xyz.shape)
        uvz[:, 0] = xyz[:, 0] / xyz[:, 2]
        uvz[:, 1] = xyz[:, 1] / xyz[:, 2]
        uvz[:, 2] = xyz[:, 2]

        # 从cupy变为numpy
        uvz = self.cp2np(uvz)


        return uvz

    # 将生成的uvz转换为深度图
    def generate_depth_map(self, pcd): # 传入的pcd,返回的是一个深度图

        uvz = self.camera_to_image(pcd) # 转换为uvz
        # 提取u,v,z
        u = uvz[:, 0]
        v = uvz[:, 1]
        z = uvz[:, 2]
        # 按距离填充生成深度图，近距离覆盖远距离
        width, height = self.width , self.height
        valid = np.bitwise_and(np.bitwise_and((u >= 0), (u < width)),
                               np.bitwise_and((v >= 0), (v < height)))
        img_z = np.full((height, width), np.inf)
        # 将深度值填充到深度图中
        for i in range(len(uvz)):
            if valid[i]:
                img_z[int(v[i]), int(u[i])] = min(img_z[int(v[i]), int(u[i])], z[i])
        # 小洞和“透射”消除
        img_z_shift = np.array([img_z, \
                                np.roll(img_z, 1, axis=0), \
                                np.roll(img_z, -1, axis=0), \
                                np.roll(img_z, 1, axis=1), \
                                np.roll(img_z, -1, axis=1)])
        img_z = np.min(img_z_shift, axis=0) # img_z 是一个height*width的矩阵
        # 转为可以显示的图像
        img_z = np.where(img_z > self.max_depth, self.max_depth, img_z)
        img_z = cv2.normalize(img_z, None, 0, 200, cv2.NORM_MINMAX, cv2.CV_8U)
        # img_z = cv2.normalize(img_z, None, 0, 200, cv2.NORM_MINMAX, cv2.CV_8U) # 远的看不到，就把最大值调小
        img_jet = cv2.applyColorMap(img_z, cv2.COLORMAP_JET)
        return img_jet

    # 传入一个点云，选取距离的中值点
    def get_center_mid_distance(self, pcd):
        pc = self.get_points(pcd)



        # 判空
        if len(pc) == 0:
            return [0,0,0]

        pc = self.np2cp(pc)
        # 计算每个点的距离
        distances = cp.linalg.norm(pc, axis=1)
        # 找到中值点的索引
        center_idx = cp.argsort(distances)[len(distances) // 2]
        center = pc[center_idx]
        center = self.cp2np(center)
        return center


    # 获取投影后落在深度图矩形框内的点云 , 并不是反向映射，而是直接提取落在矩形框内的点云
    def get_points_in_box(self, pcd, box): # 传入的为pcd格式点云，box是一个元组，包含了矩形框的左上角和右下角的坐标：(min_u, min_v, max_u, max_v)，返回的是一个n*3的矩阵，n是点云的数量，是np.array格式的
        # box是一个元组，包含了矩形框的左上角和右下角的坐标：(min_u, min_v, max_u, max_v)
        # 好像没有小孔成像的感觉，似乎并不是一个锥形
        min_u, min_v, max_u, max_v = box
        print(box)
        # 提取像素坐标系下坐标
        uvz = self.camera_to_image(pcd)
        # numpy到cupy
        uvz = self.np2cp(uvz)
        # 提取u,v,z
        u = uvz[:, 0]
        v = uvz[:, 1]
        z = uvz[:, 2]
        # print("z",z)
        # 创建一个mask，标记落在矩形框中的点云,因为bitwise_and每次只能操作两个数组，所以需要分开操作
        mask1 = cp.bitwise_and(u >= min_u, u <= max_u)
        mask2 = cp.bitwise_and(v >= min_v, v <= max_v)
        mask3 = cp.bitwise_and(mask1, mask2)
        mask = cp.bitwise_and(mask3, z <= self.max_depth) # 滤除超出最大深度的点云
        # 获得落在矩形框中的点云的点云的index,pcd.points才是要筛选的点云
        box_points = cp.asarray(pcd.points)[mask]

        box_points = self.cp2np(box_points)


        return box_points # 返回的是一个n*3的矩阵，n是点云的数量，是np.array格式的

    # 传入一个图片，手动选择一个矩形框，返回这个矩形框的坐标（min_u, min_v, max_u, max_v）
    def select_box(self, img):
        # 选择矩形框
        box = cv2.selectROI('select_box', img, False, False)  # box的坐标是（x，y，w，h）
        cv2.destroyWindow('select_box')

        # 调整box的坐标，使其变为（min_u, min_v, max_u, max_v）
        min_u, min_v, w, h = box
        max_u = min_u + w
        max_v = min_v + h
        box = (min_u, min_v, max_u, max_v)

        return box

    # 深拷贝点云
    def copy_pcd(self,pcd):
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        return new_pcd

    # 对点云进行DBSCAN聚类
    def cluster(self, pcd):
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels_np = np.array(pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points,
                                                    print_progress=self.print_cluster_progress))
        # Early return if labels_np is empty or all -1
        if len(labels_np) == 0 or np.all(labels_np == -1):
            return np.array([]), np.array([0, 0, 0])

        # Convert NumPy arrays to CuPy arrays for GPU acceleration
        labels = cp.asarray(labels_np)
        pcd_points = cp.asarray(pcd.points)

        # Compute cluster sizes
        max_label = labels.max().item()
        cluster_sizes = cp.array([cp.sum(labels == i).item() for i in range(max_label + 1)])

        # Early return if cluster_sizes is empty
        if len(cluster_sizes) == 0:
            return np.array([]), np.array([0, 0, 0])

        # Find the index of the largest cluster
        max_cluster_idx = cp.argmax(cluster_sizes)

        # Find all points in the largest cluster
        max_cluster_points = pcd_points[labels == max_cluster_idx]

        # Compute the centroid of the largest cluster
        centroid = cp.mean(max_cluster_points, axis=0)

        # Convert CuPy arrays back to NumPy arrays before returning
        return max_cluster_points.get(), centroid.get()

    # def cluster(self,pcd): # 传入的是一个open3d的pcd格式的点云，返回的是一个numpy格式的点云和一个中心点，如果没有找到任何簇，返回一个空的np和中心点
    #     # TODO:根据点云的距离来判断点云的稀疏程度，然后根据稀疏程度来调整eps和min_points
    #     # 使用DBSCAN进行聚类
    #     with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #         labels = np.array(pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=self.print_cluster_progress))
    #     # print("labels",labels)
    #     # 如果没有找到任何簇，返回一个空的点云和中心点
    #
    #     if len(labels) == 0:
    #         return np.array([]), np.array([0, 0, 0])
    #     if labels.argmax() == -1:
    #         return np.array([]), np.array([0, 0, 0])
    #
    #     # 计算每个簇的大小
    #     max_label = labels.max()
    #     # print(f"point cloud has {max_label + 1} clusters")
    #     cluster_sizes = [len(np.where(labels == i)[0]) for i in range(max_label + 1)]
    #
    #     # 判空
    #     if len(cluster_sizes) == 0:
    #         return np.array([]), np.array([0, 0, 0])
    #
    #
    #
    #
    #     # 找到最大簇的索引
    #     max_cluster_idx = cp.argmax(cp.asarray(cluster_sizes))
    #
    #     # 将pcd.points转为cupy数组
    #     pc = cp.asarray(pcd.points)
    #
    #     # 找到最大簇的所有点
    #     max_cluster_points = pc[cp.asarray(labels) == max_cluster_idx]
    #
    #     # 计算最大簇的中心点
    #     centroid = cp.mean(max_cluster_points, axis=0)
    #
    #     return max_cluster_points.get(), centroid.get() # 返回的是一个numpy格式的点云和一个中心点,中心点为[x,y,z]

    # 对传入点云进行滤波去除离群点和噪声点
    def remove_outliers(self, pcd):
        # 使用StatisticalOutlierRemoval滤波器去除离群点和噪声点
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
        return pcd

    # 对pcd做体素降采样
    def voxel_down_sample(self, pcd):
        # 使用VoxelGrid滤波器进行体素降采样
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        return pcd

    # 对pcd进行滤波
    def filter(self, pcd):
        # 体素降采样
        pcd = self.voxel_down_sample(pcd)
        # 去除离群点和噪声点
        pcd = self.remove_outliers(pcd)

        return pcd

    # 求一个点[x,y,z]的距离
    def get_distance(self, point):
        point = np.array(point)
        return np.sqrt(np.sum(point ** 2))

    # 传入相机坐标系的点云[x,y,z]，返回赛场坐标系的点云[x,y,z]，TODO:完成方法
    def camera_to_field(self, point):
        # 传入相机坐标系的点云[x,y,z]，返回赛场坐标系的点云[x,y,z]
        # 传入的是一个np.array格式的点云，返回的也是一个np.array格式的点云
        # 直接乘以外参矩阵即可
        point = np.hstack((point, 1))
        point = np.dot(self.camera_to_field_matrix, point)
        # 把最后的1删掉
        point = point[:3]
        return point


    # 传入两个点，返回两个点的距离
    def get_distance_between_2points(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    # 传入一个点云，返回一个点云的中心点（x,y,z）



# converter = Converter()
# # 读取../pcd_data/points/1224_indoor1.pcd
# pcd = o3d.io.read_point_cloud('./1224_scene3.pcd')
# converter.show_pcd_info(pcd) # 看初始的信息
# # 将激光雷达坐标系下的点云转换到相机坐标系下
# converter.lidar_to_camera(pcd)
# # 备份一份pcd,深拷贝,没有copy
# # pcd_backup = pcd
# pcd_backup = converter.copy_pcd(pcd)
#
# converter.show_pcd_info(pcd) # 看转换后的信息
# # 可视化点云
# o3d.visualization.draw_geometries([pcd])
#
# #
# # 获得深度图
# imgz = converter.generate_depth_map(pcd)
# # 选择roi
# box = converter.select_box(imgz)
# # 打印roi框的坐标
# print(box)
# # 获取roi内的点云
# box_points = converter.get_points_in_box(pcd_backup, box)
# # 打印点云的数量
# print(len(box_points))
# # 可视化点云
# pcd_box = o3d.geometry.PointCloud()
# pcd_box.points = o3d.utility.Vector3dVector(box_points)
# # 打印筛选点的信息
# converter.show_pcd_info(pcd_box)
# o3d.visualization.draw_geometries([pcd_box])
# cv2.imshow('imgz',imgz)
# cv2.waitKey(0)