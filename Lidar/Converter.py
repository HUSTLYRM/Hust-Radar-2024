# 将激光雷达坐标系下的点云转换到相机坐标系下

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
import numpy as np
import open3d as o3d
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
    def __init__(self, data_loader_path = 'parameters.yaml'):
        # 传入data_loader路径,用data_loader初始化类
        with open(data_loader_path, 'r') as file:
            data_loader = yaml.safe_load(file)
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
        # 相机坐标系到图像坐标系的内参矩阵，3*3的矩阵
        self.intrinsic_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        # 图像坐标系到相机坐标系的内参矩阵，3*3的矩阵
        self.intrinsic_matrix_inv = np.linalg.inv(self.intrinsic_matrix)
        # 激光雷达到相机的外参矩阵，4*4的矩阵，前三列为旋转矩阵，第四列为平移矩阵
        self.extrinsic_matrix = np.hstack((self.R, self.T))
        self.extrinsic_matrix = np.vstack((self.extrinsic_matrix, [0, 0, 0, 1]))
        # 相机到激光雷达的外参矩阵，4*4的矩阵，前三列为旋转矩阵，第四列为平移矩阵
        self.extrinsic_matrix_inv = np.linalg.inv(self.extrinsic_matrix)
        print(self.extrinsic_matrix)
        print(self.intrinsic_matrix)


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
        # Add a column of ones to the points
        pc = np.hstack((pc, np.ones((pc.shape[0], 1))))
        pc = np.dot(pc, self.extrinsic_matrix.T)
        # 提取前三列
        pc = pc[:, :3]
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
        # Add a column of ones to the points
        pc = np.hstack((pc, np.ones((pc.shape[0], 1))))
        pc = np.dot(pc, self.extrinsic_matrix_inv.T)
        # 提取前三列
        pc = pc[:, :3]
        self.update_pcd(pcd, pc)


    def camera_to_image(self, pcd): # 传入的是一个open3d的pcd格式的点云，返回的是一个n*3的矩阵，n是点云的数量，是np.array格式的
        # 相机坐标系下的点云批量乘以内参矩阵，得到图像坐标系下的u,v和z,类似于深度图的生成
        pc = self.get_points(pcd)
        xyz = np.dot(pc, self.intrinsic_matrix.T) # 得到的uvz是一个n*3的矩阵，n是点云的数量，是np.array格式的
        # 之前深度图没正确生成是因为没有提取z出来，导致原来的uv错误过大了
        # 要获得u,v,z，需要将xyz的第三列除以第三列
        uvz = np.zeros(xyz.shape)
        uvz[:, 0] = xyz[:, 0] / xyz[:, 2]
        uvz[:, 1] = xyz[:, 1] / xyz[:, 2]
        uvz[:, 2] = xyz[:, 2]


        return uvz

    # 将生成的uvz转换为深度图
    def generate_depth_map(self, uvz): # 传入的uvz是一个n*3的矩阵，n是点云的数量，是np.array格式的
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


    # 获取投影后落在深度图矩形框内的点云 , 并不是反向映射，而是直接提取落在矩形框内的点云
    def get_points_in_box(self, pcd, box, max_depth):
        # box是一个元组，包含了矩形框的左上角和右下角的坐标：(min_u, min_v, max_u, max_v)
        min_u, min_v, max_u, max_v = box
        # 提取点云坐标
        uvz = self.camera_to_image(pcd)
        # 提取u,v,z
        u = uvz[:, 0]
        v = uvz[:, 1]
        z = uvz[:, 2]
        # 创建一个mask，标记落在矩形框中的点云,因为bitwise_and每次只能操作两个数组，所以需要分开操作
        mask1 = np.bitwise_and(u >= min_u, u <= max_u)
        mask2 = np.bitwise_and(v >= min_v, v <= max_v)
        mask3 = np.bitwise_and(mask1, mask2)
        mask = np.bitwise_and(mask3, z <= self.max_depth) # 滤除超出最大深度的点云
        # 获得落在矩形框中的点云
        box_points = uvz[mask]
        return box_points

converter = Converter()
# 读取../pcd_data/points/1224_indoor1.pcd
pcd = o3d.io.read_point_cloud('./1224_scene3.pcd')
converter.show_pcd_info(pcd) # 看初始的信息
# 将激光雷达坐标系下的点云转换到相机坐标系下
converter.lidar_to_camera(pcd)
converter.show_pcd_info(pcd) # 看转换后的信息
# 可视化点云
o3d.visualization.draw_geometries([pcd])
#
# 获得深度图
uvz = converter.camera_to_image(pcd)
imgz = converter.generate_depth_map(uvz)
cv2.imshow('imgz',imgz)
cv2.waitKey(0)