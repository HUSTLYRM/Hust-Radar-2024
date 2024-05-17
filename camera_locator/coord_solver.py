import math
import cv2
import numpy as np

from global_variables import *


def get_wld_coord(cam_pos, cam_coord):
    wld_coord = np.matmul(cam_pos.pose_R, cam_coord)  # np.transpose(tag.pose_R)
    wld_coord += cam_pos.pose_t
    return wld_coord


def euler2rotation(roll, yaw, pitch):
    R_z = np.array([[math.cos(roll), -math.sin(roll), 0.],
                    [math.sin(roll), math.cos(roll), 0.],
                    [0., 0., 1.]])
    R_x = np.array([[1., 0., 0.],
                    [0., math.cos(pitch), -math.sin(pitch)],
                    [0., math.sin(pitch), math.cos(pitch)]])
    R_y = np.array([[math.cos(yaw), 0., math.sin(yaw)],
                    [0., 1., 0.],
                    [-math.sin(yaw), 0., math.cos(yaw)]])
    # rotation = np.around(np.matmul(np.matmul(R_z, R_x), R_y), decimals=2)
    rotation = np.matmul(np.matmul(R_z, R_x), R_y)
    return rotation


def xyz2translation(x, y, z):
    return np.array([[x],
                     [y],
                     [z]])


class CameraPoseSolver:
    def __init__(self, color, left_cam_cfg):
        self.color = color
        self.main_cam_info = left_cam_cfg
        # main pos: left camera in bincam w.r.t bottom left of the field (origin of the map)
        self.main_pose_R = None
        self.main_pose_t = None
        # extrinsic: between the cameras
        self.R_longfocal2main = None
        self.R_right2main = None
        self.R_realsense2main = None
        # [w, h]: the outer bound
        self.landmark_size = MySize(660, 495)
        # temp
        self.main2field_R = None
        self.main2field_t = None
        # anchor or constant
        self.flag = 0  # constant by default
        # constants
        self.cam_pos = None
        self.sign = None
        self.cos_arc_pitch = None
        self.sin_arc_pitch = None
        self.cos_arc_yaw = None
        self.sin_arc_yaw = None

    def init_by_anchor(self, anchor):
        self.flag = 1
        true_points = None
        if self.color == RED:
            # TO-DO: 确定左上角z坐标 (615 为图纸值) -> 615 mm to ground
            top_left = [8805, 15000 - 5730, 615]
            top_right = [top_left[0], top_left[1] - self.landmark_size.w, top_left[2]]
            bottom_right = [top_left[0], top_left[1] - self.landmark_size.w, top_left[2] - self.landmark_size.h]
            bottom_left = [top_left[0], top_left[1], top_left[2] - self.landmark_size.h]

            true_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        elif self.color == BLUE:
            top_left = [19195, 15000 - 9270, 615]
            top_right = [top_left[0], top_left[1] + self.landmark_size.w, top_left[2]]
            bottom_right = [top_left[0], top_left[1] + self.landmark_size.w, top_left[2] - self.landmark_size.h]
            bottom_left = [top_left[0], top_left[1], top_left[2] - self.landmark_size.h]

            true_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        else:
            print('Bad color set')

        print('p3d_np')
        print(true_points)
        print('p2d')
        print(anchor.vertexes)
        pixel_points = np.array(anchor.vertexes, dtype=np.float32)
        print('p2d_np')
        print(pixel_points)

        _, rotation_vector, self.main2field_t = cv2.solvePnP(true_points,
                                                             pixel_points,
                                                             np.array(self.main_cam_info['intrinsic']),
                                                             np.array(self.main_cam_info['distortion'])
                                                             # or empty np.array(1, 4) and remap before passing
                                                             )

        self.main_pose_R = cv2.Rodrigues(rotation_vector)[0]
        self.main2field_R = np.transpose(self.main_pose_R)
        pass

    def init_by_constant(self, constant_dict):
        self.flag = 0

        cam_bias = [225., 395., 1399.5]
        if self.color == BLUE:  # self = BLUE
            radar_base = [28988.19, 6017.49, 2500]  # blue base
            self.cam_pos = [radar_base[0] + cam_bias[0], radar_base[1] - cam_bias[1], radar_base[2] + cam_bias[2]]
            self.sign = -1
        elif self.color == RED:  # self = RED
            radar_base = [-987.55, 9018.02, 2500]  # red base
            self.cam_pos = [radar_base[0] - cam_bias[0], radar_base[1] + cam_bias[1], radar_base[2] + cam_bias[2]]
            self.sign = 1

        arc_pitch = 2. * np.pi / 180
        arc_yaw = 11. * np.pi / 180

        self.sin_arc_pitch = np.sin(arc_pitch)
        self.cos_arc_pitch = np.cos(arc_pitch)
        self.sin_arc_yaw = np.sin(arc_yaw)
        self.cos_arc_yaw = np.cos(arc_yaw)
        pass

    def load_intrinsics(self):
        pass

    def get_field_coord(self, cam_coord):
        field_coord = None
        if self.flag == 0:
            x = cam_coord[2][0] * self.cos_arc_pitch
            y = cam_coord[0][0]
            x_ = x * self.cos_arc_yaw - y * self.sin_arc_yaw
            y_ = x * self.sin_arc_yaw + y * self.cos_arc_yaw
            field_coord = [[self.sign * x_ + self.cam_pos[0]],
                            [-1 * self.sign * y_ + self.cam_pos[1]],
                            [0.]]
            pass
        elif self.flag == 1:
            print('main2field_R')
            print(self.main2field_R)
            print('main2field_t')
            print(self.main2field_t)
            print('cam_coord')
            print(cam_coord)

            field_coord = cam_coord - self.main2field_t
            field_coord = np.matmul(self.main2field_R, field_coord)

        return field_coord
