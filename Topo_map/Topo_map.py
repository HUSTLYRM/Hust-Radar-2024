"""
UPDATE: 5.8.24

初代拓扑图
`Topo_map.py` 主要用于处理和管理地图拓扑信息。它定义了多个类，包括 `Point`，`MathAnalays`，`Area` 和 `Topology`。

`Point` 类用于表示二维空间中的点，`MathAnalays` 类提供了一些数学分析方法，主要是判断点是否在给定的多边形区域内。
`Area` 类定义了地图上的区域，包括区域的边界、高度、是否为高地等属性，以及一些方法，如添加或移除车辆、判断点是否在区域内等。

`Topology` 类是主要的类，它维护了地图的拓扑结构，包括区域、边、车辆的区域映射等信息。
它提供了一些方法，如定义区域、构建拓扑图、为车辆分配区域、更新车辆信息等，（更新车辆信息个人认为应该放Car里，这俩仅做了定义，没有向外传递）
主线程是run,参数是车辆列表，与 Carlist 中get_map_info返回值一致，拓扑图构造需要接收可视化窗口对象，与watcher中的窗口对象保持一致。
还有一些辅助方法，如可以快速的获取车辆所在区域，区域之间边的信息。

此外，`Topo_map.py` 还从一个YAML文件中读取地图配置信息，包括地图的宽度和高度、区域的定义、边的定义等。

但由于接口未完成，因此好多地方没有测试，所以可能会有一些接入性问题，若有问题请联系jjy

"""
import pygame

from ruamel.yaml import YAML

map_cfg_path = "./top_map.yaml"
map_cfg = YAML().load(open(map_cfg_path, encoding='Utf-8', mode='r'))  # 将main_config.yanl文件加载到 mian_cfg中

""""""""""""


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        # 重载小于运算符，先比较y坐标，然后比较x坐标
        if self.y == other.y:
            return self.x < other.x
        return self.y < other.y


# 主要是判断点是否在区域内
class MathAnalays:
    def __init__(self):
        pass

    def on_segment(p1, p2, q):
        # 判断点q是否在线段p1p2上
        return (min(p1.x, p2.x) <= q.x <= max(p1.x, p2.x)) and \
            (min(p1.y, p2.y) <= q.y <= max(p1.y, p2.y)) and \
            ((p2.x - p1.x) * (q.y - p1.y) == (q.x - p1.x) * (p2.y - p1.y))

    def contain(position, polygon_vertices):
        point = Point(*position)
        count = 0
        n = len(polygon_vertices)

        for i in range(n):
            p1 = Point(*polygon_vertices[i])
            p2 = Point(*polygon_vertices[(i + 1) % n])

            if p1 == p2:
                continue  # 忽略重复的点

            if MathAnalays.on_segment(p1, p2, point):
                return True  # 如果点在边上，返回True

            # 确保p1是低点，p2是高点
            if p1 > p2:
                p1, p2 = p2, p1

            # 检查射线是否在边的端点与异侧的点之间
            if point.y == p1.y or point.y == p2.y:
                point.y += 0.1  # 对于水平边缘，微调点y值以避免交点计数错误

            # 只考虑向上或者平行(y相等)的边，排除向下的边
            if point.y > p1.y and point.y <= p2.y:
                if point.x <= max(p1.x, p2.x):
                    # 求直线与射线的交点，只考虑x坐标
                    xinters = (point.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                    if xinters == point.x:
                        return True  # 直线交点与点重合
                    if p1.x == p2.x or point.x <= xinters:
                        count += 1

        return count % 2 != 0  # 奇数次表示在多边形内


# 定义区域的数据结构
class Area:
    def __init__(self, id, vertices, height=0, is_hightland=False, extended_vertices=None, neighbors=None,
                 importance=1):
        # if neighbors is None:
        #     neighbors = []
        self.extended_vertices = None
        self.id = id  # 区域的id
        self.height = height  # 区域的高度
        self.is_highland = is_hightland  # 是否是高地
        if self.is_highland:  # 如果是高地，计算扩展区域
            self.calculate_extended_vertices()
        self.vertices = vertices  # 区域的边界点坐标
        self.vehicles_in_area = []  # 存储区域内的车辆编号
        self.importance = importance  # 暂时没想好在决策阶段不同区域的差异用什么来表征，姑且暂定为 importance
        self.attack_edges = []  # 攻击边
        self.access_edges = []  # 通行边

    def calculate_extended_vertices(self, expansion_factor=10):
        # 计算高地区域的扩展区域顶点
        # 但由于区域的几何形状不太好，尝试了几次后决定之后采用硬编码
        pass

    # 使用多边形点包含算法来判断点是否在区域内
    def contains_point(self, point):
        x, y = point[:2]  # 只取x，y坐标进行判断
        return MathAnalays.contain((x, y), self.vertices) or MathAnalays.contain((x, y), self.extended_vertices)

    # 绘制区域
    def draw(self, surface, color, thickness):
        if len(self.vertices) > 1:
            pygame.draw.lines(surface, color, True, self.vertices, thickness)

    # 添加车辆
    def add_vehicle(self, vehicle_id):
        if vehicle_id not in self.vehicles_in_area:
            self.vehicles_in_area.append(vehicle_id)

    # 移除车辆
    def remove_vehicle(self, vehicle_id):
        if vehicle_id in self.vehicles_in_area:
            self.vehicles_in_area.remove(vehicle_id)

    # 其他方法...


# 定义拓扑图的数据结构，实例化时接收一个车辆列表，与Carlist保持一致
class Topology:
    def __init__(self, show_window):
        self.dead_car_list = []  # 死亡车辆列表，这部分建议在carlist维护， 此处仅仅做更新
        self.avail_car_list = [] # 有效车辆列表，这部分建议在carlist维护， 此处仅仅做更新
        self.car_list = [] # 所有车辆列表
        self.width = map_cfg['map_width'] # 地图宽度
        self.height = map_cfg['map_height'] # 地图高度
        self.window = show_window # 可视化窗口对象，与watcher中的窗口对象保持一致
        self.vehicles = {}  # {vehicle_id: Vehicle_instance} 车辆字典，快速查找
        self.edges = {}  # 构建有向拓扑图
        self.areas1 = {}  # 我方区域
        self.areas2 = {}  # 敌方区域
        self.define_areas()  # 初始化所有区域
        self.areas = {**self.areas2, **self.areas1}  # 合并两个区域字典
        self.build_map()  # 建图

    # 定义区域
    def define_areas(self):
        # 定义区域和计算拓展区域，从配置文件中读取区域信息
        for node in map_cfg['nodes']:
            # print(node['id'], end=' ') 调试信息
            # print(node['corners'])
            is_highland = node.get('is_highland', False)  # 使用 get 方法防止 KeyError
            height = node.get('height', 0)  # 默认高度为0
            if node['flag'] == 'red':
                # 创建区域实例
                self.areas1[node['id']] = Area(node['id'], node['corners'], is_highland, height)
            else:
                # pass
                self.areas2[node['id']] = Area(node['id'], node['corners'], is_highland, height)
        pass  # 暂时未定义区域划分的具体细节

    # 构建拓扑图
    def build_map(self):
        #  access_edges 通行边，attack_edges 攻击边
        for edge in map_cfg['access_edges']:
            self.add_edge(edge['source'], edge['target'], 1)
            pass
        for edge in map_cfg['attack_edges']:
            # print(self.areas1[edge['source']])
            self.add_edge(edge['source'], edge['target'], 2)
            pass

    # 为车辆分配区域
    def assign_vehicle_to_area(self, car_id, car_position):
        # 判断并分配车辆到区域，分配的是车辆的id
        x, y, z = car_position
        for area_id, area in self.areas.items():
            # 如果在高地，先加进去
            if area.is_highland and z > area.height and area.contains_point((x, y)):
                self.add_vehicle_to_area(car_id, area_id)
                # area.vehicles_in_area.append(car_id)
                return  # 车辆已分配至一个区域
        # 如果不符合高地的条件或者不在任何高地区域的扩展区域内，进行普通区域的判断
        for area_id, area in self.areas.items():
            if area.contains_point((x, y)):
                self.add_vehicle_to_area(car_id, area_id)
                # area.vehicles_in_area.append(car_id)
                return  # 车辆已分配至一个区域

    # 更新车辆信息
    def update(self, result):
        self.car_list, self.avail_car_list = result, result
        for car in self.car_list:
            if car.life == 0:
                self.avail_car_list.remove(car)
                self.dead_car_list.append(car)
        for car in self.dead_car_list:
            if car.life > 0:
                self.avail_car_list.append(car)
                self.dead_car_list.remove(car)
        # 对于车辆列表中的每一辆车，如果在pre_list中，就把health -- ，否则直接加入，health为指定值
        pass

    # 需要调用的主方法，参数为车辆列表，在主循环中调用时可以实时赋值
    def run(self, carlist):
        while True:
            self.play_with_car(carlist)
            self.update(carlist)
            pass

    def pre_play_with_car(self, position, id):
        self.define_areas()
        # screen = pygame.display.set_mode((800, 600))

        for area in self.areas1:
            polygon_color = (34, 155, 246)
            polygon_vertices = area.vertices
            # area.draw(self.window, (34, 155, 246), 3)  # 以橙色色线条显示每个区域
            if MathAnalays.contain(position, polygon_vertices):
                area.add_vehicle(id)
                pygame.draw.polygon(self.window, polygon_color, polygon_vertices)

        for area in self.areas2:
            polygon_color = (226, 111, 6)
            polygon_vertices = area.vertices
            # area.draw(self.window, (34, 155, 246), 3)  # 以橙色色线条显示每个区域
            if MathAnalays.contain(position, polygon_vertices):
                pygame.draw.polygon(self.window, polygon_color, polygon_vertices)
                area.add_vehicle(id)
    # 可视化部分
    def play_with_car(self, result):
        # result = Car.CarList.get_map_info()
        for key, value in result:
            car_id = key
            position = value
            # x, y, z = position
            self.assign_vehicle_to_area(car_id, position)

        # 可视化部分
        # 我方区域有车，呈现橙色，敌方区域有车，呈现蓝色，一个区域同时有蓝色和红色车，呈另一个不知道的颜色
        polygon_color_1, polygon_color_2, polygon_color_3 = (34, 155, 246), (226, 111, 6), (120, 120, 120)
        for area in self.areas1.value():
            polygon_vertices = area.vertices
            if area.vehicles_in_area is not None:
                cnt, flag1, flag2 = 0, 0, 0
                for car in area.vehicles_in_area:
                    if car in map_cfg['RedCarsID'] and flag1 == 0:
                        cnt += 1
                    elif car in map_cfg['BlueCarsID'] and flag2 == 0:
                        cnt += 1
                if cnt == 1:
                    pygame.draw.polygon(self.window, polygon_color_1, polygon_vertices)
                else:
                    pygame.draw.polygon(self.window, polygon_color_3, polygon_vertices)
                    pass

        for area in self.areas2.value():
            polygon_vertices = area.vertices
            if area.vehicles_in_area is not None:
                cnt, flag1, flag2 = 0, 0, 0
                for car in area.vehicles_in_area:
                    if car in map_cfg['RedCarsID'] and flag1 == 0:
                        cnt += 1
                    elif car in map_cfg['BlueCarsID'] and flag2 == 0:
                        cnt += 1
                if cnt == 1:
                    pygame.draw.polygon(self.window, polygon_color_2, polygon_vertices)
                else:
                    pygame.draw.polygon(self.window, polygon_color_3, polygon_vertices)
                    pass

    # 可视化部分，绘制区域
    def play(self):
        # 遍历字典 self.areas1 和 self.areas2 中的所有元素
        for key in self.areas1:
            self.areas1[key].draw(self.window, (34, 155, 246), 3)
        for key in self.areas2:
            self.areas2[key].draw(self.window, (226, 111, 6), 3)

    # 输出部分，获取区域信息
    def get_area_info(self, area_id):
        area = self.areas.get(area_id)
        if area is not None:
            return {
                'id': area.id,
                'height': area.height,
                'is_highland': area.is_highland,
                'vertices': area.vertices,
                'vehicles_in_area': area.vehicles_in_area,
                'importance': area.importance,
                'attack_edges': area.attack_edges,
                'access_edges': area.access_edges
            }
        else:
            return None

    # 输出部分，获取车辆信息
    def get_vehicle_info(self, vehicle_id):
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is not None:
            return {
                'car_id': vehicle.id,
                'area': vehicle[vehicle_id]
            }
        else:
            return None

    def get_edge_info(self, source_id, target_id):
        source_area = self.areas.get(source_id)
        target_area = self.areas.get(target_id)
        if source_area is not None and target_area is not None:
            return {
                'source': self.get_area_info(source_id),
                'target': self.get_area_info(target_id),
                'is_accessible': target_id in source_area.access_edges,
                'is_attackable': target_id in source_area.attack_edges
            }
        else:
            return None

    def get_highland_info(self, area_id):
        area = self.areas.get(area_id)
        if area is not None and area.is_highland:
            return {
                'id': area.id,
                'height': area.height,
                'vertices': area.vertices,
                'extended_vertices': area.extended_vertices
            }
        else:
            return None


    def get_all_area_info(self):
        return {area_id: self.get_area_info(area_id) for area_id in self.areas.keys()}

    def get_all_highland_info(self):
        return {area_id: self.get_highland_info(area_id) for area_id, area in self.areas.items() if area.is_highland}

    def get_all_vehicle_info(self):
        return {vehicle_id: self.get_vehicle_info(vehicle_id) for vehicle_id in self.vehicles.keys()}

    # 添加边的工具方法
    def add_edge(self, source: object, target: object, Type=0):
        if Type == 1:
            # print(self.areas1[source])
            self.areas[source].access_edges.append(target)
            pass
        else:
            # print(self.areas2[source])
            self.areas[source].attack_edges.append(target)
            pass

    # 获取角点的工具方法
    def get_area_corners(self, areas_id):  # 获取角点
        return self.areas[areas_id].get('corners', [])

    # """"""""""""""""""""""""""""""""""""""""""""""""

    def get_vehicle_in_area(self, area_id):
        area = self.areas.get(area_id)
        if area and area.vehicles_in_area:
            return area.vehicles_in_area
        return -1

    # 添加车辆的方法现在需要同时更新车辆实例和区域内的车辆清单
    def add_vehicle_to_area(self, vehicle_id, area_id):
        if area_id in self.areas and vehicle_id not in self.vehicles:
            self.areas[area_id].vehicles_in_area.append(vehicle_id)
            self.vehicles[vehicle_id] = area_id

    # 实现区域内车辆的移除也需要同步更新
    def remove_vehicle_from_area(self, vehicle_id, area_id):
        if area_id in self.areas and vehicle_id in self.vehicles:
            self.areas[area_id].vehicles_in_area = [v for v in self.areas[area_id].vehicles_in_area if
                                                    v.id != vehicle_id]
            del self.vehicles[vehicle_id]
