import cv2

class MySize:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def get(self):
        return self.w, self.h

    def copy(self):
        obj = MySize(self.w, self.h)
        return obj

    def __getitem__(self, item):
        if item == 0:
            return self.w
        elif item == 1:
            return self.h
        else:
            return None

    def __lt__(self, other):
        return self.w < other.w and self.h < other.h

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __eq__(self, other):
        return self.w == other.w and self.h == other.h

    def __ne__(self, other):
        pass

    def __gt__(self, other):
        return self.w > other.w and self.h > other.h

    def __ge__(self, other):
        pass



class PointsPicker:
    def __init__(self):
        self.image_to_display = None
        self.image_to_display_copy = None
        self.window_size = None
        self.raw_image = None
        self.zoomed_image = None
        self.points_to_display = []

        self.window = "set anchor"
        self.window_position = [0, 0]  # pixel coordinate of window w.r.t reshaped image
        self.window_position_copy = self.window_position.copy()
        self.leftbutton_down_position = [0, 0]
        self.leftbutton_pushing_position = [0, 0]

        self.zoom_ratio = 1
        self.zoom_stride = 0.1

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.leftbutton_down_position = [x, y]
            # note that deep copy needed
            self.window_position_copy = self.window_position.copy()
            self.display()
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):
            pointer_position = [x, y]
            self.move(pointer_position)
            self.display()
        elif event == cv2.EVENT_MOUSEWHEEL:
            former_zoom_ratio = self.zoom_ratio
            self.update_zoom_ratio(flags)
            self.zoom(x, y, former_zoom_ratio)
            self.display()
        elif event == cv2.EVENT_LBUTTONDOWN:
            rightbutton_down_position = [x, y]
            corresponding_point = [((x + self.window_position[0]) / self.zoom_ratio),
                                   ((y + self.window_position[1]) / self.zoom_ratio)]
            quantified_point = [int(corresponding_point[0]),
                                int(corresponding_point[1])]
            self.points_to_display.append(corresponding_point)
            param[0].append(quantified_point)
            self.display()
            pass
        else:
            cv2.imshow(self.window, self.image_to_display)
            pass

    def display(self):
        for point in self.points_to_display:
            point_on_display = (int(point[0] * self.zoom_ratio - self.window_position[0]),
                                int(point[1] * self.zoom_ratio - self.window_position[1]))
            msg = '(x:' + str(int(point[0])) + ',y:' + str(int(point[1])) + ')'
            cv2.circle(self.image_to_display,
                       point_on_display,
                       1,
                       (0, 0, 255),
                       thickness=-1)
            cv2.putText(self.image_to_display,
                        msg,
                        point_on_display,
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0,
                        (0, 0, 255),
                        thickness=1)
        cv2.imshow(self.window, self.image_to_display)

    # 矫正窗口在图片中的位置
    def check_location(self, image_size):
        for i in range(2):
            if self.window_position[i] + self.window_size[i] > image_size[i] > self.window_size[i]:
                self.window_position[i] = image_size[i] - self.window_size[i]
            elif self.window_position[i] + self.window_size[i] > image_size[i] and image_size[i] < self.window_size[i]:
                self.window_position[i] = 0
            if self.window_position[i] < 0:
                self.window_position[i] = 0

    # flag: up/down, stride: , ratio:
    def update_zoom_ratio(self, flag):
        # up: zoom in
        if flag > 0:
            temp = self.zoom_ratio + self.zoom_stride
            if temp > 4:  # 1 + self.zoom_stride * 20:  smooth
                self.zoom_ratio = 4  # 1 + self.zoom_stride * 20
            else:
                self.zoom_ratio = temp
        # down: zoom out
        else:
            temp = self.zoom_ratio - self.zoom_stride
            if temp < 1:
                self.zoom_ratio = 1
            else:
                self.zoom_ratio = temp
        self.zoom_ratio = round(self.zoom_ratio, 2)

    def zoom(self, x, y, former_zoom_ratio):
        zoomed_image_size = MySize(int(self.raw_image.shape[1] * self.zoom_ratio),
                                   int(self.raw_image.shape[0] * self.zoom_ratio))

        self.zoomed_image = cv2.resize(self.raw_image,
                                       (zoomed_image_size.w, zoomed_image_size.h),
                                       interpolation=cv2.INTER_AREA)
        new_size = None
        if zoomed_image_size <= self.window_size:
            new_size = zoomed_image_size.copy()
        elif zoomed_image_size > self.window_size:
            new_size = self.window_size.copy()

        cv2.resizeWindow(self.window, new_size.w, new_size.h)
        self.window_position = [int((self.window_position[0] + x) * self.zoom_ratio / former_zoom_ratio - x),
                                int((self.window_position[1] + y) * self.zoom_ratio / former_zoom_ratio - y)]

        self.check_location(zoomed_image_size)
        self.image_to_display = self.zoomed_image[self.window_position[1]:self.window_position[1] + new_size.h,
                                                  self.window_position[0]:self.window_position[0] + new_size.w]
        self.image_to_display_copy = self.image_to_display.copy()

    def move(self, pointer_position):
        zoomed_h, zoomed_w = self.zoomed_image.shape[0:2]
        window_w, window_h = self.window_size.get()
        show_size = None
        if zoomed_w <= window_w and zoomed_h <= window_h:
            show_size = MySize(zoomed_w, zoomed_h)
            self.window_position = [0, 0]
        elif zoomed_w > window_w and zoomed_h < window_h:
            show_size = MySize(window_w, zoomed_h)
            self.window_position[0] = self.window_position_copy[0] + self.leftbutton_down_position[0] - \
                                      pointer_position[0]
        elif zoomed_w < window_w and zoomed_h > window_h:
            show_size = MySize(zoomed_w, window_h)
            self.window_position[1] = self.window_position_copy[1] + self.leftbutton_down_position[1] - \
                                      pointer_position[1]
        else:
            show_size = MySize(window_w, window_h)
            self.window_position[0] = self.window_position_copy[0] + self.leftbutton_down_position[0] - \
                                      pointer_position[0]
            self.window_position[1] = self.window_position_copy[1] + self.leftbutton_down_position[1] - \
                                      pointer_position[1]
        self.check_location(MySize(zoomed_w, zoomed_h))
        self.image_to_display = self.zoomed_image[self.window_position[1]:self.window_position[1] + show_size.h,
                                                  self.window_position[0]:self.window_position[0] + show_size.w]
        self.image_to_display_copy = self.image_to_display.copy()

    def init_frame(self, image):
        self.raw_image = image
        print(self.raw_image.shape)
        self.window_size = MySize(self.raw_image.shape[1], self.raw_image.shape[0])

        self.zoomed_image = self.raw_image.copy()
        self.image_to_display = self.raw_image[self.window_position[1]:self.window_position[1] + self.window_size.h,
                                               self.window_position[0]:self.window_position[0] + self.window_size.w]
        self.image_to_display_copy = self.image_to_display.copy()

    def caller(self, image, anchor):
        self.init_frame(image)

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.window_size.w, self.window_size.h)
        cv2.moveWindow(self.window, 200, 100)

        cv2.setMouseCallback(self.window, self.mouse_callback, (anchor, ))
        # note that waitKey is needed
        while True:
            key = cv2.waitKey(400)
            print(key)
            if key == 100 or key == 68:  # d
                anchor.pop()
                self.points_to_display.pop()
                self.image_to_display = self.image_to_display_copy.copy()
                self.display()
                print('deleted point')
            if key == 113 or key == 81:  # q
                break
        cv2.destroyAllWindows()

    def resume(self, anchor):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.window_size.w, self.window_size.h)
        cv2.moveWindow(self.window, 200, 100)

        cv2.setMouseCallback(self.window, self.mouse_callback, (anchor,))
        while True:
            key = cv2.waitKey(400)
            print(key)
            if key == 100 or key == 68:
                anchor.pop()
                self.points_to_display.pop()
                self.image_to_display = self.image_to_display_copy.copy()
                self.display()
                print('deleted point')
            if key == 113 or key == 81:
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    from anchor import Anchor
    _image = cv2.imread('./0.jpg')
    _anchor = Anchor()

    pp = PointsPicker()
    pp.caller(_image, _anchor)
    while len(_anchor) < 4:
        pp.resume(_anchor)

    cv2.waitKey(1000)
    print('final')
    print(_anchor.vertexes)
