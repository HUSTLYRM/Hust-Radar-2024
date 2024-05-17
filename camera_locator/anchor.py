import cv2

from camera_locator.point_picker import PointsPicker


class Anchor:
    def __init__(self):
        self.vertexes = [[0, 0] for _ in range(4)]
        self.idx = 0

    def append(self, point):
        if 0 <= self.idx < 4:
            self.vertexes[self.idx] = point
            self.idx += 1
            return True
        else:
            return False

    def pop(self):
        if self.idx > 0:
            self.idx -= 1
            return self.vertexes[self.idx]
        else:
            return None

    def __getitem__(self, item):
        if 0 <= item < self.idx:
            return self.vertexes[item]
        else:
            # raise
            print("idx exceed bound")
            return None

    def __len__(self):
        return self.idx
        # len(self.vertexes) is always 4

    def clear(self):
        while self.idx > 0:
            self.pop()


# vertexes used when using landmark R0/B0
# self.anchor = None
def set_by_hand(image, obj):
    if isinstance(obj, Anchor):
        pp = PointsPicker()
        pp.caller(image, obj)
        while len(obj) < 4:
            pp.resume(obj)

        cv2.waitKey(1000)
        print('final')
        print(obj.vertexes)
    else:
        pass


def set_by_detector():
    pass


